"""Orchestrates fetch -> downcast -> zarr write, in memory-safe batches.

Two entry points: run_test_mode() and run_full_mode(). Both share the same
batch loop; they differ only in point-set size and time range/chunking.

This module has no console/GUI output of its own — run_pipeline() reports
progress through an optional on_progress(step, total, message) callback
instead of printing, and failure reporting is exposed as pure helpers
(format_failure_summary, write_failure_log) so both the CLI and the GUI can
format/display results their own way. This also matters for the GUI
specifically: it runs under pythonw.exe, where sys.stdout/stderr are None, so
anything in this module that unconditionally printed would crash it.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import fetch, grid, store
from .config import Config, TEST_MODE_POINTS, TEST_MODE_YEARS

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def _daily_row(daily: dict, variable: str, expected_len: int) -> np.ndarray | None:
    values = daily.get(variable)
    if values is None or len(values) != expected_len:
        return None
    return np.array([np.nan if v is None else v for v in values], dtype=np.float32)


def _iter_time_chunks(start_date: _dt.date, end_date: _dt.date, years_per_chunk: int):
    cur = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(years=years_per_chunk) - pd.Timedelta(days=1), end_ts)
        yield cur.date(), chunk_end.date()
        cur = chunk_end + pd.Timedelta(days=1)


def _iter_point_batches(points: pd.DataFrame, batch_size: int):
    for start in range(0, len(points), batch_size):
        yield points.iloc[start:start + batch_size]


def run_pipeline(
    config: Config,
    points: pd.DataFrame,
    start_date: _dt.date,
    end_date: _dt.date,
    years_per_time_chunk: int,
    on_progress: Optional[ProgressCallback] = None,
) -> dict:
    """Runs the full batch loop for a given point set and date range.

    on_progress(step, total_steps, message), if given, is called after each
    fetched+written batch. Returns a summary dict:
    {n_points, n_batches, failures: [...]}.
    """
    full_time_index = store.build_time_index(config, end_date)
    store.init_store(config.store_path, points, full_time_index, config)

    session = fetch.build_session(config)
    limiter = fetch.RateLimiter(config.rate_limit_per_sec)

    failures: list[dict] = []
    point_batches = list(_iter_point_batches(points, config.batch_size))
    time_chunks = list(_iter_time_chunks(start_date, end_date, years_per_time_chunk))
    total_steps = len(point_batches) * len(time_chunks)
    step = 0

    for batch in point_batches:
        point_start = int(batch["point_id"].iloc[0])
        point_stop = int(batch["point_id"].iloc[-1]) + 1
        point_slice = slice(point_start, point_stop)

        for chunk_start, chunk_end in time_chunks:
            n_days = (chunk_end - chunk_start).days + 1
            time_offset = store.date_to_offset(chunk_start, config)
            time_slice = slice(time_offset, time_offset + n_days)

            results = fetch.fetch_batch(batch, chunk_start, chunk_end, config, session, limiter)

            data: dict[str, np.ndarray] = {
                var: np.full((len(batch), n_days), np.nan, dtype=np.float32)
                for var in config.variables
            }
            for row_idx, result in enumerate(results):
                if result.error is not None or result.daily is None:
                    failures.append({
                        "point_id": result.point_id,
                        "lat": result.lat,
                        "lon": result.lon,
                        "time_range": f"{chunk_start.isoformat()}..{chunk_end.isoformat()}",
                        "reason": result.error or "no data",
                        "timestamp": _dt.datetime.utcnow().isoformat(),
                    })
                    continue
                row_ok = True
                row_data = {}
                for var in config.variables:
                    row = _daily_row(result.daily, var, n_days)
                    if row is None:
                        row_ok = False
                        break
                    row_data[var] = row
                if not row_ok:
                    failures.append({
                        "point_id": result.point_id,
                        "lat": result.lat,
                        "lon": result.lon,
                        "time_range": f"{chunk_start.isoformat()}..{chunk_end.isoformat()}",
                        "reason": "response length mismatch or missing variable",
                        "timestamp": _dt.datetime.utcnow().isoformat(),
                    })
                    continue
                for var, row in row_data.items():
                    data[var][row_idx, :] = row

            store.write_region(config.store_path, data, point_slice, time_slice)
            step += 1
            if on_progress is not None:
                on_progress(
                    step, total_steps,
                    f"points {point_slice.start}-{point_slice.stop - 1}, "
                    f"{chunk_start.isoformat()}..{chunk_end.isoformat()}",
                )
            else:
                log.info("batch %d/%d done (points %d-%d, %s..%s)",
                          step, total_steps, point_slice.start, point_slice.stop - 1,
                          chunk_start.isoformat(), chunk_end.isoformat())

    store.finalize_store(config.store_path)
    return {"n_points": len(points), "n_batches": total_steps, "failures": failures}


def format_failure_summary(summary: dict) -> str:
    failures = summary["failures"]
    if not failures:
        return "All points fetched successfully — no failures."

    unique_points = {f["point_id"] for f in failures}
    lines = [
        f"{len(unique_points)} point(s) had fetch failures "
        f"({len(failures)} failed batch/chunk attempts):",
    ]
    for f in failures[:20]:
        lines.append(f"  point {f['point_id']:>6}  ({f['lat']:.3f}, {f['lon']:.3f})  "
                      f"[{f['time_range']}]  {f['reason']}")
    if len(failures) > 20:
        lines.append(f"  ... and {len(failures) - 20} more (see failure log)")
    return "\n".join(lines)


def write_failure_log(summary: dict, store_path: str) -> Optional[str]:
    """Writes the full failure list to a JSON file next to the store.
    Returns the file path, or None if there were no failures."""
    failures = summary["failures"]
    if not failures:
        return None
    log_path = f"{store_path}.failures_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(log_path, "w") as fh:
        json.dump(failures, fh, indent=2)
    return log_path


def run_test_mode(config: Config, on_progress: Optional[ProgressCallback] = None) -> dict:
    points = grid.generate_test_grid(config, TEST_MODE_POINTS)
    end_date = config.end_date()
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=TEST_MODE_YEARS)).date()
    return run_pipeline(config, points, start_date, end_date,
                         years_per_time_chunk=config.time_chunk_years, on_progress=on_progress)


def run_full_mode(config: Config, on_progress: Optional[ProgressCallback] = None) -> dict:
    points = grid.generate_full_grid(config)
    end_date = config.end_date()
    return run_pipeline(
        config, points, config.archive_start_date, end_date,
        years_per_time_chunk=config.time_chunk_years, on_progress=on_progress,
    )
