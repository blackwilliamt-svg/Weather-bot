"""Orchestrates fetch -> downcast -> zarr write, in memory-safe batches.

Two entry points: run_test_mode() and run_full_mode(). Both share the same
batch loop; they differ only in point-set size and time range/chunking.

This module has no console/GUI output of its own — run_pipeline() reports
progress through an optional on_progress(ProgressInfo) callback instead of
printing, and failure reporting is exposed as pure helpers
(format_failure_summary, write_failure_log) so both the CLI and the GUI can
format/display results their own way. This also matters for the GUI
specifically: it runs under pythonw.exe, where sys.stdout/stderr are None, so
anything in this module that unconditionally printed would crash it.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import fetch, grid, store
from .config import Config, TEST_MODE_POINTS, TEST_MODE_YEARS

log = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    step: int
    total_steps: int
    message: str
    bytes_downloaded: int  # cumulative, across the whole run so far
    elapsed_sec: float
    eta_sec: Optional[float]  # None until at least one step has completed
    failures_so_far: int = 0  # points/batches skipped so far this run


ProgressCallback = Callable[[ProgressInfo], None]


# -- cross-process coordination -------------------------------------------
#
# A fetch can be started three different ways against the same store: the
# local GUI, the CLI (including the systemd-run droplet service), and now
# the web dashboard's own "Droplet Pull" tab (webapp.py) -- and any of
# those could be running unattended for a long time. Two small file-based
# mechanisms keep them from stepping on each other without needing the
# dashboard to reach across process boundaries (signals, PIDs) to a
# process it may not have started:
#
# - FetchLock: mutual exclusion, so a second Start doesn't launch a
#   competing writer against the same store while one is already running.
# - request_stop()/stop_requested(): a cooperative halt flag any running
#   fetch checks between batches, regardless of which process started it
#   -- this is what lets the dashboard's Stop button halt a systemd-owned
#   pull cleanly without ever sending it a signal directly.

FETCH_LOCK_STALE_SEC = 600  # see FetchLock docstring for why 10 minutes


def _fetch_lock_path(store_path: str) -> str:
    return f"{store_path}.fetch.lock"


class FetchLock:
    """Cross-process mutual exclusion for a running fetch against one
    store -- deliberately staleness-aware, unlike train.TrainingLock's
    plain existence check. Fetch's whole design promise is "resumable
    after ANY interruption, no exceptions" (crash, OOM kill, power loss,
    `systemctl stop`), so a lock that could be left behind forever by an
    unclean death and then require someone to find and delete a file by
    hand would quietly break that promise. Instead, the lock file's mtime
    is a heartbeat -- refreshed once per completed batch by run_pipeline
    -- and the lock is only considered held while that heartbeat is
    younger than FETCH_LOCK_STALE_SEC (10 minutes: comfortably longer than
    the worst realistic single-batch stall under network retries/backoff,
    short enough that a genuine crash self-heals well within a normal
    systemd restart cycle). A stale lock is silently reclaimed by whoever
    next calls acquire() -- exactly like resuming after any other kind of
    interruption, no manual cleanup ever required.
    """

    def __init__(self, store_path: str):
        self.path = _fetch_lock_path(store_path)
        self._held = False

    def acquire(self) -> None:
        if os.path.exists(self.path):
            age = time.time() - os.path.getmtime(self.path)
            if age < FETCH_LOCK_STALE_SEC:
                raise RuntimeError(
                    f"A fetch is already running against this store (last heartbeat {age:.0f}s "
                    "ago) -- likely the systemd service, another dashboard session, or a separate "
                    "CLI run. Stop it first: the dashboard's Stop button (or requesting a stop "
                    "against this store some other way) works regardless of which process started "
                    "it, since it doesn't need to know or signal that process directly."
                )
            # Stale: the previous owner died without releasing it. Reclaiming it is the
            # self-healing case this class exists for, not an error.
            log.info("reclaiming stale fetch lock at %s (heartbeat %.0fs old)", self.path, age)
        self.touch()
        self._held = True

    def touch(self) -> None:
        with open(self.path, "w") as fh:
            fh.write(str(os.getpid()))

    def release(self) -> None:
        if self._held and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        self._held = False


def _stop_request_path(store_path: str) -> str:
    return f"{store_path}.fetch_stop_request"


def request_stop(store_path: str) -> None:
    """Cooperative, cross-process halt signal for a running fetch against
    this store. run_pipeline checks for this file before every batch (in
    addition to any in-process should_stop callback), so it works whether
    the running fetch is a dashboard-launched background thread or a
    separate systemd/CLI process -- same clean-stop guarantee either way:
    the current in-flight batch finishes and is checkpointed, nothing is
    lost, and it's exactly as resumable as any other interruption."""
    open(_stop_request_path(store_path), "w").close()


def stop_requested(store_path: str) -> bool:
    return os.path.exists(_stop_request_path(store_path))


def _clear_stop_request(store_path: str) -> None:
    try:
        os.remove(_stop_request_path(store_path))
    except OSError:
        pass


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
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Runs the full batch loop for a given point set and date range.

    Resumable: if a matching, unfinished run was interrupted previously (same
    store path, grid, chunking, and — critically — the same start/end date,
    which callers must resolve via a manifest rather than recomputing "today"
    fresh; see run_full_mode), already-completed batches are skipped instead
    of re-fetched, and the store is not re-initialized (which would wipe it).

    should_stop(), if given, is checked before each batch, alongside a
    file-based stop request any *other* process can also make (see
    request_stop()) — either way it's a deliberate stop using the exact same
    checkpoint/resume machinery as an unplanned interruption: whatever's
    already written stays written, and a later run with the same parameters
    picks up right where this one stopped.

    Acquires a FetchLock for config.store_path for the duration of the run,
    so a second run_pipeline call against the same store (from another
    process, e.g. the dashboard vs. the systemd service) fails fast instead
    of both writing to the store at once.

    on_progress(ProgressInfo), if given, is called after each fetched+written
    batch. Returns a summary dict:
    {n_points, n_batches, resumed_from, stopped, bytes_downloaded, elapsed_sec, failures: [...]}.
    """
    lock = FetchLock(config.store_path)
    lock.acquire()
    try:
        return _run_pipeline_locked(
            config, points, start_date, end_date, years_per_time_chunk, lock, on_progress, should_stop,
        )
    finally:
        lock.release()


def _run_pipeline_locked(
    config: Config,
    points: pd.DataFrame,
    start_date: _dt.date,
    end_date: _dt.date,
    years_per_time_chunk: int,
    lock: "FetchLock",
    on_progress: Optional[ProgressCallback],
    should_stop: Optional[Callable[[], bool]],
) -> dict:
    # Any stop request left over from a previous run against this store
    # (normally cleared the moment it's honored, below) shouldn't
    # immediately halt a brand new one.
    _clear_stop_request(config.store_path)

    def _effective_should_stop() -> bool:
        if stop_requested(config.store_path):
            return True
        return should_stop() if should_stop is not None else False

    point_batches = list(_iter_point_batches(points, config.batch_size))
    time_chunks = list(_iter_time_chunks(start_date, end_date, years_per_time_chunk))
    total_steps = len(point_batches) * len(time_chunks)
    grid_fp = store.grid_fingerprint(points)

    manifest = store.read_manifest(config.store_path)
    resumed_from = 0
    if manifest is not None and store.manifest_matches(manifest, config, grid_fp, years_per_time_chunk) \
            and manifest.get("start_date") == start_date.isoformat() \
            and manifest.get("end_date") == end_date.isoformat():
        resumed_from = min(int(manifest.get("completed_steps", 0)), total_steps)

    if resumed_from == 0:
        full_time_index = store.build_time_index(config, end_date)
        store.init_store(config.store_path, points, full_time_index, config)

    # One shared session + rate limiter across every concurrent worker: the
    # limiter's pacing/backoff state must be shared so that a 429 seen by any
    # one worker slows down the whole fleet, not just that worker.
    session = fetch.build_session(config)
    limiter = fetch.AdaptiveRateLimiter(config)

    failures: list[dict] = []
    step = resumed_from
    bytes_downloaded = 0
    start_time = time.monotonic()

    if resumed_from > 0 and on_progress is not None:
        on_progress(ProgressInfo(resumed_from, total_steps,
                                  "resuming previous run...", 0, 0.0, None))
    elif resumed_from > 0:
        log.info("resuming: skipping %d/%d already-completed batches", resumed_from, total_steps)

    flat_steps = [(batch, chunk_start, chunk_end)
                  for batch in point_batches for chunk_start, chunk_end in time_chunks]

    def _process_and_write(idx: int, batch, chunk_start, chunk_end, outcome: fetch.BatchOutcome) -> None:
        nonlocal step, bytes_downloaded
        point_start = int(batch["point_id"].iloc[0])
        point_stop = int(batch["point_id"].iloc[-1]) + 1
        point_slice = slice(point_start, point_stop)

        n_days = (chunk_end - chunk_start).days + 1
        time_offset = store.date_to_offset(chunk_start, config)
        time_slice = slice(time_offset, time_offset + n_days)

        bytes_downloaded += outcome.bytes_downloaded
        data: dict[str, np.ndarray] = {
            var: np.full((len(batch), n_days), np.nan, dtype=np.float32)
            for var in config.variables
        }
        for row_idx, result in enumerate(outcome.results):
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
        step = idx + 1
        store.write_manifest(config.store_path, config, grid_fp, years_per_time_chunk,
                              start_date, end_date, step)
        lock.touch()  # refresh the heartbeat -- see FetchLock

        steps_done_this_session = step - resumed_from
        elapsed_sec = time.monotonic() - start_time
        eta_sec = (elapsed_sec / steps_done_this_session) * (total_steps - step) \
            if steps_done_this_session > 0 else None
        message = (f"points {point_slice.start}-{point_slice.stop - 1}, "
                   f"{chunk_start.isoformat()}..{chunk_end.isoformat()}")
        if on_progress is not None:
            on_progress(ProgressInfo(step, total_steps, message, bytes_downloaded, elapsed_sec, eta_sec,
                                      failures_so_far=len(failures)))
        else:
            log.info("batch %d/%d done (%s) — %.1f MB downloaded so far",
                      step, total_steps, message, bytes_downloaded / 1e6)

    # Fetches run concurrently (bounded by config.concurrency), but writes
    # happen strictly in order on this thread — the pool only ever hides
    # network latency, it never touches the zarr store, so there's no
    # concurrent-write risk even though several batches' data can be
    # in flight (and land in the same zarr chunk file) at once.
    stopped = False
    stop_reason: Optional[str] = None
    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as executor:
        pending: dict[int, Future] = {}
        submit_idx = resumed_from

        def _submit_next() -> None:
            nonlocal submit_idx
            if submit_idx < len(flat_steps):
                b, cs, ce = flat_steps[submit_idx]
                pending[submit_idx] = executor.submit(fetch.fetch_batch, b, cs, ce, config, session, limiter)
                submit_idx += 1

        for _ in range(min(max(1, config.concurrency), total_steps - resumed_from)):
            _submit_next()

        next_write_idx = resumed_from
        while next_write_idx < total_steps:
            if _effective_should_stop():
                stopped = True
                stop_reason = "user"
                _clear_stop_request(config.store_path)  # honored -- don't block the next run
                break  # in-flight futures are awaited (harmlessly) when the pool exits below

            batch, chunk_start, chunk_end = flat_steps[next_write_idx]
            outcome = pending.pop(next_write_idx).result()

            if outcome.rate_limited is not None:
                # An hour/day-scale budget is exhausted: stop the whole run
                # rather than writing NaN + a permanent failure for this and
                # every remaining batch. Nothing for this step (or later
                # ones) is written or checkpointed, so it's fully retried on
                # the next resumed run, exactly like an unplanned stop.
                stopped = True
                stop_reason = f"rate_limit_{outcome.rate_limited}"
                log.warning("%s rate limit hit at batch %d/%d — stopping, resumable later",
                            outcome.rate_limited, next_write_idx + 1, total_steps)
                if on_progress is not None:
                    limit_name = "daily" if outcome.rate_limited == "day" else "hourly"
                    on_progress(ProgressInfo(
                        step, total_steps,
                        f"{limit_name} rate limit hit — stopping (resumable)",
                        bytes_downloaded, time.monotonic() - start_time, None,
                        failures_so_far=len(failures),
                    ))
                break

            _process_and_write(next_write_idx, batch, chunk_start, chunk_end, outcome)
            next_write_idx += 1
            _submit_next()

    store.finalize_store(config.store_path)
    # Deliberately not cleared: keeping the manifest (now at completed_steps
    # == total_steps) makes a finished run idempotent — rerunning with the
    # same parameters recognizes it's already fully fetched and does nothing,
    # rather than wiping and re-fetching a store that didn't need it.
    return {
        "n_points": len(points),
        "n_batches": total_steps,
        "completed_steps": step,
        "resumed_from": resumed_from,
        "already_complete": resumed_from >= total_steps,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "bytes_downloaded": bytes_downloaded,
        "elapsed_sec": time.monotonic() - start_time,
        "failures": failures,
    }


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "estimating..."
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


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


def _status_path(store_path: str) -> str:
    return f"{store_path}.status.json"


def write_status_file(store_path: str, info: ProgressInfo) -> None:
    """Small JSON snapshot for checking a fetch's progress without a
    terminal attached to the process — `cat <store_path>.status.json`, or
    the source the dashboard's "Droplet Pull" tab polls. Written by any
    caller driving run_pipeline via its on_progress callback (the CLI
    fetch command, and the dashboard's own background thread alike), so
    the reading side doesn't need to know which one produced it."""
    status = {
        "step": info.step,
        "total_steps": info.total_steps,
        "percent": round(100 * info.step / info.total_steps, 2) if info.total_steps else 0,
        "bytes_downloaded": info.bytes_downloaded,
        "bytes_downloaded_human": format_bytes(info.bytes_downloaded),
        "elapsed_sec": round(info.elapsed_sec, 1),
        "elapsed_human": format_duration(info.elapsed_sec),
        "eta_sec": info.eta_sec,
        "eta_human": format_duration(info.eta_sec),
        "failures_so_far": info.failures_so_far,
        "message": info.message,
        "updated_utc": _dt.datetime.utcnow().isoformat(),
    }
    try:
        with open(_status_path(store_path), "w") as fh:
            json.dump(status, fh, indent=2)
    except OSError as exc:
        log.warning("could not write status file: %s", exc)


def read_status_file(store_path: str) -> Optional[dict]:
    """The most recent write_status_file() snapshot, or None if a fetch
    has never run against this store (this session or otherwise)."""
    path = _status_path(store_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_latest_failures(store_path: str, limit: int = 200) -> Optional[dict]:
    """The most recent write_failure_log() output for this store (there
    can be several, one per run that had failures — picks the newest by
    filename, which embeds a UTC timestamp). Returns None if there isn't
    one. `limit` caps how many individual failure records come back (the
    file itself is uncapped, matching CLI/GUI behavior) since a dashboard
    panel isn't the place to render thousands of rows."""
    candidates = sorted(glob.glob(f"{store_path}.failures_*.json"))
    if not candidates:
        return None
    path = candidates[-1]
    try:
        with open(path) as fh:
            failures = json.load(fh)
    except (OSError, ValueError):
        return None
    return {
        "path": path,
        "total": len(failures),
        "failures": failures[:limit],
    }


def estimate_run(config: Config, mode: str) -> dict:
    """Cheap, no-network estimate of a run's size/shape, for display before
    starting: point/day/variable counts, number of HTTP requests, and a rough
    compressed-store size range (float32 raw size / an assumed 2-3x zstd
    compression ratio typical for daily weather data)."""
    if mode == "test":
        points = grid.generate_test_grid(config, TEST_MODE_POINTS)
        end_date = config.end_date()
        start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=TEST_MODE_YEARS)).date()
    else:
        points = grid.generate_full_grid(config)
        start_date = config.archive_start_date
        end_date = config.end_date()

    n_points = len(points)
    n_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    n_vars = len(config.variables)
    n_point_batches = -(-n_points // config.batch_size)  # ceil div
    n_time_chunks = len(list(_iter_time_chunks(start_date, end_date, config.time_chunk_years)))
    n_requests = n_point_batches * n_time_chunks

    raw_bytes = n_points * n_days * n_vars * 4
    if mode == "full":
        # Full pull is fixed-pace, sequential (concurrency 1) — see
        # run_full_mode — so this isn't a best case, it's the actual expected
        # duration: n_requests * the fixed per-request interval, plus
        # whatever small overhead individual requests add beyond that.
        min_seconds = n_requests * config.full_pull_request_interval_sec
    else:
        # Test mode fetches fast/adaptively (see run_test_mode); this is only
        # the pacing floor concurrency approaches, not a guaranteed time.
        min_seconds = n_requests / config.rate_limit_per_sec if config.rate_limit_per_sec > 0 else None
    return {
        "n_points": n_points,
        "n_days": n_days,
        "n_requests": n_requests,
        "raw_bytes": raw_bytes,
        "compressed_bytes_low": raw_bytes / 3,
        "compressed_bytes_high": raw_bytes / 2,
        "min_seconds": min_seconds,
    }


def _resolve_end_date(config: Config, points: pd.DataFrame, years_per_time_chunk: int) -> _dt.date:
    """end_date normally means "today minus the archive lag", which drifts
    day to day — fine for a single run, but a resumed run days later must
    reuse whatever end_date the ORIGINAL run committed to, or every batch
    already fetched near the end of the range would mismatch and force a
    full restart. If a matching in-progress manifest exists, reuse its
    end_date; otherwise resolve a fresh one from today."""
    manifest = store.read_manifest(config.store_path)
    if manifest is not None:
        grid_fp = store.grid_fingerprint(points)
        if store.manifest_matches(manifest, config, grid_fp, years_per_time_chunk):
            try:
                return _dt.date.fromisoformat(manifest["end_date"])
            except (KeyError, ValueError):
                pass
    return config.end_date()


def run_test_mode(
    config: Config,
    on_progress: Optional[ProgressCallback] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    points = grid.generate_test_grid(config, TEST_MODE_POINTS)
    end_date = _resolve_end_date(config, points, config.time_chunk_years)
    start_date = (pd.Timestamp(end_date) - pd.DateOffset(years=TEST_MODE_YEARS)).date()
    return run_pipeline(config, points, start_date, end_date,
                         years_per_time_chunk=config.time_chunk_years,
                         on_progress=on_progress, should_stop=should_stop)


def run_full_mode(
    config: Config,
    on_progress: Optional[ProgressCallback] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Full pull ALWAYS paces at config.full_pull_request_interval_sec with
    concurrency 1, regardless of config.rate_limit_per_sec/concurrency (those
    apply to test mode only). This takes priority over general fetch-speed
    settings by design: Open-Meteo's daily request quota, not the per-minute
    or per-hour ones, is what actually bounds a sustained multi-day pull —
    bursting faster just hits the daily cap sooner and then sits idle until
    it resets, it doesn't finish the pull any sooner overall."""
    points = grid.generate_full_grid(config)
    end_date = _resolve_end_date(config, points, config.time_chunk_years)
    paced_config = config.with_overrides(
        rate_limit_per_sec=1.0 / config.full_pull_request_interval_sec,
        concurrency=1,
    )
    return run_pipeline(
        paced_config, points, config.archive_start_date, end_date,
        years_per_time_chunk=config.time_chunk_years,
        on_progress=on_progress, should_stop=should_stop,
    )
