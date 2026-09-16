"""Continuous walk-forward training over the existing Zarr archive.

Walks forward through the store's time index one calendar day at a time,
updating the OnlineModel (see model.py) with that day's real observations
across every grid point and variable in a single vectorized step, and
recording a compact error-metric point after each day for the training
progress chart. That's the walk-forward part: at every step, the model is
scored on a day it has not yet trained on (an honest, out-of-sample error),
*then* updated with that day's answer before moving to the next one --
exactly the "expanding window" validation style used for financial time
series, just applied to grid points and weather variables instead of
tickers and returns.

Historical backlog is read from the store in multi-week blocks (a
double-digit number of MB at a time) rather than one xarray call per day,
to keep I/O reasonable on a small droplet. Once the walk reaches the most
recent day the store actually has, it switches to a slow poll loop and
resumes automatically if the store ever grows (a fresh/larger full pull
against the same path) -- by design there is no fixed end date, only a
manual stop. Progress (current day + model weights) is checkpointed
periodically so training survives a restart without re-walking from 1990.

Known limitation (shared with the ingestion phase, see README): a store's
time extent is fixed at creation. If nothing ever re-runs `fetch --mode
full` with a later end date, "caught up" here just means caught up to
whatever the store already had -- the loop will sit in live_wait polling
for a store that never grows, which is correct behavior, just not
magically fetching new days on its own.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import inventory
from .config import DAILY_VARIABLES
from .model import OnlineModel

log = logging.getLogger(__name__)

# Backlog is read from the store this many days at a time. Sized against
# the CURRENT grid scope (~17,000 CONUS points x 18 variables at 0.25deg
# spacing): 21 days x ~17,300 points x 18 vars x 4 bytes (float32) is
# ~25MB resident at a time, freed after each block -- the same target this
# was originally sized to at the previous (0.5deg, ~4,300-point) scope, just
# with a shorter block since there are ~4x as many points now. If the grid
# ever changes again, rescale this so n_points * BLOCK_DAYS stays roughly
# constant (~360,000) to hold peak memory steady on a small droplet.
BLOCK_DAYS = 21
CHECKPOINT_EVERY_STEPS = 200  # during backlog catch-up; live mode checkpoints every step (cheap, ~1/day)
LIVE_POLL_INTERVAL_SEC = 300  # how often to re-check the store for new data once caught up
METRICS_MAX_ROWS = 10_000  # bound the on-disk metrics log for a process meant to run forever


def _model_path(store_path: str) -> str:
    return f"{store_path}.model.npz"


def _state_path(store_path: str) -> str:
    return f"{store_path}.train_state.json"


def _metrics_path(store_path: str) -> str:
    return f"{store_path}.train_metrics.jsonl"


def _lock_path(store_path: str) -> str:
    return f"{store_path}.train.lock"


class TrainingLock:
    """A plain lock file so two training processes never run against the
    same store's checkpoint at once (they'd race writing the same npz/json
    files). Deliberately simple -- no PID/liveness checking, which isn't
    portable across the Windows dev machine and Linux droplet this project
    runs on -- so a process that dies uncleanly leaves a stale lock behind;
    the error message says how to clear it.
    """

    def __init__(self, store_path: str):
        self.path = _lock_path(store_path)
        self._acquired = False

    def acquire(self) -> None:
        if os.path.exists(self.path):
            raise RuntimeError(
                f"A training lock already exists at {self.path}. Either another "
                "training process is running against this store, or a previous "
                "one didn't shut down cleanly. Stop it first, or delete the lock "
                "file if you're sure nothing is running."
            )
        with open(self.path, "w") as fh:
            fh.write(str(os.getpid()))
        self._acquired = True

    def release(self) -> None:
        if self._acquired and os.path.exists(self.path):
            os.remove(self.path)
        self._acquired = False


@dataclass
class TrainingProgress:
    phase: str  # "catching_up" | "live_wait" | "stopped"
    current_date: str
    total_days_known: int
    days_processed_total: int
    latest_metrics: Optional[dict]
    message: str


ProgressCallback = Callable[[TrainingProgress], None]


def _read_state(store_path: str) -> Optional[dict]:
    path = _state_path(store_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_state(store_path: str, state: dict) -> None:
    with open(_state_path(store_path), "w") as fh:
        json.dump(state, fh)


def _append_metric(store_path: str, record: dict) -> None:
    with open(_metrics_path(store_path), "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _maybe_compact_metrics(store_path: str) -> None:
    """Keeps the metrics log bounded for a process meant to run forever:
    once it passes METRICS_MAX_ROWS, drop every other row so recent history
    stays dense and old history gets progressively coarser, instead of the
    file growing without bound over years of continuous operation."""
    path = _metrics_path(store_path)
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError:
        return
    if len(lines) <= METRICS_MAX_ROWS:
        return
    kept = lines[::2]
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.writelines(kept)
    os.replace(tmp, path)


def read_recent_metrics(store_path: str, limit: int = 1000) -> list[dict]:
    """Tail of the on-disk metrics log, for seeding the dashboard chart on
    page load (before any live progress has arrived this session)."""
    path = _metrics_path(store_path)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        lines = fh.readlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def read_training_state(store_path: str) -> Optional[dict]:
    """Last-checkpointed position, for showing status before training has
    been (re)started this session."""
    return _read_state(store_path)


def model_exists(store_path: str) -> bool:
    return os.path.exists(_model_path(store_path))


def load_model(store_path: str) -> OnlineModel:
    return OnlineModel.load(_model_path(store_path))


def _date_at(time_index: pd.DatetimeIndex, idx: int, total: int) -> str:
    if 0 <= idx < total:
        return time_index[idx].date().isoformat()
    return time_index[-1].date().isoformat() if total > 0 else ""


def _sleep_or_stop(seconds: float, should_stop: Optional[Callable[[], bool]]) -> bool:
    """Sleeps in small increments so a stop request lands quickly rather
    than only after the full poll interval. Returns True if stopped."""
    waited = 0.0
    step = 1.0
    while waited < seconds:
        if should_stop is not None and should_stop():
            return True
        time.sleep(min(step, seconds - waited))
        waited += step
    return should_stop is not None and should_stop()


def run_walk_forward(
    store_path: str,
    on_progress: Optional[ProgressCallback] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    """Runs until should_stop() returns True (checked between every day's
    update, and during the live-mode poll wait) -- there is no other exit
    condition. Safe to call repeatedly against the same store: it resumes
    from the last checkpoint rather than re-walking from the start.
    """
    lock = TrainingLock(store_path)
    lock.acquire()
    try:
        ds = inventory.open_store(store_path)
        n_points = ds.sizes["point"]
        variables = [v for v in DAILY_VARIABLES if v in ds.data_vars]
        time_index = pd.DatetimeIndex(ds["time"].values)
        n_days_total = len(time_index)

        state = _read_state(store_path)
        model_path = _model_path(store_path)
        if (
            state is not None
            and os.path.exists(model_path)
            and state.get("variables") == variables
            and state.get("n_points") == n_points
        ):
            model = OnlineModel.load(model_path)
            day_idx = min(int(state.get("current_day_idx", 0)), n_days_total)
            total_updates = int(state.get("total_updates", 0))
            log.info("resuming walk-forward training at day %d/%d", day_idx, n_days_total)
        else:
            model = OnlineModel(n_points, variables)
            day_idx = 0
            total_updates = 0
            log.info("starting a fresh walk-forward model (%d points x %d variables)",
                      n_points, len(variables))

        steps_since_checkpoint = 0

        def _checkpoint(idx: int) -> None:
            model.save(model_path)
            _write_state(store_path, {
                "current_day_idx": idx,
                "total_updates": total_updates,
                "n_points": n_points,
                "variables": variables,
                "updated_utc": _dt.datetime.utcnow().isoformat(),
            })

        while True:
            if should_stop is not None and should_stop():
                _checkpoint(day_idx)
                if on_progress is not None:
                    on_progress(TrainingProgress(
                        "stopped", _date_at(time_index, day_idx, n_days_total),
                        n_days_total, day_idx, None, "Stopped by user.",
                    ))
                return

            if day_idx >= n_days_total:
                _checkpoint(day_idx)
                if on_progress is not None:
                    on_progress(TrainingProgress(
                        "live_wait", _date_at(time_index, day_idx, n_days_total),
                        n_days_total, day_idx, None,
                        f"Caught up through {time_index[-1].date()}. Waiting for new data...",
                    ))
                if _sleep_or_stop(LIVE_POLL_INTERVAL_SEC, should_stop):
                    _checkpoint(day_idx)
                    if on_progress is not None:
                        on_progress(TrainingProgress(
                            "stopped", _date_at(time_index, day_idx, n_days_total),
                            n_days_total, day_idx, None, "Stopped by user.",
                        ))
                    return
                ds = inventory.open_store(store_path)
                new_time_index = pd.DatetimeIndex(ds["time"].values)
                if len(new_time_index) > n_days_total:
                    time_index = new_time_index
                    n_days_total = len(time_index)
                continue

            # -- backlog catch-up: pull one block of days into memory --
            block_end = min(day_idx + BLOCK_DAYS, n_days_total)
            block_ds = ds.isel(time=slice(day_idx, block_end)).compute()
            block = np.stack([block_ds[v].values for v in variables], axis=-1)  # (point, time_block, var)
            block_ds.close()

            for offset in range(block_end - day_idx):
                if should_stop is not None and should_stop():
                    _checkpoint(day_idx + offset)
                    if on_progress is not None:
                        on_progress(TrainingProgress(
                            "stopped", _date_at(time_index, day_idx + offset, n_days_total),
                            n_days_total, day_idx + offset, None, "Stopped by user.",
                        ))
                    return

                date = time_index[day_idx + offset]
                actual = block[:, offset, :]  # (n_points, n_vars)
                metrics = model.step(date.dayofyear, actual)
                total_updates += 1
                _append_metric(store_path, {
                    "date": date.date().isoformat(),
                    "day_idx": day_idx + offset,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "n_obs": metrics["n_obs"],
                    "per_variable_mae": metrics["per_variable_mae"],
                })

                steps_since_checkpoint += 1
                if steps_since_checkpoint >= CHECKPOINT_EVERY_STEPS:
                    _checkpoint(day_idx + offset + 1)
                    _maybe_compact_metrics(store_path)
                    steps_since_checkpoint = 0

                if on_progress is not None:
                    on_progress(TrainingProgress(
                        "catching_up", date.date().isoformat(), n_days_total,
                        day_idx + offset + 1, metrics,
                        f"day {day_idx + offset + 1:,}/{n_days_total:,}",
                    ))

            day_idx = block_end
            _checkpoint(day_idx)
            _maybe_compact_metrics(store_path)
    finally:
        lock.release()
