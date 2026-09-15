"""Web dashboard: live-visualizes walk-forward training and Monte Carlo
simulation, and holds the start/stop controls for both.

Both processes run as background threads inside this same server process,
each with its own threading.Event the Stop button sets -- the training
loop and the simulation generator both check it between steps (see
train.run_walk_forward / simulate.run_monte_carlo), so a stop takes effect
within one day-step or one forecast-day, not instantly but quickly, and
always checkpoints/exits cleanly rather than being killed mid-write.

The dashboard itself polls a couple of small JSON status endpoints on a
timer (see templates/dashboard.html) rather than using server-sent events
or websockets -- simpler to reason about correctly on Flask's built-in
dev server, and at roughly one poll per second the result is
indistinguishable from "live" for a process that updates once per
simulated day anyway.
"""
from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, jsonify, render_template, request

from . import inventory, simulate, train
from .config import DAILY_VARIABLES

log = logging.getLogger(__name__)

METRICS_BUFFER_SIZE = 5000  # in-memory chart history this session; older backlog comes from disk on first load
SIM_FRAME_PACE_SEC = 0.12  # purely cosmetic pacing between forecast days so the ensemble visibly "builds up"


def _decimate(records: list[dict], target: int = 1000) -> list[dict]:
    if len(records) <= target:
        return records
    step = len(records) / target
    return [records[int(i * step)] for i in range(target)]


class TrainingManager:
    """Owns the single walk-forward training thread for one store. Only
    one can run at a time (see train.TrainingLock, which also enforces
    this across separate processes, e.g. a headless `weatherbot train`
    accidentally started alongside the dashboard)."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status: dict = {"phase": "idle", "message": "Not started this session."}
        self._metrics_buffer: deque = deque(maxlen=METRICS_BUFFER_SIZE)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("Training is already running.")
            self._stop_event.clear()
            self._status = {"phase": "starting", "message": "Starting..."}
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _on_progress(self, progress: train.TrainingProgress) -> None:
        with self._lock:
            self._status = {
                "phase": progress.phase,
                "current_date": progress.current_date,
                "total_days_known": progress.total_days_known,
                "days_processed_total": progress.days_processed_total,
                "message": progress.message,
                "latest_metrics": progress.latest_metrics,
            }
            if progress.latest_metrics is not None and progress.latest_metrics.get("mae") is not None:
                self._metrics_buffer.append({
                    "date": progress.current_date,
                    "day_idx": progress.days_processed_total - 1,
                    "mae": progress.latest_metrics["mae"],
                    "rmse": progress.latest_metrics["rmse"],
                    "per_variable_mae": progress.latest_metrics.get("per_variable_mae", {}),
                })

    def _run(self) -> None:
        try:
            train.run_walk_forward(self.store_path, on_progress=self._on_progress,
                                    should_stop=self._stop_event.is_set)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the dashboard, not just logged
            log.exception("walk-forward training crashed")
            with self._lock:
                self._status = {"phase": "error", "message": f"Training crashed: {exc}"}

    def status(self) -> dict:
        with self._lock:
            status = dict(self._status)
        status["running"] = self.is_running()
        if status.get("phase") in (None, "idle"):
            disk_state = train.read_training_state(self.store_path)
            if disk_state is not None:
                status["message"] = (
                    f"Last checkpoint: day {disk_state.get('current_day_idx', 0):,}, "
                    f"as of {disk_state.get('updated_utc', '?')} (not running this session)."
                )
        return status

    def recent_metrics(self, limit: int = 1000) -> list[dict]:
        with self._lock:
            buf = list(self._metrics_buffer)
        if not buf:
            buf = train.read_recent_metrics(self.store_path, limit=limit)
        return _decimate(buf, target=limit)


class SimulationManager:
    """Owns the single Monte Carlo simulation thread for one store. A new
    run cannot start while one is in progress -- stop it first -- so the
    dashboard only ever needs to track one ensemble at a time."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frames: list[dict] = []
        self._done = True
        self._error: str | None = None
        self._meta: dict = {}

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, point_idx: int, point_info: dict, start_date: _dt.date,
              variables: list[str], horizon_days: int, n_paths: int) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("A simulation is already running. Stop it first.")
            if not train.model_exists(self.store_path):
                raise RuntimeError("No trained model yet -- start walk-forward training first.")
            self._stop_event.clear()
            self._frames = []
            self._done = False
            self._error = None
            self._meta = {
                "point": point_info,
                "variables": variables,
                "horizon_days": horizon_days,
                "n_paths": n_paths,
                "start_date": start_date.isoformat(),
                "warning": None,
            }
            self._thread = threading.Thread(
                target=self._run, args=(point_idx, start_date, variables, horizon_days, n_paths), daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self, point_idx, start_date, variables, horizon_days, n_paths) -> None:
        try:
            model = train.load_model(self.store_path)
            min_obs = simulate.min_observations(model, point_idx, variables)
            if min_obs < simulate.MIN_OBSERVATIONS_FOR_CONFIDENCE:
                with self._lock:
                    self._meta["warning"] = (
                        f"Only {min_obs} training observation(s) so far at this point for the "
                        "selected variable(s) -- the model is still warming up, so this forecast "
                        "may be unstable."
                    )
            for frame in simulate.run_monte_carlo(
                model, point_idx, start_date, variables, horizon_days, n_paths,
                should_stop=self._stop_event.is_set,
            ):
                frame_dict = {
                    "day_offset": frame.day_offset,
                    "date": frame.date,
                    "paths": {v: np.round(vals, 3).tolist() for v, vals in frame.paths.items()},
                    "percentiles": frame.percentiles,
                    "done": frame.done,
                }
                with self._lock:
                    self._frames.append(frame_dict)
                time.sleep(SIM_FRAME_PACE_SEC)
            with self._lock:
                self._done = True
        except Exception as exc:  # noqa: BLE001 -- surfaced to the dashboard, not just logged
            log.exception("monte carlo simulation crashed")
            with self._lock:
                self._error = str(exc)
                self._done = True

    def status(self, since: int = 0) -> dict:
        with self._lock:
            return {
                "running": self.is_running(),
                "done": self._done,
                "error": self._error,
                "meta": dict(self._meta),
                "frames": list(self._frames[since:]),
                "total_frames": len(self._frames),
            }


def create_app(store_path: str) -> Flask:
    app = Flask(__name__)

    ds = inventory.open_store(store_path)
    lats = ds["lat"].values.astype(float)
    lons = ds["lon"].values.astype(float)
    point_ids = np.arange(ds.sizes["point"])
    store_variables = [v for v in DAILY_VARIABLES if v in ds.data_vars]
    time_min = str(ds["time"].values.min())[:10]
    time_max = str(ds["time"].values.max())[:10]
    ds.close()

    training = TrainingManager(store_path)
    simulation = SimulationManager(store_path)

    @app.get("/")
    def dashboard():
        return render_template(
            "dashboard.html",
            store_path=store_path,
            n_points=len(point_ids),
            variables=store_variables,
            time_min=time_min,
            time_max=time_max,
            default_horizon=simulate.DEFAULT_HORIZON_DAYS,
            max_horizon=simulate.MAX_HORIZON_DAYS,
            default_paths=simulate.DEFAULT_N_PATHS,
            max_paths=simulate.MAX_N_PATHS,
        )

    @app.get("/api/store/summary")
    def store_summary():
        return jsonify({
            "store_path": store_path,
            "n_points": len(point_ids),
            "variables": store_variables,
            "time_min": time_min,
            "time_max": time_max,
        })

    @app.get("/api/points")
    def points():
        return jsonify([
            {"point_id": int(pid), "lat": float(lat), "lon": float(lon)}
            for pid, lat, lon in zip(point_ids, lats, lons)
        ])

    @app.get("/api/training/status")
    def training_status():
        return jsonify(training.status())

    @app.get("/api/training/metrics")
    def training_metrics():
        limit = request.args.get("limit", default=1000, type=int)
        return jsonify(training.recent_metrics(limit=max(10, min(limit, 5000))))

    @app.post("/api/training/start")
    def training_start():
        try:
            training.start()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"ok": True})

    @app.post("/api/training/stop")
    def training_stop():
        training.stop()
        return jsonify({"ok": True})

    @app.post("/api/simulate/start")
    def simulate_start():
        body = request.get_json(force=True, silent=True) or {}
        chosen_vars = [v for v in body.get("variables", []) if v in store_variables]
        if not chosen_vars:
            return jsonify({"error": "Select at least one known variable."}), 400

        point_id = body.get("point_id")
        if point_id is not None:
            try:
                point_idx = int(point_id)
            except (TypeError, ValueError):
                return jsonify({"error": "point_id must be an integer."}), 400
            if not (0 <= point_idx < len(point_ids)):
                return jsonify({"error": f"point_id out of range 0..{len(point_ids) - 1}."}), 400
        else:
            try:
                lat = float(body["lat"])
                lon = float(body["lon"])
            except (KeyError, TypeError, ValueError):
                return jsonify({"error": "Provide either point_id, or both lat and lon."}), 400
            point_idx = simulate.nearest_point_idx(lats, lons, lat, lon)

        horizon_days = int(body.get("horizon_days", simulate.DEFAULT_HORIZON_DAYS))
        n_paths = int(body.get("n_paths", simulate.DEFAULT_N_PATHS))
        point_info = {"point_id": int(point_idx), "lat": float(lats[point_idx]), "lon": float(lons[point_idx])}

        try:
            simulation.start(
                point_idx, point_info, _dt.date.fromisoformat(time_max), chosen_vars, horizon_days, n_paths,
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"ok": True, "point": point_info})

    @app.get("/api/simulate/status")
    def simulate_status():
        since = request.args.get("since", default=0, type=int)
        return jsonify(simulation.status(since=max(0, since)))

    @app.post("/api/simulate/stop")
    def simulate_stop():
        simulation.stop()
        return jsonify({"ok": True})

    return app
