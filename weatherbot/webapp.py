"""Web dashboard: live-visualizes walk-forward training, Monte Carlo
simulation, and the droplet's full data pull, and holds the start/stop
controls for all three.

Training and simulation run as background threads inside this same server
process, each with its own threading.Event the Stop button sets -- the
training loop and the simulation generator both check it between steps
(see train.run_walk_forward / simulate.run_monte_carlo), so a stop takes
effect within one day-step or one forecast-day, not instantly but quickly,
and always checkpoints/exits cleanly rather than being killed mid-write.

The full-pull tab is different in one important way: the pull it's
showing/controlling is often NOT this process's own thread -- on a
droplet it's normally the separate `weatherbot.service` systemd unit (see
README "Deploying the dashboard on a droplet"), so this dashboard can be
restarted (or never started at all) without interrupting it. Status comes
from the same `<store_path>.status.json` file the CLI already writes
(pipeline.read_status_file), and Stop works via a cooperative file flag
(pipeline.request_stop) rather than an in-process Event, specifically so
it can halt that separate process too, cleanly, without this dashboard
ever needing to know its PID or send it a signal. See FetchManager and
pipeline.FetchLock/request_stop for the full reasoning.

The dashboard itself polls a couple of small JSON status endpoints on a
timer (see templates/dashboard.html) rather than using server-sent events
or websockets -- simpler to reason about correctly on Flask's built-in
dev server, and at roughly one poll per second the result is
indistinguishable from "live" for processes that update once per
simulated day (training) or once per batch (a full pull) anyway.
"""
from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from flask import Flask, jsonify, render_template, request

from . import inventory, pipeline, simulate, train
from .config import DEFAULT_CONFIG, DAILY_VARIABLES

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


class FetchManager:
    """Surfaces and controls the full data pull for one store, whether
    it's running as this dashboard's own background thread, as the
    separate `weatherbot.service` systemd unit, or as a plain CLI
    invocation someone left running in a terminal -- deliberately unlike
    TrainingManager/SimulationManager, this doesn't assume it owns
    whatever's running.

    Status is read from `<store_path>.status.json` (pipeline.status_file),
    which any of those three write identically, plus whether THIS
    process's own thread happens to be alive. Stop calls
    pipeline.request_stop(), a cooperative file flag every run_pipeline
    loop checks between batches regardless of which process is running it
    -- so Stop works the same way (finishes the in-flight batch, then
    halts cleanly, fully resumable) no matter who started the pull. Start
    only ever launches a pull from *this* thread; if one is already
    running elsewhere, pipeline.FetchLock makes that attempt fail fast
    with a clear "already running" message instead of writing to the
    store twice.
    """

    DISPLAY_FRESH_SEC = 120  # "still active" window for the status display -- not a correctness
    # boundary (pipeline.FetchLock's own, more conservative staleness window is what actually
    # guards against two writers; this one only controls what the dashboard tells the user).

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: str | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("A full pull is already running from this dashboard.")
            # Fail fast with a clear message if something else already
            # holds the store's FetchLock (systemd, another dashboard
            # session, a plain CLI run). run_pipeline would catch this
            # anyway once the background thread starts, but probing
            # synchronously here means Start gets an immediate, accurate
            # rejection instead of only surfacing it a moment later via
            # the status/error field. Tiny inherent race (another process
            # could acquire the lock in the gap between this probe and
            # the thread's real acquire) is harmless either way --
            # run_pipeline's own lock acquisition is still the actual
            # guard against a double-write, this is purely about giving
            # faster feedback in the common case.
            probe = pipeline.FetchLock(self.store_path)
            probe.acquire()
            probe.release()
            self._error = None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        pipeline.request_stop(self.store_path)

    def _run(self) -> None:
        try:
            config = DEFAULT_CONFIG.with_overrides(store_path=self.store_path)

            def on_progress(info: pipeline.ProgressInfo) -> None:
                pipeline.write_status_file(self.store_path, info)

            summary = pipeline.run_full_mode(config, on_progress=on_progress)
            pipeline.write_failure_log(summary, self.store_path)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the dashboard, not just logged
            log.exception("full pull crashed")
            with self._lock:
                self._error = str(exc)

    def status(self) -> dict:
        status_file = pipeline.read_status_file(self.store_path)
        fresh = False
        if status_file is not None:
            try:
                updated = _dt.datetime.fromisoformat(status_file["updated_utc"])
                fresh = (_dt.datetime.utcnow() - updated).total_seconds() < self.DISPLAY_FRESH_SEC
            except (KeyError, ValueError):
                fresh = False
        dashboard_running = self.is_running()
        with self._lock:
            error = self._error
        return {
            "active": dashboard_running or fresh,
            "dashboard_started": dashboard_running,
            "external_active": fresh and not dashboard_running,
            "status": status_file,
            "error": error,
            "failures": pipeline.read_latest_failures(self.store_path),
        }


class _StoreMeta:
    """Lazily-populated, cached-once store metadata (points, variables,
    date range).

    store_path may not exist yet when the dashboard starts: the "Droplet
    Pull" tab's whole point is to let someone start the very first full
    pull from here, on a fresh droplet, before any archive exists -- so
    unlike everything else in this module, this can't assume
    inventory.open_store() will succeed. ensure() is called at the top of
    every route that needs this metadata and retries opening the store
    each time until it first succeeds (cheap: a consolidated zarr open is
    one small metadata read, not the archive's actual weather data), then
    caches it for the rest of the process's life -- store shape shouldn't
    change after creation, matching every other resumability assumption
    in this codebase.
    """

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.lats = np.array([])
        self.lons = np.array([])
        self.point_ids = np.array([], dtype=int)
        self.variables: list[str] = []
        self.time_min: Optional[str] = None
        self.time_max: Optional[str] = None
        self.loaded = False

    def ensure(self) -> bool:
        if self.loaded:
            return True
        try:
            ds = inventory.open_store(self.store_path)
        except Exception:  # noqa: BLE001 -- store doesn't exist yet, or isn't a valid zarr store
            return False
        try:
            self.lats = ds["lat"].values.astype(float)
            self.lons = ds["lon"].values.astype(float)
            self.point_ids = np.arange(ds.sizes["point"])
            self.variables = [v for v in DAILY_VARIABLES if v in ds.data_vars]
            self.time_min = str(ds["time"].values.min())[:10]
            self.time_max = str(ds["time"].values.max())[:10]
        finally:
            ds.close()
        self.loaded = True
        return True


def create_app(store_path: str) -> Flask:
    app = Flask(__name__)

    meta = _StoreMeta(store_path)
    meta.ensure()  # best-effort at startup; routes below retry if this store doesn't exist yet

    # Computed once (grid generation is a real, if modest, computation --
    # not something to redo on every dashboard page load) and static for
    # this process's life: it depends only on DEFAULT_CONFIG, not on
    # whether the store exists yet, so it's exactly what "Droplet Pull"
    # needs to show before a first pull has ever been started.
    fetch_estimate = pipeline.estimate_run(DEFAULT_CONFIG.with_overrides(store_path=store_path), "full")

    training = TrainingManager(store_path)
    simulation = SimulationManager(store_path)
    fetch_mgr = FetchManager(store_path)

    @app.get("/")
    def dashboard():
        meta.ensure()
        return render_template(
            "dashboard.html",
            store_path=store_path,
            store_exists=meta.loaded,
            n_points=len(meta.point_ids),
            variables=meta.variables,
            time_min=meta.time_min,
            time_max=meta.time_max,
            default_horizon=simulate.DEFAULT_HORIZON_DAYS,
            max_horizon=simulate.MAX_HORIZON_DAYS,
            default_paths=simulate.DEFAULT_N_PATHS,
            max_paths=simulate.MAX_N_PATHS,
            fetch_estimate=fetch_estimate,
        )

    @app.get("/api/store/summary")
    def store_summary():
        exists = meta.ensure()
        return jsonify({
            "store_path": store_path,
            "exists": exists,
            "n_points": len(meta.point_ids),
            "variables": meta.variables,
            "time_min": meta.time_min,
            "time_max": meta.time_max,
        })

    @app.get("/api/points")
    def points():
        meta.ensure()
        return jsonify([
            {"point_id": int(pid), "lat": float(lat), "lon": float(lon)}
            for pid, lat, lon in zip(meta.point_ids, meta.lats, meta.lons)
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
        if not meta.ensure():
            return jsonify({"error": "No archive yet -- run a full pull first (see the Droplet Pull tab)."}), 409
        body = request.get_json(force=True, silent=True) or {}
        chosen_vars = [v for v in body.get("variables", []) if v in meta.variables]
        if not chosen_vars:
            return jsonify({"error": "Select at least one known variable."}), 400

        point_id = body.get("point_id")
        if point_id is not None:
            try:
                point_idx = int(point_id)
            except (TypeError, ValueError):
                return jsonify({"error": "point_id must be an integer."}), 400
            if not (0 <= point_idx < len(meta.point_ids)):
                return jsonify({"error": f"point_id out of range 0..{len(meta.point_ids) - 1}."}), 400
        else:
            try:
                lat = float(body["lat"])
                lon = float(body["lon"])
            except (KeyError, TypeError, ValueError):
                return jsonify({"error": "Provide either point_id, or both lat and lon."}), 400
            point_idx = simulate.nearest_point_idx(meta.lats, meta.lons, lat, lon)

        horizon_days = int(body.get("horizon_days", simulate.DEFAULT_HORIZON_DAYS))
        n_paths = int(body.get("n_paths", simulate.DEFAULT_N_PATHS))
        point_info = {
            "point_id": int(point_idx), "lat": float(meta.lats[point_idx]), "lon": float(meta.lons[point_idx]),
        }

        try:
            simulation.start(
                point_idx, point_info, _dt.date.fromisoformat(meta.time_max),
                chosen_vars, horizon_days, n_paths,
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

    @app.get("/api/fetch/status")
    def fetch_status():
        return jsonify(fetch_mgr.status())

    @app.post("/api/fetch/start")
    def fetch_start():
        try:
            fetch_mgr.start()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"ok": True})

    @app.post("/api/fetch/stop")
    def fetch_stop():
        fetch_mgr.stop()
        return jsonify({"ok": True})

    return app
