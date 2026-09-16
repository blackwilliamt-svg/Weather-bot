"""CLI entry point: python -m weatherbot <fetch|inventory|train|serve> ..."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from tqdm import tqdm

from . import inventory, pipeline, train as train_module
from .config import DEFAULT_CONFIG, DEFAULT_FULL_STORE_PATH, DEFAULT_TEST_STORE_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weatherbot")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", help="Run the ingestion pipeline")
    fetch_p.add_argument("--mode", choices=["test", "full"], required=True)
    fetch_p.add_argument("--store", dest="store_path", default=None)
    fetch_p.add_argument("--spacing-deg", type=float, default=None)
    fetch_p.add_argument("--point-chunk", type=int, default=None)
    fetch_p.add_argument("--time-chunk-days", type=int, default=None)
    fetch_p.add_argument("--zstd-level", type=int, default=None)
    fetch_p.add_argument("--batch-size", type=int, default=None)
    fetch_p.add_argument("--time-chunk-years", type=int, default=None)
    fetch_p.add_argument("--rate-limit", dest="rate_limit_per_sec", type=float, default=None,
                          help="test mode only -- full pull always uses --full-pull-interval-sec")
    fetch_p.add_argument("--concurrency", type=int, default=None,
                          help="test mode only -- full pull always runs sequentially (concurrency 1)")
    fetch_p.add_argument("--full-pull-interval-sec", dest="full_pull_request_interval_sec",
                          type=float, default=None,
                          help="full pull's fixed, sequential per-request pace (default: 9s)")

    inv_p = sub.add_parser("inventory", help="Inspect an existing zarr store")
    inv_p.add_argument("--store", dest="store_path", default=DEFAULT_CONFIG.store_path)
    inv_p.add_argument("--list-points", action="store_true")

    train_p = sub.add_parser(
        "train", help="Run continuous walk-forward training headlessly (no dashboard)"
    )
    train_p.add_argument("--store", dest="store_path", default=DEFAULT_CONFIG.store_path)

    serve_p = sub.add_parser(
        "serve", help="Launch the web dashboard (training + Monte Carlo simulation, with visualization/controls)"
    )
    serve_p.add_argument("--store", dest="store_path", default=DEFAULT_CONFIG.store_path)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch":
        store_path = args.store_path or (
            DEFAULT_TEST_STORE_PATH if args.mode == "test" else DEFAULT_FULL_STORE_PATH
        )
        config = DEFAULT_CONFIG.with_overrides(
            store_path=store_path,
            spacing_deg=args.spacing_deg,
            point_chunk=args.point_chunk,
            time_chunk_days=args.time_chunk_days,
            zstd_level=args.zstd_level,
            batch_size=args.batch_size,
            time_chunk_years=args.time_chunk_years,
            rate_limit_per_sec=args.rate_limit_per_sec,
            concurrency=args.concurrency,
            full_pull_request_interval_sec=args.full_pull_request_interval_sec,
        )

        est = pipeline.estimate_run(config, args.mode)
        if args.mode == "full":
            pace_note = (f"fixed pace, 1 request/{config.full_pull_request_interval_sec:g}s, "
                         "sequential")
            time_label = "Expected time"
        else:
            pace_note = f"concurrency={config.concurrency}, adaptive pacing"
            time_label = "Best-case time (pacing floor, unlimited concurrency)"
        print(f"Estimate: {est['n_points']:,} points x {est['n_days']:,} days x "
              f"{len(config.variables)} variables, over {est['n_requests']:,} requests "
              f"({pace_note}). "
              f"Store size ~{pipeline.format_bytes(est['compressed_bytes_low'])}-"
              f"{pipeline.format_bytes(est['compressed_bytes_high'])} compressed "
              f"({pipeline.format_bytes(est['raw_bytes'])} raw). "
              f"{time_label}: {pipeline.format_duration(est['min_seconds'])}.\n")

        # SIGTERM (what `systemctl stop` sends) and Ctrl-C both request a
        # stop through the same file-based mechanism the dashboard's Stop
        # button uses (pipeline.request_stop) rather than just dying --
        # per-batch checkpointing already makes an abrupt kill safe to
        # resume from, but going through request_stop also releases the
        # FetchLock's heartbeat immediately instead of waiting out its
        # staleness window, so a restart (or the dashboard) can pick the
        # store back up right away.
        def _handle_signal(signum, _frame):
            print(f"\nReceived signal {signum} -- stopping after the current batch (checkpointing)...")
            pipeline.request_stop(config.store_path)

        signal.signal(signal.SIGINT, _handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handle_signal)

        with tqdm(desc="fetching", unit="batch") as pbar:
            def on_progress(info: pipeline.ProgressInfo) -> None:
                if pbar.total != info.total_steps:
                    pbar.total = info.total_steps
                pbar.n = info.step
                pbar.set_postfix_str(
                    f"{pipeline.format_bytes(info.bytes_downloaded)} downloaded, "
                    f"ETA {pipeline.format_duration(info.eta_sec)} | {info.message}"
                )
                pbar.refresh()
                pipeline.write_status_file(config.store_path, info)

            try:
                if args.mode == "test":
                    summary = pipeline.run_test_mode(config, on_progress=on_progress)
                else:
                    summary = pipeline.run_full_mode(config, on_progress=on_progress)
            except RuntimeError as exc:
                # Most likely: FetchLock already held by another process
                # (the systemd service, the dashboard, or a second CLI
                # invocation) against this same store.
                print(f"\n{exc}")
                return 1

        if summary["already_complete"]:
            print(f"\nAlready complete — {summary['n_points']} points, {summary['n_batches']} "
                  "fetch batches were all fetched by a previous run. Nothing to do.")
        elif summary["stopped"]:
            reason = summary.get("stop_reason")
            why = {"rate_limit_day": "Open-Meteo's daily request budget is exhausted (resets in up to 24h).",
                   "rate_limit_hour": "Open-Meteo's hourly request budget is exhausted."}.get(reason, "Stopped.")
            print(f"\n{why} {summary['completed_steps']}/{summary['n_batches']} batches done this run. "
                  "Resumable — run the same command again later to continue.")
        else:
            resumed_note = f" (resumed from batch {summary['resumed_from']})" if summary["resumed_from"] else ""
            print(f"\nDone{resumed_note}. {summary['n_points']} points, {summary['n_batches']} fetch "
                  f"batches total, {len(summary['failures'])} failed attempts.")
            print(f"Downloaded {pipeline.format_bytes(summary['bytes_downloaded'])} this session "
                  f"in {pipeline.format_duration(summary['elapsed_sec'])}.")
        print(pipeline.format_failure_summary(summary))
        log_path = pipeline.write_failure_log(summary, config.store_path)
        if log_path:
            print(f"Full failure list written to: {log_path}")
        return 0

    if args.command == "inventory":
        print(inventory.summarize(args.store_path, list_points=args.list_points))
        return 0

    if args.command == "train":
        return _run_train(args.store_path)

    if args.command == "serve":
        return _run_serve(args.store_path, args.host, args.port)

    parser.print_help()
    return 1


def _run_train(store_path: str) -> int:
    """Headless walk-forward training with no web UI -- for a droplet
    that doesn't need visualization, or debugging. Runs until stopped:
    Ctrl-C locally, or SIGTERM (what `systemctl stop` sends) on a droplet
    both trigger a clean checkpoint-and-exit rather than an abrupt kill,
    same philosophy as the fetch pipeline's Stop button.
    """
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        print(f"\nReceived signal {signum} -- stopping after the current step (checkpointing)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    def on_progress(progress: train_module.TrainingProgress) -> None:
        print(f"[{progress.phase}] {progress.current_date} — {progress.message}", flush=True)

    train_module.run_walk_forward(store_path, on_progress=on_progress, should_stop=stop_event.is_set)
    return 0


def _run_serve(store_path: str, host: str, port: int) -> int:
    from . import webapp  # deferred: only `serve` needs Flask installed

    app = webapp.create_app(store_path)
    print(f"WeatherBot dashboard at http://{host}:{port} (store: {store_path})")
    app.run(host=host, port=port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
