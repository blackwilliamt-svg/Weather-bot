"""CLI entry point: python -m weatherbot <fetch|inventory> ..."""
from __future__ import annotations

import argparse
import logging
import sys

from tqdm import tqdm

from . import inventory, pipeline
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
    fetch_p.add_argument("--rate-limit", dest="rate_limit_per_sec", type=float, default=None)
    fetch_p.add_argument("--concurrency", type=int, default=None)

    inv_p = sub.add_parser("inventory", help="Inspect an existing zarr store")
    inv_p.add_argument("--store", dest="store_path", default=DEFAULT_CONFIG.store_path)
    inv_p.add_argument("--list-points", action="store_true")

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
        )

        est = pipeline.estimate_run(config, args.mode)
        print(f"Estimate: {est['n_points']:,} points x {est['n_days']:,} days x "
              f"{len(config.variables)} variables, over {est['n_requests']:,} requests "
              f"(concurrency={config.concurrency}). "
              f"Store size ~{pipeline.format_bytes(est['compressed_bytes_low'])}-"
              f"{pipeline.format_bytes(est['compressed_bytes_high'])} compressed "
              f"({pipeline.format_bytes(est['raw_bytes'])} raw). "
              f"Best-case time (pacing floor, unlimited concurrency): "
              f"{pipeline.format_duration(est['min_seconds'])}.\n")

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

            if args.mode == "test":
                summary = pipeline.run_test_mode(config, on_progress=on_progress)
            else:
                summary = pipeline.run_full_mode(config, on_progress=on_progress)

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

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
