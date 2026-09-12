"""Batched Open-Meteo historical archive API client.

Handles multi-location batching (comma-joined lat/lon), rate limiting, and
retry/backoff on transient failures. Never raises for a single bad point —
per-point failures are reported back to the caller so the pipeline can skip
and continue.
"""
from __future__ import annotations

import datetime as _dt
import logging
import random
import threading
import time
from dataclasses import dataclass

import pandas as pd
import requests

from .config import Config

log = logging.getLogger(__name__)

# Open-Meteo's free/keyless tier returns a 429 for three different reasons
# with very different recovery times: a "Minutely" message clears in about a
# minute (ordinary adaptive backoff, capped by config.max_backoff_sec, is
# enough — fine to retry in-process). "Hourly" and "Daily" mean a much bigger
# budget is spent — recovering takes 15+ minutes to a full day, far too long
# to block a worker thread inside a retry loop for. Those are NOT retried
# in-process at all: fetch_batch returns immediately with rate_limited set,
# and the pipeline stops the whole run (leaving the batch unwritten, so a
# later resume retries it) instead of sleeping a thread or — worse —
# recording it as a permanent per-point failure.
_MINUTE_LIMIT_MIN_WAIT_SEC = 60.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class FetchResult:
    point_id: int
    lat: float
    lon: float
    daily: dict | None  # raw Open-Meteo "daily" block, or None on failure
    error: str | None = None


@dataclass
class BatchOutcome:
    results: list[FetchResult]
    bytes_downloaded: int  # total response bytes across all attempts (incl. retries)
    rate_limited: str | None = None  # None | "hour" | "day" — see module docstring above


def build_session(config: Config) -> requests.Session:
    return requests.Session()


class AdaptiveRateLimiter:
    """Thread-safe pacing shared across all concurrent workers.

    Starts at config.rate_limit_per_sec (a floor, not a fixed throttle) and
    only slows down in response to actual 429s/errors — via on_success()
    (gradually speeds back up toward the floor after a clean streak) and
    on_rate_limited()/on_hourly_limit() (backs off, capped at
    config.max_backoff_sec for ordinary cases; the hourly case gets its own
    long fixed cooldown instead, applied globally so every worker pauses,
    not just the one that got the 429).
    """

    def __init__(self, config: Config):
        self._min_interval = 1.0 / config.rate_limit_per_sec if config.rate_limit_per_sec > 0 else 0.0
        self._max_interval = config.max_backoff_sec
        self._current_interval = self._min_interval
        self._next_allowed = 0.0
        self._success_streak = 0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            dispatch_at = max(now, self._next_allowed)
            self._next_allowed = dispatch_at + self._current_interval
        sleep_for = dispatch_at - now
        if sleep_for > 0:
            time.sleep(sleep_for)

    def on_success(self) -> None:
        with self._lock:
            self._success_streak += 1
            # Every few clean requests, ease back toward the floor pace.
            if self._success_streak >= 3 and self._current_interval > self._min_interval:
                self._current_interval = max(self._min_interval, self._current_interval * 0.7)
                self._success_streak = 0

    def on_rate_limited(self) -> None:
        with self._lock:
            self._success_streak = 0
            self._current_interval = min(self._max_interval, max(self._current_interval * 2, 1.0))
            self._next_allowed = max(self._next_allowed, time.monotonic() + _MINUTE_LIMIT_MIN_WAIT_SEC)

    def on_long_cooldown(self, seconds: float) -> None:
        """For hour/day-scale limits — sets a global pause, but callers must
        NOT block a worker thread waiting it out (see fetch_batch); this only
        affects the pace of whatever gets dispatched next, after the caller
        has already decided to stop and resume later."""
        with self._lock:
            self._success_streak = 0
            self._next_allowed = max(self._next_allowed, time.monotonic() + seconds)


def fetch_batch(
    points: pd.DataFrame,
    start_date: _dt.date,
    end_date: _dt.date,
    config: Config,
    session: requests.Session,
    limiter: AdaptiveRateLimiter,
) -> BatchOutcome:
    """Fetch one batch of points (a small DataFrame with point_id/lat/lon) for
    one shared date range. Returns one FetchResult per input point, in order,
    plus the total response bytes downloaded (including retried attempts).
    Never raises: a batch-level failure marks every point in it failed; a
    per-location error inside a 200 response marks just that point failed.
    """
    lat_str = ",".join(f"{lat:.4f}" for lat in points["lat"])
    lon_str = ",".join(f"{lon:.4f}" for lon in points["lon"])
    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(config.variables),
        "timezone": "UTC",
    }

    last_reason: str | None = None
    resp: requests.Response | None = None
    bytes_downloaded = 0
    for attempt in range(config.max_retries + 1):
        limiter.wait()
        try:
            resp = session.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=config.request_timeout_sec,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            resp = None
            last_reason = f"network error: {exc}"
            sleep_s = config.backoff_base_sec * (2 ** attempt) + random.uniform(0, 1)
            log.warning("network error fetching batch (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, config.max_retries + 1, exc, sleep_s)
            time.sleep(sleep_s)
            continue

        bytes_downloaded += len(resp.content)
        if resp.status_code == 200:
            limiter.on_success()
            break

        last_reason = f"HTTP {resp.status_code}: {resp.text[:200]}"

        if resp.status_code == 429:
            reason_text = resp.text.lower()
            if "daily" in reason_text or "hourly" in reason_text:
                # Hour/day-scale budget exhausted: don't retry in-process —
                # that would block this thread for 15+ min to a full day.
                # Return immediately so the pipeline can stop the whole run
                # cleanly and resume it later instead.
                severity = "day" if "daily" in reason_text else "hour"
                cooldown = 86400.0 if severity == "day" else 900.0
                limiter.on_long_cooldown(cooldown)
                log.warning("%s rate limit hit — stopping this run, resumable later: %s",
                            severity, last_reason)
                results = [
                    FetchResult(int(r.point_id), float(r.lat), float(r.lon), None, last_reason)
                    for r in points.itertuples()
                ]
                return BatchOutcome(results, bytes_downloaded, rate_limited=severity)

        if resp.status_code not in _RETRYABLE_STATUS or attempt == config.max_retries:
            break  # not retryable (e.g. 400 "too much data") or out of attempts

        if resp.status_code == 429:
            limiter.on_rate_limited()
            sleep_s = _MINUTE_LIMIT_MIN_WAIT_SEC
        else:
            sleep_s = config.backoff_base_sec * (2 ** attempt) + random.uniform(0, 1)
        log.warning("HTTP %d fetching batch (attempt %d/%d) — retrying in %.1fs: %s",
                    resp.status_code, attempt + 1, config.max_retries + 1, sleep_s, last_reason)
        time.sleep(sleep_s)

    if resp is None or resp.status_code != 200:
        reason = last_reason or "unknown fetch failure"
        results = [
            FetchResult(int(r.point_id), float(r.lat), float(r.lon), None, reason)
            for r in points.itertuples()
        ]
        return BatchOutcome(results, bytes_downloaded)

    try:
        payload = resp.json()
    except ValueError as exc:
        reason = f"invalid JSON response: {exc}"
        results = [
            FetchResult(int(r.point_id), float(r.lat), float(r.lon), None, reason)
            for r in points.itertuples()
        ]
        return BatchOutcome(results, bytes_downloaded)

    # Single location -> plain object; multiple -> list of objects (same order as input).
    records = payload if isinstance(payload, list) else [payload]

    results = []
    for row, record in zip(points.itertuples(), records):
        if not isinstance(record, dict) or record.get("error"):
            reason = record.get("reason", "unknown error") if isinstance(record, dict) else "malformed response"
            results.append(FetchResult(int(row.point_id), float(row.lat), float(row.lon), None, reason))
            continue
        daily = record.get("daily")
        if not daily:
            results.append(FetchResult(int(row.point_id), float(row.lat), float(row.lon), None, "no 'daily' block in response"))
            continue
        results.append(FetchResult(int(row.point_id), float(row.lat), float(row.lon), daily, None))
    return BatchOutcome(results, bytes_downloaded)
