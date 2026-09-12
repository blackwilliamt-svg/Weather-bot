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
import time
from dataclasses import dataclass

import pandas as pd
import requests

from .config import Config

log = logging.getLogger(__name__)

# Open-Meteo's free/keyless tier returns this when the per-minute data-volume
# budget is exceeded — its own message says to wait about a minute, so retries
# on this status use a floor well above ordinary exponential backoff.
_RATE_LIMIT_MIN_WAIT_SEC = 60.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class FetchResult:
    point_id: int
    lat: float
    lon: float
    daily: dict | None  # raw Open-Meteo "daily" block, or None on failure
    error: str | None = None


def build_session(config: Config) -> requests.Session:
    return requests.Session()


class RateLimiter:
    def __init__(self, per_sec: float):
        self._min_interval = 1.0 / per_sec if per_sec > 0 else 0.0
        self._last_call = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


def fetch_batch(
    points: pd.DataFrame,
    start_date: _dt.date,
    end_date: _dt.date,
    config: Config,
    session: requests.Session,
    limiter: RateLimiter,
) -> list[FetchResult]:
    """Fetch one batch of points (a small DataFrame with point_id/lat/lon) for
    one shared date range. Returns one FetchResult per input point, in order.
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

        if resp.status_code == 200:
            break

        last_reason = f"HTTP {resp.status_code}: {resp.text[:200]}"
        if resp.status_code not in _RETRYABLE_STATUS or attempt == config.max_retries:
            break  # not retryable (e.g. 400 "too much data") or out of attempts

        sleep_s = config.backoff_base_sec * (2 ** attempt) + random.uniform(0, 1)
        if resp.status_code == 429:
            sleep_s = max(sleep_s, _RATE_LIMIT_MIN_WAIT_SEC)
        log.warning("HTTP %d fetching batch (attempt %d/%d) — retrying in %.1fs: %s",
                    resp.status_code, attempt + 1, config.max_retries + 1, sleep_s, last_reason)
        time.sleep(sleep_s)

    if resp is None or resp.status_code != 200:
        reason = last_reason or "unknown fetch failure"
        return [
            FetchResult(int(r.point_id), float(r.lat), float(r.lon), None, reason)
            for r in points.itertuples()
        ]

    try:
        payload = resp.json()
    except ValueError as exc:
        reason = f"invalid JSON response: {exc}"
        return [
            FetchResult(int(r.point_id), float(r.lat), float(r.lon), None, reason)
            for r in points.itertuples()
        ]

    # Single location -> plain object; multiple -> list of objects (same order as input).
    records = payload if isinstance(payload, list) else [payload]

    results: list[FetchResult] = []
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
    return results
