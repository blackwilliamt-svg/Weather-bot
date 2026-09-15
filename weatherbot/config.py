"""Central configuration for the WeatherBot ingestion pipeline.

Every tunable that affects grid density, chunking, compression, storage
location, or API behavior lives here (with CLI overrides in __main__.py).
Nothing downstream should hardcode these values.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, replace

# Daily variables pulled from Open-Meteo's historical archive API.
DAILY_VARIABLES: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "surface_pressure_mean",
    "cloud_cover_mean",
    "relative_humidity_2m_mean",
]

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Land-focused bounding box (lat_min, lat_max, lon_min, lon_max): the
# contiguous United States (CONUS) only — Alaska, Hawaii, and the rest of
# North America (Canada, Mexico, Central America) are out of scope. This is
# a reduced-scope config (confirmed with the user): a coarser 0.5 deg
# (~55km) grid over a much smaller area than the old central/eastern North
# America box, so both point count and storage footprint shrink accordingly.
# Widening the box or tightening the spacing is a config change away.
CONUS_BBOX = (24.0, 50.0, -125.0, -66.5)

ARCHIVE_START_DATE = _dt.date(1990, 1, 1)
ARCHIVE_LAG_DAYS = 5  # Open-Meteo's archive typically lags a few days behind "today".

TEST_MODE_POINTS = 20
TEST_MODE_YEARS = 10

# Test and full runs default to different store paths so a validation run
# never overwrites (or gets overwritten by) the real archive.
DEFAULT_TEST_STORE_PATH = "data/weather_test.zarr"
DEFAULT_FULL_STORE_PATH = "data/weather_archive.zarr"


@dataclass(frozen=True)
class Config:
    # Grid
    bbox: tuple[float, float, float, float] = CONUS_BBOX
    spacing_deg: float = 0.5

    # Zarr layout. run_test_mode/run_full_mode callers (CLI, GUI) resolve this
    # to DEFAULT_TEST_STORE_PATH / DEFAULT_FULL_STORE_PATH when not overridden.
    store_path: str = DEFAULT_FULL_STORE_PATH
    point_chunk: int = 50
    time_chunk_days: int = 365
    zstd_level: int = 12

    # Fetch behavior.
    # Open-Meteo's free/keyless tier rejects large requests outright ("too much
    # data") and separately caps cumulative data volume per minute/hour.
    # Empirically, 15 locations x 10 years x 18 variables ("location-years" =
    # locations x years, at fixed 18 vars) gets rejected outright while 3 x 10
    # succeeds. batch_size x time_chunk_years = 60 location-years here — 2x
    # the old defaults (20), still well under the known-failing 150 — since
    # request COUNT (not just per-request volume) is what batching is meant
    # to cut: for a fixed location-years budget, total requests only depends
    # on that budget, not on how it's split between locations vs. years.
    batch_size: int = 20  # locations per HTTP request
    time_chunk_years: int = 3  # years of data fetched per HTTP request, both modes
    # concurrency/rate_limit_per_sec below apply to TEST MODE only — it's a
    # short validation run, not subject to the multi-day daily-quota budget,
    # so it's fine to fetch fast and adaptively. Full pull mode ALWAYS uses
    # full_pull_request_interval_sec at concurrency 1 instead (see
    # pipeline.run_full_mode) — Open-Meteo's free tier caps out at 600
    # calls/minute, 5,000/hour, AND 10,000/day, and for a sustained multi-day
    # pull the DAILY cap is the real bottleneck: bursting up to the minute/
    # hour limits just hits the daily one sooner, then sits idle until reset.
    # A fixed one-request-per-9s drip (~9,600/day) stays under all three
    # continuously, with margin below the exact 8.64s/request ceiling
    # (86400s / 10000), and needs no reactive backoff to stay safe.
    concurrency: int = 4  # concurrent in-flight requests (bounded worker pool)
    # rate_limit_per_sec is the FLOOR pace per lane absent any backoff (not a
    # fixed throttle) — the adaptive limiter in fetch.py starts here and only
    # slows down in response to actual 429s/errors, speeding back up on a
    # success streak. See fetch.AdaptiveRateLimiter.
    rate_limit_per_sec: float = 2.0
    full_pull_request_interval_sec: float = 9.0  # full pull's fixed, sequential pace
    max_backoff_sec: float = 90.0  # ceiling on the adaptive limiter's backoff
    max_retries: int = 5
    backoff_base_sec: float = 2.0
    request_timeout_sec: float = 60.0

    # Dates
    archive_start_date: _dt.date = ARCHIVE_START_DATE
    archive_lag_days: int = ARCHIVE_LAG_DAYS

    variables: tuple[str, ...] = tuple(DAILY_VARIABLES)

    def end_date(self) -> _dt.date:
        return _dt.date.today() - _dt.timedelta(days=self.archive_lag_days)

    def with_overrides(self, **kwargs) -> "Config":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)


DEFAULT_CONFIG = Config()
