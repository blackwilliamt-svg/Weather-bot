# WeatherBot — Phase 1: Data Ingestion

Fetches historical daily weather from Open-Meteo's historical archive API for
land grid points across North America, downcasts to float32, and stores it in
a chunked, zstd-compressed Zarr archive queryable with xarray.

This phase is ingestion only. Training, Monte Carlo simulation, and any
dashboard are future phases — see "Seam for future phases" below.

## One-time setup (Windows)

1. Install Python 3.11+ from [python.org/downloads](https://www.python.org/downloads/)
   if you don't already have it. **On the installer's first screen, check
   "Add Python to PATH"** — this matters, setup.bat won't find Python without it.
2. Double-click **`setup.bat`**. It creates a local virtual environment and
   installs everything WeatherBot needs. This only needs to be done once (run
   it again later if `requirements.txt` ever changes). It can take a few
   minutes the first time.

## Running it

Double-click **`WeatherBot.bat`**. A window opens with three buttons:

- **Run Test Pull** — 20 grid points, last 10 years of data. Finishes in a
  couple of minutes; use this to confirm everything works before committing
  to the full pull. Writes to `data/weather_test.zarr`.
- **Run Full Pull** — the real archive: ~19,000-26,000 land grid points across
  North America, 1940-present, roughly 18-27GB compressed. At the default
  rate limit this is well over 100,000 individual requests — the app shows an
  exact estimate (points/days/requests/size/minimum time) before you confirm,
  and it will likely be **days**, not hours, unless you raise `--rate-limit`
  (which risks more 429s — see "Command line" below). Leave the computer on
  and connected to the internet while it runs. Writes to
  `data/weather_archive.zarr`.
- **Show Inventory** — pick any `.zarr` folder under `data/` and see what's in
  it so far: point count, date range, variables, and size on disk — without
  loading the actual weather data into memory.

The window shows a progress bar, a running log, and live stats (files
fetched, data downloaded, elapsed/estimated time remaining) while a pull is
in progress, and both buttons stay disabled until it finishes so you can't start
two runs at once. If any grid points fail to fetch (network error, bad
response, etc.), that's not fatal — they're skipped, and a summary shows up in
the log and in a popup when the run finishes, with the full list saved next to
the store as `<store_path>.failures_<timestamp>.json`.

### Command line (optional)

Everything the GUI does is also available from the command line, with extra
tuning knobs the GUI doesn't expose:

```bash
.venv\Scripts\python.exe -m weatherbot fetch --mode test
.venv\Scripts\python.exe -m weatherbot fetch --mode full
.venv\Scripts\python.exe -m weatherbot inventory --store data/weather_archive.zarr --list-points
```

Grid density, chunking, batching, and rate limiting are all configurable
rather than hardcoded:

```bash
.venv\Scripts\python.exe -m weatherbot fetch --mode full --spacing-deg 0.15 \
    --point-chunk 50 --time-chunk-days 365 --zstd-level 12 \
    --batch-size 10 --time-chunk-years 2 --rate-limit 0.3
```

Open-Meteo's free/keyless tier rejects requests that ask for too much data at
once (many locations x many years x many variables in one call) and separately
caps total data volume per minute/hour. `--batch-size` (locations per HTTP
request) and `--time-chunk-years` (years fetched per request) control how the
pipeline splits work to stay under that; the defaults were tuned empirically
against the live API and include retry/backoff for occasional 429s, but if
you still see frequent rate-limit failures, lower those further or reduce
`--rate-limit`.

## Geographic coverage

The default grid covers central/eastern North America (roughly 29-45°N,
77-114°W) rather than the full continent. A full Central-America-to-Arctic-
Canada/Alaska box at 0.14-0.16° spacing produces ~125,000 land points and a
90GB+ compressed store — well past the ~19,000-26,000 point / ~25GB targets.
This box was chosen to hit both targets at the target spacing; it's a plain
constant (`NORTH_AMERICA_BBOX` in `config.py`), so widening it is a one-line
change if you'd rather have full continental coverage at the cost of a much
larger store.

## Storage format

- Zarr store, dims `(point, time)`. `point` is a stable integer index into the
  land-filtered grid (sorted by lat, then lon); `lat`/`lon` are coordinates
  indexed by `point`. `time` is a daily `datetime64` index from 1940-01-01.
- All data variables are `float32`, compressed with `numcodecs.Zstd` at a
  configurable "mid" level (default 12; range 1-22) — lossless beyond the
  float32 downcast itself.
- Chunked by point (default 50) and by ~1 year of days (default 365),
  configurable — sized for a ~512MB RAM serving target on the eventual
  droplet this store gets shipped to.
- Fetched data is downcast and written straight into the compressed store —
  there's no uncompressed intermediate file at any point.

## Error handling

Failed points (network errors that exhausted retries, or a bad/missing
response for a specific location) are skipped, not fatal — the run continues.
At the end, a summary is shown (in the GUI log/popup, or printed to the
console for the CLI) and the full list is written to
`<store_path>.failures_<timestamp>.json` next to the store.

## Seam for future phases

Don't reach into the zarr store directly from new code. Use
`weatherbot.inventory.open_store(path)` — it opens the store lazily via
xarray/dask (no data materializes until you actually index/compute it), and
is the intended entry point for the walk-forward training, Monte Carlo
simulation, and dashboard phases that come after this one.

Known limitation (by design, out of scope for this phase): a store's time
extent is fixed at creation time. Extending an existing store with newer days
without a full re-init is a future "incremental update" feature, not built
here.
