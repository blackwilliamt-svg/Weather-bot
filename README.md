# WeatherBot

Fetches historical daily weather from Open-Meteo's historical archive API for
land grid points across the contiguous United States (CONUS), downcasts to
float32, and stores it in a chunked, zstd-compressed Zarr archive queryable
with xarray (**Phase 1: Data Ingestion**, below) — then continuously
walk-forward trains a lightweight per-point forecasting model on that
archive and uses it to run Monte Carlo forecast simulations, both viewable
live in a web dashboard (**Phase 2: Modeling & Forecasting**, further down).

## Phase 1: Data Ingestion

## One-time setup (Windows)

1. Install Python 3.11+ from [python.org/downloads](https://www.python.org/downloads/)
   if you don't already have it. **On the installer's first screen, check
   "Add Python to PATH"** — this matters, setup.bat won't find Python without it.
2. Double-click **`setup.bat`**. It creates a local virtual environment and
   installs everything WeatherBot needs. This only needs to be done once (run
   it again later if `requirements.txt` ever changes). It can take a few
   minutes the first time.

## Running it

Double-click **`WeatherBot.bat`**. A window opens with four buttons:

- **Run Test Pull** — 20 grid points, last 10 years of data. Finishes in a
  couple of minutes; use this to confirm everything works before committing
  to the full pull. Writes to `data/weather_test.zarr`.
- **Run Full Pull** — the real archive: ~17,300 land grid points across the
  contiguous United States, 1990-present, roughly 5.2-7.8GB compressed. The
  app shows an exact estimate (points/days/requests/size/best-case time)
  before you confirm — see "Fetch performance" below for what actually
  governs how long it takes. Leave the computer on and connected to the
  internet while it runs. Writes to `data/weather_archive.zarr`.
- **Show Inventory** — pick any `.zarr` folder under `data/` and see what's in
  it so far: point count, date range, variables, and size on disk — without
  loading the actual weather data into memory.
- **Stop** — only enabled while a pull is running. Finishes whatever batch is
  currently in flight, then halts cleanly — nothing is lost or corrupted, and
  it's exactly as resumable as an unplanned interruption (see "Interruptions
  and resuming" below). Click the same run button again later to continue
  from where it stopped.

The window shows a progress bar, a running log, and live stats (files
fetched, data downloaded, elapsed/estimated time remaining) while a pull is
in progress, and both buttons stay disabled until it finishes so you can't start
two runs at once. If any grid points fail to fetch (network error, bad
response, etc.), that's not fatal — they're skipped, and a summary shows up in
the log and in a popup when the run finishes, with the full list saved next to
the store as `<store_path>.failures_<timestamp>.json`.

Once you have an archive, double-click **`WeatherBotDashboard.bat`** to open
the training/forecasting dashboard in your browser — see "Phase 2: Modeling
& Forecasting" below.

### Command line (optional)

Everything the GUI does is also available from the command line, with extra
tuning knobs the GUI doesn't expose:

```bash
.venv\Scripts\python.exe -m weatherbot fetch --mode test
.venv\Scripts\python.exe -m weatherbot fetch --mode full
.venv\Scripts\python.exe -m weatherbot inventory --store data/weather_archive.zarr --list-points
```

Grid density, chunking, batching, pacing, and concurrency are all configurable
rather than hardcoded:

```bash
.venv\Scripts\python.exe -m weatherbot fetch --mode full --spacing-deg 0.25 \
    --point-chunk 50 --time-chunk-days 365 --zstd-level 22 \
    --batch-size 20 --time-chunk-years 3 --full-pull-interval-sec 9
```

(`--rate-limit`/`--concurrency` only affect test mode — full pull always
paces at `--full-pull-interval-sec`, sequentially. See "Fetch performance".)

## Fetch performance

Open-Meteo's free/keyless tier has three request-volume caps: 600/minute,
5,000/hour, and 10,000/day. For a one-off small pull, the minute/hour caps
are what you'd notice. For the **full pull specifically** — over 11,000
requests, running a bit past a single day at this scope — the **daily** cap
is the one that actually governs total completion time: bursting up to the
minute/hour limits just hits the daily one sooner and then sits idle until
it resets, it doesn't finish any sooner overall. So full pull and test mode
are paced differently on purpose:

- **Full pull mode: fixed, sequential pacing.** One batched request every
  `--full-pull-interval-sec` (default 9s — 86,400s/day ÷ 10,000 requests =
  8.64s exact ceiling, 9s leaves a small margin), always at concurrency 1,
  **regardless of** `--rate-limit`/`--concurrency` (those apply to test mode
  only). This is deliberate: a steady predictable drip that's designed to
  never trip a 429, rather than bursting and reactively backing off. At
  ~9,600 requests/day this also stays comfortably under the minute/hour caps
  the whole time, so those never come into play for a full pull that's
  behaving normally.
- **Batching** (`--batch-size` locations x `--time-chunk-years` per request)
  is still the main lever for the request COUNT itself: for a fixed
  per-request data budget, count only depends on that budget (locations x
  years), not how it's split between the two — so it's really one knob,
  "location-years per request". Empirically, 15 locations x 10 years (150
  location-years) gets rejected outright while 3 x 10 (30) succeeds; the
  defaults (20 x 3 = 60) sit with margin on both sides, found by testing
  against the live API, not guessed. At the current CONUS/0.25° scope
  (~17,300 points, 1990-present) that keeps full pull's request count to
  ~11,300 (vs. ~32,900 at a naive 10x2 batching) — at the fixed 9s pace,
  that's the difference between a ~28-hour run and a ~3.4-day one.
- **Test mode** stays fast and adaptive — several batches at once
  (`--concurrency`, default 4) with a rate limiter that starts fast and only
  backs off after an actual 429 (shared across workers, so one worker's 429
  slows the whole pool), easing back toward the floor after a clean streak.
  It's a short validation run, not subject to the daily-quota budget a full
  pull has to live within.

Regardless of mode, writes to the zarr store happen one at a time, strictly
in fetch order, on the main thread — concurrency (test mode) only ever
parallelizes the network wait, never the write, so there's no risk of
concurrent writes corrupting a store even when several batches land in the
same chunk file.

An hour- or day-scale 429 (which shouldn't happen under normal full-pull
pacing, but is handled if it does) doesn't retry in a loop or get recorded
as a failure — the run stops cleanly and is fully resumable once the budget
resets, exactly like the Stop button (see "Interruptions and resuming").

At the fixed full-pull pace, the app's upfront estimate (~28 hours, a bit
over a day, at current defaults) is the actual expected duration, not just
a best case — verify it against the live ETA once you start a run.

## Running full pull on a droplet

A ~28-hour continuous run is a bit long to tie up a Windows machine you need
to use/sleep/reboot during that window. Full pull mode is plain CLI — no GUI
dependency at all — so it runs fine headless on a small Linux droplet
instead; test mode stays a local Windows GUI run as before, this section is
full pull only.

### One-time droplet setup

On a fresh Ubuntu/Debian droplet:

```bash
sudo apt update && sudo apt install -y python3 python3-venv git
git clone https://github.com/blackwilliamt-svg/Weather-bot.git weatherbot
cd weatherbot
chmod +x setup.sh && ./setup.sh
```

`setup.sh` is `setup.bat`'s Linux equivalent — creates `.venv`, installs
`requirements.txt`. Same dependencies as Windows; nothing in them is
platform-specific.

### Starting it so it survives disconnects AND reboots

A plain `nohup ... &` survives your SSH session ending, but not a droplet
reboot. Since this is checkpointed (see "Interruptions and resuming") and
expected to run for about a day, a **systemd service** is the more robust
choice — it restarts automatically on a crash or reboot, with no one needing
to notice and re-launch it:

```bash
# edit YOUR_USERNAME and the paths in weatherbot.service first, then:
sudo cp weatherbot.service /etc/systemd/system/weatherbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now weatherbot
```

`Restart=on-failure` plus the checkpoint/resume support means a reboot or
crash just picks up from the last completed batch — nothing is lost or
re-fetched. If you'd rather not deal with systemd, the simpler alternative
is a `tmux`/`screen` session (`tmux new -s weatherbot`, run the fetch
command inside it, detach with `Ctrl-b d`) — that survives your SSH
disconnecting but NOT a reboot, so only use it if you're confident the
droplet won't restart during the run.

### Checking progress remotely

- **`journalctl -u weatherbot -f`** — live-tails the service's output (one
  line roughly every 9 seconds, matching the fixed pace) if you used systemd.
- **`cat data/weather_archive.zarr.status.json`** (or
  `watch -n 30 cat data/weather_archive.zarr.status.json` to auto-refresh) —
  a small JSON snapshot updated after every batch: `step`/`total_steps`/
  `percent`, `bytes_downloaded`, `elapsed`/`eta`, and `failures_so_far`. This
  works regardless of how you started the process.
- **`python -m weatherbot inventory --store data/weather_archive.zarr`** —
  the same inventory command as local use, run over SSH: point count, date
  range, variables, size on disk so far.
- Failed grid points accumulate in `data/weather_archive.zarr.failures_*.json`
  as always (skip-and-log, not fatal) — check it once the run finishes, or
  any time via `cat`.

### Disk space — read this before starting

The estimated compressed store size is **~5.2-7.8GB** (max-level zstd
compression, see "Storage format" below) — still comfortably inside a
default **~25GB** droplet SSD alongside the OS/Python/venv, but with a
narrower margin than a smaller archive would leave. If you're on a smaller
droplet than that, or want more headroom, either:

- Attach a separate DigitalOcean **Volume** (block storage) and point
  `--store` at a path on it, so the store doesn't compete with the boot
  disk at all, or
- Resize the droplet to a larger disk tier first.

Either way, keep an eye on `df -h` during the run — it's much better to
notice this early than to have the pull fail from a full disk partway
through.

### Downloading the finished store

Transfer the directory directly rather than tarring it first — `rsync`
handles "one large directory of many small files" far better than a naive
recursive copy, and — usefully for a large transfer over a home connection —
it's resumable: if it drops partway through, re-running the exact same
command only transfers what's missing/changed.

From Windows, the simplest way to get `rsync` is via WSL (`wsl --install`,
then `sudo apt install -y rsync` inside it once):

```bash
# run from inside WSL; /mnt/e/... reaches your Windows E: drive
rsync -avz --progress \
    your_user@droplet_ip:~/weatherbot/data/weather_archive.zarr/ \
    "/mnt/e/Weather Bot/data/weather_archive.zarr/"
```

If you'd rather not set up WSL, plain `scp` (built into PowerShell on modern
Windows) works too, just without the resumability or the efficiency on many
small files — expect it to be slower for a store this size:

```powershell
scp -r your_user@droplet_ip:~/weatherbot/data/weather_archive.zarr "E:\Weather Bot\data\weather_archive.zarr"
```

WinSCP is a reasonable GUI alternative to either — point it at the same
remote path and it handles resuming an interrupted transfer on its own.

## Geographic coverage

The grid covers only the contiguous United States (CONUS) — roughly
24-50°N, 125-66.5°W — at 0.25° (~22-28km) spacing, land points only, ~17,300
points. Alaska, Hawaii, and the rest of North America (Canada, Mexico,
Central America) are explicitly out of scope for this reduced-scope build;
the bounding box is a plain constant (`CONUS_BBOX` in `config.py`), so
widening it is a one-line change if you'd rather have broader coverage at
the cost of a larger store, and `--spacing-deg` is a CLI override if you'd
rather go finer or coarser than 0.25° without touching the bounding box.

## Storage format

- Zarr store, dims `(point, time)`. `point` is a stable integer index into the
  land-filtered grid (sorted by lat, then lon); `lat`/`lon` are coordinates
  indexed by `point`. `time` is a daily `datetime64` index from 1990-01-01.
- All data variables are `float32`, compressed with `numcodecs.Zstd` at a
  configurable level (default 22, zstd's max/"ultra" level — full pull
  prioritizes minimizing the archive's footprint on the droplet over
  compression/decompression speed; range 1-22) — lossless beyond the
  float32 downcast itself, at any level.
- Chunked by point (default 50) and by ~1 year of days (default 365),
  configurable — sized for a ~512MB RAM serving target on the eventual
  droplet this store gets shipped to. That target is about decompressing
  and reading chunks back, not about writing them: each chunk is small
  regardless of level (well under 100KB per point/variable/year at this
  chunking), so zstd's "ultra" levels — which have a reputation for needing
  meaningfully more memory on *large* inputs — cost only a few KB of extra
  compression-context memory here, confirmed by direct measurement
  (`zstandard.ZstdCompressor.memory_size()`) rather than assumed; the actual
  cost of level 22 is slower compression per chunk (single-digit
  milliseconds, vs. fractions of a millisecond at level 12), which adds up
  to low tens of minutes across the whole archive — negligible next to a
  multi-day, network-paced full pull.
- Fetched data is downcast and written straight into the compressed store —
  there's no uncompressed intermediate file at any point.

## Error handling

Failed points (network errors that exhausted retries, or a bad/missing
response for a specific location) are skipped, not fatal — the run continues.
At the end, a summary is shown (in the GUI log/popup, or printed to the
console for the CLI) and the full list is written to
`<store_path>.failures_<timestamp>.json` next to the store.

## Interruptions and resuming

A full pull can take a long time (see the estimate the app shows before you
confirm — often days, not hours, at the default rate limit), so it's built to
survive being interrupted: reboot, sleep, closing the app, a power loss,
whatever. Progress is checkpointed to `<store_path>.progress.json` after
every fetched batch. Re-running the same mode against the same store just
picks up where it left off instead of starting over — it won't re-fetch
already-completed batches or wipe the store. If a run finishes completely,
that checkpoint stays in place, so re-running it again is a fast no-op rather
than an accidental full re-pull.

This only works as long as the underlying parameters haven't changed (grid,
chunk sizes, variables, compression level) — change any of those and it
starts a fresh run against that store path instead, same as if no checkpoint
existed. If you ever want to force a clean re-pull, delete both the
`.progress.json` file and the store folder.

## Seam for future phases

Don't reach into the zarr store directly from new code. Use
`weatherbot.inventory.open_store(path)` — it opens the store lazily via
xarray/dask (no data materializes until you actually index/compute it). This
is exactly what `weatherbot/train.py` (Phase 2, below) uses to read the
archive; nothing in that phase touches the zarr store's on-disk layout
directly.

Known limitation (by design, out of scope for the ingestion phase): a
store's time extent is fixed at creation time. Extending an existing store
with newer days without a full re-init is not built here — see "How
'continuous' actually behaves" under Phase 2 for what this means in practice
for training.

## Phase 2: Modeling & Forecasting

Two long-running background processes, both controlled and visualized from
one web dashboard:

- **Walk-forward training** (`weatherbot/train.py`, `weatherbot/model.py`) —
  continuously steps a lightweight forecasting model forward through the
  archive one calendar day at a time, in the same style used for financial
  time series: on each step, score the model's prediction against a day it
  hasn't seen yet (an honest out-of-sample error), *then* let it learn from
  that day before moving to the next one. There's no fixed end date — once
  it catches up to the most recent day the store has, it waits and resumes
  automatically if the store ever grows, and otherwise just keeps running
  until you stop it.
- **Monte Carlo simulation** (`weatherbot/simulate.py`) — using the
  currently-trained model, simulates many independent random future paths
  for one location, day by day, and summarizes them into an ensemble mean
  and percentile confidence bands (a "fan chart") rather than a single
  deterministic forecast.

### The model

Rather than one heavyweight model per grid point, every `(point, variable)`
pair — ~17,300 points x 18 variables ≈ 311,600 pairs at full CONUS scope —
gets its own small linear model: two Fourier harmonics of day-of-year (the
seasonal component) plus lag-1 and lag-7 autoregressive terms, 7 weights in
total. Every point and variable is updated together in one vectorized numpy
pass per calendar day — there's no per-point Python loop — which is what
keeps this affordable on a small droplet: the full-grid model checkpoint is
only a few MB, and one day's update is a handful of elementwise array
operations, not a training run in the usual sense.

Targets and lag features are tracked in a running-normalized space (an
adaptive mean/std per point/variable, see `model.py` for why a naive
first-sample estimate is unstable and what replaces it) so one global
learning rate behaves sensibly whether the variable is temperature (°C),
surface pressure (~1000 hPa), or precipitation (mostly 0, occasionally
large) — without that, variables on very different scales would need
separate tuning or would swamp each other.

### How "continuous" actually behaves

Training has no defined stopping condition other than the Stop button —
that part is real. But it's still bounded by what the store actually
contains: as noted above, a store's time extent is fixed when it's created,
so "waiting for new data" in practice means waiting for you to run a fresh
`fetch --mode full` that produces a store covering more days (or for
previously-failed points to get backfilled into the *same* range by a
resumed pull). The dashboard's status line makes the current phase explicit
(`catching_up` vs. `live_wait`) so this is never ambiguous while it's
running.

Progress is checkpointed to `<store_path>.model.npz` (the weights and all
running statistics) and `<store_path>.train_state.json` (current position)
— every ~200 days during backlog catch-up, and every day once caught up to
live polling (trivial overhead at that point). A restart — crash, `Stop`,
`systemctl restart`, a reboot — resumes from the last checkpoint instead of
re-walking from 1990. A `<store_path>.train.lock` file stops two training
processes from ever running against the same store at once (e.g. the
headless `train` CLI started alongside the dashboard by accident); delete it
by hand only if you're sure the process that made it is actually gone.

The per-day error log (`<store_path>.train_metrics.jsonl`, what feeds the
training chart) is capped at 10,000 rows — once it grows past that, every
other row is dropped, so recent history stays dense and old history gets
progressively coarser instead of the file growing without bound over years
of continuous operation.

### Monte Carlo simulation

Each simulated path starts from the model's current lag state at the chosen
point, then for each forecast day: predict (seasonal + lag terms), add
Gaussian noise sized to that point/variable's own tracked residual variance,
and feed the noisy result back in as next day's lag-1. Paths diverge from
each other over the horizon this way — uncertainty compounds the further out
the forecast goes, rather than every path just being the same central
forecast with independent per-day noise. At the end, each forecast day's
values across all paths are summarized into percentiles (5/25/50/75/95) for
the confidence-band chart.

This is deliberately a small, on-demand computation — one point, up to 500
paths, up to a 60-day horizon (both capped in `simulate.py` for this
reason) — not something run across the whole grid, so it stays cheap enough
to re-run interactively from the dashboard on a small droplet.

### The dashboard

```bash
.venv\Scripts\python.exe -m weatherbot serve --store data/weather_archive.zarr
```

Opens on `http://127.0.0.1:8000` by default (`--host`/`--port` to change).
Two panels:

- **Walk-forward training** — Start/Stop buttons, current phase/date/
  progress, and a live chart of MAE/RMSE as the model steps through history
  (or waits for more of it).
- **Monte Carlo simulation** — pick a location (grid point ID, or nearest to
  a typed latitude/longitude), variable(s), horizon, and ensemble size, then
  Start/Stop; a spaghetti plot of individual paths fills in day by day as
  they're generated, alongside a confidence-band chart building up the same
  way.

Both processes run as background threads inside the dashboard server itself
— stopping either just sets a flag the corresponding loop checks between
steps (between one day of training, or one forecast day of simulation), so
it always finishes its current step and checkpoints/exits cleanly rather
than being killed mid-write. The dashboard polls its own small status
endpoints roughly once a second rather than using websockets/SSE — simpler
to run correctly on Flask's built-in server, and indistinguishable from
"live" at that update rate.

For headless training with no web UI (e.g. a minimal droplet, or debugging):

```bash
.venv\Scripts\python.exe -m weatherbot train --store data/weather_archive.zarr
```

Ctrl-C (or `systemctl stop`, which sends the same signal the service unit
below relies on) stops it the same clean way as the dashboard's Stop button.
Don't run this alongside the dashboard against the same store — the lock
file will refuse the second one.

### Resource footprint (why this fits a small droplet)

At the current CONUS/0.25° scope (~17,300 points, 1990-present, 18
variables, ~311,600 point/variable pairs — roughly 4x the pairs a 0.5°
grid had):

- **Model checkpoint**: ~43MB uncompressed (weights + running stats + lag
  history for every point/variable pair; scales linearly with pair count,
  so ~4x the previous ~10MB), smaller on disk via `np.savez_compressed`.
  Still trivial to read/write/checkpoint repeatedly on a small droplet.
- **Training memory**: backlog catch-up reads the store in blocks
  (`BLOCK_DAYS` in `train.py`, now 21 days — cut from the previous 90 so
  that `n_points x BLOCK_DAYS` stays roughly constant as the grid got ~4x
  denser) rather than the whole 36-year history at once — roughly 25MB
  resident at a time, freed after each block, matching the original
  ~25-30MB target this was sized for at the smaller grid. Live polling
  once caught up reads a single day at a time (negligible). If the grid
  ever changes again, rescale `BLOCK_DAYS` so `n_points x BLOCK_DAYS`
  stays close to today's ~360,000 to hold this steady.
- **Training CPU**: one day's update is a handful of vectorized numpy
  operations over ~311,600 (point, variable) pairs — roughly 4x the
  per-day arithmetic of the previous scope, still just a few million
  floating point ops per day. Catching up through the full 1990-present
  backlog is expected to take single-digit minutes of CPU time on a small
  droplet, most of it spent decompressing zarr chunks rather than
  computing.
- **Monte Carlo memory/CPU**: one point, capped at 500 paths x 60 days x
  however many variables you pick — a few hundred KB and well under a
  second of compute, regardless of grid size (unaffected by the spacing
  change), since it only ever touches one point's model state.

### Deploying the dashboard on a droplet

`weatherbot-dashboard.service` mirrors `weatherbot.service` (see "Running
full pull on a droplet" above) — edit `YOUR_USERNAME` and the paths, then:

```bash
sudo cp weatherbot-dashboard.service /etc/systemd/system/weatherbot-dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable --now weatherbot-dashboard
```

**The dashboard has no authentication or HTTPS** — its Start/Stop controls
are not something to expose directly to the public internet. If you bind it
to `0.0.0.0` so it's reachable from outside the droplet, put it behind a
firewall rule limited to your own IP, or better, leave it bound to
`127.0.0.1` (the default) and reach it through an SSH tunnel instead:

```bash
ssh -L 8000:localhost:8000 your_user@droplet_ip
```

then open `http://localhost:8000` locally, same as if it were running on
your own machine.
