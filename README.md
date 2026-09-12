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

Double-click **`WeatherBot.bat`**. A window opens with four buttons:

- **Run Test Pull** — 20 grid points, last 10 years of data. Finishes in a
  couple of minutes; use this to confirm everything works before committing
  to the full pull. Writes to `data/weather_test.zarr`.
- **Run Full Pull** — the real archive: ~19,000-26,000 land grid points across
  North America, 1940-present, roughly 18-27GB compressed. The app shows an
  exact estimate (points/days/requests/size/best-case time) before you
  confirm — see "Fetch performance" below for what actually governs how long
  it takes. Leave the computer on and connected to the internet while it
  runs. Writes to `data/weather_archive.zarr`.
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
.venv\Scripts\python.exe -m weatherbot fetch --mode full --spacing-deg 0.15 \
    --point-chunk 50 --time-chunk-days 365 --zstd-level 12 \
    --batch-size 20 --time-chunk-years 3 --full-pull-interval-sec 9
```

(`--rate-limit`/`--concurrency` only affect test mode — full pull always
paces at `--full-pull-interval-sec`, sequentially. See "Fetch performance".)

## Fetch performance

Open-Meteo's free/keyless tier has three request-volume caps: 600/minute,
5,000/hour, and 10,000/day. For a one-off small pull, the minute/hour caps
are what you'd notice. For the **full pull specifically** — tens of
thousands of requests over multiple days — the **daily** cap is the one that
actually governs total completion time: bursting up to the minute/hour
limits just hits the daily one sooner and then sits idle until it resets,
it doesn't finish any sooner overall. So full pull and test mode are paced
differently on purpose:

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
  against the live API, not guessed. This is what got full pull's request
  count down to ~36,700 (from ~111,000 at the original 10x2 defaults) — at
  the fixed 9s pace, that's the difference between an ~11-day run and an
  ~3.8-day one.
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

At the fixed full-pull pace, the app's upfront estimate (~3.8 days at
current defaults) is the actual expected duration, not just a best case —
verify it against the live ETA once you start a run.

## Running full pull on a droplet

A ~3.8-day continuous run isn't something to leave running on a Windows
machine you need to use/sleep/reboot. Full pull mode is plain CLI — no GUI
dependency at all — so it runs fine headless on a small Linux droplet; test
mode stays a local Windows GUI run as before, this section is full pull only.

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
expected to run for days, a **systemd service** is the more robust choice —
it restarts automatically on a crash or reboot, with no one needing to
notice and re-launch it:

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

The estimated compressed store size is **~18-27GB**, and this droplet's SSD
is **~25GB total**, some of which the OS/Python/venv already use. **The
store alone may not comfortably fit**, and definitely won't if you also try
to make a copy of it (see below). Before starting a multi-day run, either:

- Attach a separate DigitalOcean **Volume** (block storage) and point
  `--store` at a path on it, so the store doesn't compete with the boot
  disk at all, or
- Resize the droplet to a larger disk tier first.

Either way, keep an eye on `df -h` during the run — it's much better to
notice this early than to have the pull fail from a full disk on day 3.

### Downloading the finished store

**Don't tar it first.** A zarr store this size, tar'd into a single archive
*on the same disk*, needs roughly double the space (original + archive) —
which won't fit on an already-tight 25GB disk. Transfer the directory
directly instead; `rsync` handles "one large directory of many small files"
far better than a naive recursive copy, and — usefully for a large transfer
over a home connection — it's resumable: if it drops partway through,
re-running the exact same command only transfers what's missing/changed.

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
xarray/dask (no data materializes until you actually index/compute it), and
is the intended entry point for the walk-forward training, Monte Carlo
simulation, and dashboard phases that come after this one.

Known limitation (by design, out of scope for this phase): a store's time
extent is fixed at creation time. Extending an existing store with newer days
without a full re-init is a future "incremental update" feature, not built
here.
