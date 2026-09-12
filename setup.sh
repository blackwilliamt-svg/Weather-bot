#!/usr/bin/env bash
# One-time setup for running WeatherBot's full pull on a headless Linux
# droplet. Mirrors setup.bat's job on Windows: create a venv, install deps.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. On Ubuntu/Debian: sudo apt update && sudo apt install -y python3 python3-venv"
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup complete. See README.md's 'Running full pull on a droplet' section"
echo "for how to start it and leave it running unattended."
