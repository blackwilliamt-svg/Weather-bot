@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Double-click setup.bat first.
    pause
    exit /b 1
)

start "" http://127.0.0.1:8000/
".venv\Scripts\python.exe" -m weatherbot serve --store data\weather_archive.zarr
