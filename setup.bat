@echo off
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on your PATH.
    echo Install Python 3.11+ from https://www.python.org/downloads/ and check
    echo "Add Python to PATH" during setup, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Installing dependencies (this can take a few minutes the first time)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Setup complete. Double-click WeatherBot.bat to launch the app.
pause
