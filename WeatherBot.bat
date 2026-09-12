@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Double-click setup.bat first.
    pause
    exit /b 1
)

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m weatherbot.gui
) else (
    start "" ".venv\Scripts\python.exe" -m weatherbot.gui
)
