@echo off
REM start.bat — convenience launcher for cmd.exe users
cd /d "%~dp0"
if not exist "kiosk_venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run bootstrap.ps1 first.
    exit /b 1
)
call "kiosk_venv\Scripts\activate.bat"
python start.py
