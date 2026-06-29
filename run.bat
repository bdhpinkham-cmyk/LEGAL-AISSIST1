@echo off
REM ---------------------------------------------------------------------------
REM Pro Se Legal Intelligence - one-step launcher (Windows)
REM
REM Double-click this file. On first run it creates a virtual environment and
REM installs dependencies; after that it just launches the app.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>&1
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    echo Python 3.11+ is required but was not found.
    echo Install it from https://www.python.org/downloads/ ^(check "Add Python to PATH"^).
    pause
    exit /b 1
  )
)

if not exist ".venv" (
  echo First-time setup: creating a virtual environment...
  %PY% -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo Checking dependencies ^(first run may take a minute^)...
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

REM Optional browser for the Court Portal tab; ignored if unavailable.
python -m playwright install chromium >nul 2>&1

echo Launching Pro Se Legal Intelligence...
python main.py

if %errorlevel% neq 0 (
  echo.
  echo The app exited with an error. Press any key to close.
  pause >nul
)
