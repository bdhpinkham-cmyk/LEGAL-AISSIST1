#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pro Se Legal Intelligence — one-step launcher (macOS / Linux)
#
# Double-click (or run `./run.sh`). On first run it creates a virtual
# environment and installs dependencies; after that it just launches the app.
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

# Pick a Python 3 interpreter.
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3.11+ is required but was not found. Install it from https://www.python.org/downloads/"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "First-time setup: creating a virtual environment..."
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Checking dependencies (first run may take a minute)..."
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Optional: install the browser used by the Court Portal tab. Skipped silently
# if Playwright isn't available — every other feature works without it.
python -m playwright install chromium >/dev/null 2>&1 || true

echo "Launching Pro Se Legal Intelligence..."
python main.py
