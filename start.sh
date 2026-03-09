#!/bin/bash
# start.sh — AMD Adapt Kiosk launcher
# Usage: bash ~/facemorph-kiosk/start.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "[ERROR] venv not found. Run: bash setup_ubuntu.sh"
    exit 1
fi

cd "$SCRIPT_DIR"
source "$VENV/bin/activate"

# Install any missing packages silently
pip install -q uvicorn fastapi opencv-contrib-python 2>/dev/null

echo ""
echo "============================================================"
echo "  AMD ADAPT KIOSK — http://localhost:8000"
echo "============================================================"
echo ""

python3 main.py
