#!/bin/bash
# start.sh — AMD Adapt Kiosk
# Usage: bash ~/facemorph-kiosk/start.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "[ERROR] venv not found. Run: bash setup_ubuntu.sh"
    exit 1
fi

cd "$SCRIPT_DIR"
source "$VENV/bin/activate"

# Safety net — fix opencv if mediapipe clobbered it
python3 -c "import cv2; cv2.face.LBPHFaceRecognizer_create()" 2>/dev/null || {
    echo "[..] Fixing opencv-contrib..."
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
    pip install --no-cache-dir "opencv-contrib-python>=4.9.0.80" -q
}

echo ""
echo "============================================================"
echo "  AMD ADAPT KIOSK — http://localhost:8000"
echo "============================================================"
echo ""

python3 main.py
