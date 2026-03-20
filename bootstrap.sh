#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  bootstrap.sh — One-command setup for AI Scene Style Kiosk
#
#  Usage:
#    git clone <repo-url> && cd facemorph-kiosk && bash bootstrap.sh
#
#  What it does:
#    1. Installs system packages (build tools, OpenCV deps, etc.)
#    2. Creates a Python venv (kiosk_venv)
#    3. Installs PyTorch with ROCm support (AMD GPU)
#    4. Installs all Python dependencies
#    5. Downloads all AI models (~15 GB total)
#    6. Sets GPU device permissions
#    7. Prints instructions to start the kiosk
#
#  Requirements:
#    - Ubuntu 22.04+ (or similar Debian-based distro)
#    - AMD GPU with ROCm support (RX 7000 series tested)
#    - ~20 GB free disk space for models
#    - Internet connection for downloads
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/kiosk_venv"
ROCM_TORCH_URL="https://download.pytorch.org/whl/rocm6.2"

echo ""
echo "======================================================="
echo "  AI Scene Style Kiosk — Bootstrap Setup"
echo "======================================================="
echo ""

# ── 1. System packages ──────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-dev python3-pip \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libopencv-dev \
    wget curl git \
    2>/dev/null
echo "  [OK] System packages installed"

# ── 2. Python venv ──────────────────────────────────────────────────
echo ""
echo "[2/6] Setting up Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  [OK] venv already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "  [OK] Created venv at $VENV_DIR"
fi

# Activate venv for the rest of the script
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ── 3. PyTorch with ROCm ────────────────────────────────────────────
echo ""
echo "[3/6] Installing PyTorch with ROCm support..."
echo "       (This may take a few minutes)"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  [OK] PyTorch with GPU support already installed"
    python -c "import torch; print(f'  PyTorch {torch.__version__} — GPU: {torch.cuda.get_device_name(0)}')"
else
    pip install torch torchvision torchaudio --index-url "$ROCM_TORCH_URL" -q
    echo "  [OK] PyTorch installed"
    python -c "import torch; print(f'  PyTorch {torch.__version__}')"
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
    else
        echo "  [WARN] GPU not detected — check ROCm drivers"
    fi
fi

# ── 4. Python dependencies ──────────────────────────────────────────
echo ""
echo "[4/6] Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" -q
# huggingface-hub is needed by setup_models.py but not in requirements.txt
pip install huggingface-hub -q
echo "  [OK] All Python packages installed"

# ── 5. Download AI models ───────────────────────────────────────────
echo ""
echo "[5/6] Downloading AI models (~15 GB, may take a while)..."
echo "       Models are cached in ~/.cache/huggingface/ and ~/kiosk_models/"
echo ""
python "$SCRIPT_DIR/setup_models.py"

# ── 6. GPU permissions ──────────────────────────────────────────────
echo ""
echo "[6/6] Setting GPU device permissions..."
for dev in /dev/kfd /dev/dri/renderD128; do
    if [ -e "$dev" ]; then
        sudo chmod 666 "$dev" 2>/dev/null && echo "  [OK] $dev" || echo "  [WARN] Could not set $dev permissions"
    fi
done

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Setup complete!"
echo ""
echo "  To start the kiosk:"
echo ""
echo "    source kiosk_venv/bin/activate"
echo "    python start.py"
echo ""
echo "  Then open http://localhost:8000 in a browser."
echo "======================================================="
echo ""
