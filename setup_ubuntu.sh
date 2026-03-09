#!/bin/bash
# setup_ubuntu.sh — AMD Adapt Kiosk full setup
# Ubuntu 22.04 or 24.04 · AMD W7900 + ROCm
set -e

echo ""
echo "======================================================"
echo "  AMD ADAPT KIOSK — Ubuntu Setup"
echo "======================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[INFO] Repo: $SCRIPT_DIR"

# ── 0. Fix broken packages ─────────────────────────────────────────────────
echo ""
echo "── Step 0: System prep ──"
sudo apt --fix-broken install -y 2>/dev/null || true
sudo dpkg --configure -a 2>/dev/null || true

# ── 1. System packages ─────────────────────────────────────────────────────
echo ""
echo "── Step 1: System packages ──"
sudo apt update -qq
sudo apt install -y \
    python3 python3-venv python3-pip python-is-python3 \
    git wget curl gnupg2 build-essential

# ── 2. ROCm / GPU permissions ──────────────────────────────────────────────
echo ""
echo "── Step 2: GPU ──"
if command -v rocm-smi &>/dev/null; then
    echo "[OK] ROCm detected"
    sudo chmod 666 /dev/kfd 2>/dev/null || true
    sudo chmod 666 /dev/dri/renderD128 2>/dev/null || true
    sudo usermod -a -G render,video "$USER" 2>/dev/null || true
    IS_ROCM=true
else
    echo "[WARN] ROCm not found — run: bash install_rocm.sh first"
    IS_ROCM=false
fi

# ── 3. Kiosk venv ──────────────────────────────────────────────────────────
echo ""
echo "── Step 3: Kiosk Python environment ──"
cd "$SCRIPT_DIR"

# Always start fresh
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools -q

# PyTorch — ROCm or CPU
if [ "$IS_ROCM" = true ]; then
    echo "[..] Installing PyTorch ROCm 6.2..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 -q
else
    echo "[..] Installing PyTorch CPU..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

# Core packages — install mediapipe early so it can pull in opencv-python
pip install -q \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "numpy>=2.0.0" \
    "Pillow>=11.0.0" \
    "mediapipe>=0.10.30,<0.11" \
    "rembg" \
    "python-multipart==0.0.9" \
    "aiofiles==23.2.1" \
    "requests==2.31.0" \
    "onnxruntime"

# Install opencv-contrib LAST — mediapipe pulls in opencv-python which
# overwrites it, so we force reinstall contrib after everything else
echo "[..] Installing opencv-contrib-python (last, after mediapipe)..."
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
pip install --no-cache-dir "opencv-contrib-python>=4.9.0.80"

# Verify cv2.face is available
python3 -c "import cv2; cv2.face.LBPHFaceRecognizer_create(); print('[OK] cv2.face OK — version', cv2.__version__)"

deactivate
echo "[OK] Kiosk environment ready"

# ── 4. ComfyUI ─────────────────────────────────────────────────────────────
echo ""
echo "── Step 4: ComfyUI ──"
COMFY_DIR="$HOME/ComfyUI"
COMFY_VENV="$HOME/comfyui-venv"

# Always reinstall ComfyUI cleanly
rm -rf "$COMFY_DIR" "$COMFY_VENV"

echo "[..] Cloning ComfyUI v0.3.10..."
git clone --branch v0.3.10 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" 2>/dev/null || \
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"

python3 -m venv "$COMFY_VENV"
source "$COMFY_VENV/bin/activate"
pip install --upgrade pip wheel -q

if [ "$IS_ROCM" = true ]; then
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 -q
else
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

# Install ComfyUI deps (skip torch lines, skip comfy-aimdo)
cd "$COMFY_DIR"
grep -v "^torch\|^torchvision\|^torchaudio\|comfy.aimdo\|comfy-aimdo" requirements.txt > /tmp/comfy_req.txt
pip install -q -r /tmp/comfy_req.txt
pip install -q torchsde kornia spandrel requests
pip uninstall -y comfy-aimdo 2>/dev/null || true
cd "$SCRIPT_DIR"

# Verify GPU
python3 -c "
import torch
ok = torch.cuda.is_available()
print(f'[{\"OK\" if ok else \"WARN\"}] GPU visible: {ok}')
if ok: print(f'     GPU: {torch.cuda.get_device_name(0)}')
"
deactivate

# ── 5. Models ──────────────────────────────────────────────────────────────
echo ""
if [ "$IS_ROCM" = true ]; then
    echo "── Step 5: AI Models ──"
    MODELS="$COMFY_DIR/models"
    mkdir -p "$MODELS/checkpoints" "$MODELS/controlnet"

    SDXL="$MODELS/checkpoints/sd_xl_turbo_1.0_fp16.safetensors"
    if [ ! -f "$SDXL" ]; then
        echo "[..] Downloading SDXL-Turbo (~3.1 GB)..."
        wget --show-progress -q \
            "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors" \
            -O "$SDXL"
        echo "[OK] SDXL-Turbo done"
    else
        echo "[OK] SDXL-Turbo already present — skipping download"
    fi

    CNET="$MODELS/controlnet/control-lora-canny-rank128.safetensors"
    if [ ! -f "$CNET" ]; then
        echo "[..] Downloading ControlNet (~660 MB)..."
        wget --show-progress -q \
            "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors" \
            -O "$CNET"
        echo "[OK] ControlNet done"
    else
        echo "[OK] ControlNet already present — skipping download"
    fi
else
    echo "── Step 5: Models skipped (ROCm not detected) ──"
fi

# ── 6. Copy launch script ──────────────────────────────────────────────────
echo ""
echo "── Step 6: Launch scripts ──"
cp "$SCRIPT_DIR/start_comfyui.sh" "$HOME/start_comfyui.sh"
chmod +x "$HOME/start_comfyui.sh"
echo "[OK] ~/start_comfyui.sh installed"

echo ""
echo "======================================================"
echo "  Setup Complete!"
echo "======================================================"
echo ""
echo "  Every time you want to run the kiosk:"
echo ""
echo "  Terminal 1:  ~/start_comfyui.sh"
echo "  Terminal 2:  bash $SCRIPT_DIR/start.sh"
echo ""
