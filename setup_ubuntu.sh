#!/bin/bash
# setup_ubuntu.sh — AMD Adapt Kiosk full setup
# Ubuntu 22.04 or 24.04 · AMD W7900 + ROCm

echo ""
echo "======================================================"
echo "  AMD ADAPT KIOSK — Ubuntu Setup"
echo "======================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UBUNTU_VER=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
echo "[INFO] Ubuntu $UBUNTU_VER detected"

# ── 0. Fix broken packages ────────────────────────────────────────────────────
echo ""
echo "── Step 0: Fixing package state ──"
sudo apt --fix-broken install -y 2>/dev/null || true
sudo dpkg --configure -a 2>/dev/null || true

# ── 1. Base packages ──────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Base packages ──"
sudo apt update -qq
sudo apt install -y python3-venv python3-pip python-is-python3 git wget curl gnupg2

# ── 2. GPU + ROCm device permissions ─────────────────────────────────────────
echo ""
echo "── Step 2: GPU setup ──"
if command -v rocm-smi &>/dev/null; then
    echo "[OK] ROCm detected"
    rocm-smi 2>/dev/null | head -5 || true
    # Fix device permissions so PyTorch can see the GPU without full logout
    sudo chmod 666 /dev/kfd 2>/dev/null || true
    sudo chmod 666 /dev/dri/renderD128 2>/dev/null || true
    sudo usermod -a -G render,video $USER 2>/dev/null || true
    IS_ROCM=true
else
    echo "[WARN] ROCm not found — run bash install_rocm.sh first"
    IS_ROCM=false
fi

# ── 3. Kiosk Python venv ──────────────────────────────────────────────────────
echo ""
echo "── Step 3: Kiosk Python dependencies ──"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv 2>/dev/null || {
        sudo apt install -y python3-venv
        python3 -m venv venv
    }
fi

source venv/bin/activate
pip install --upgrade pip wheel -q

# Ensure opencv-contrib (not base opencv) — they conflict
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true

if [ "$IS_ROCM" = true ]; then
    echo "[..] Installing PyTorch ROCm 6.2..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 -q
else
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

pip install -q \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "opencv-contrib-python>=4.9.0.80" \
    "numpy>=2.0.0" \
    "Pillow>=11.0.0" \
    "mediapipe>=0.10.30,<0.11" \
    "rembg" \
    "python-multipart==0.0.9" \
    "aiofiles==23.2.1" \
    "requests==2.31.0" \
    "onnxruntime"
pip install -q "insightface==0.7.3" 2>/dev/null || true
echo "[OK] Kiosk packages installed"
deactivate

# ── 4. ComfyUI ────────────────────────────────────────────────────────────────
echo ""
echo "── Step 4: ComfyUI ──"
COMFY_DIR="$HOME/ComfyUI"

# Remove any existing broken install
if [ -d "$COMFY_DIR" ]; then
    echo "[..] Removing existing ComfyUI install..."
    rm -rf "$COMFY_DIR"
fi

# Clone known-good version that works on AMD ROCm without comfy-aimdo issues
echo "[..] Cloning ComfyUI v0.3.10 (AMD-compatible)..."
git clone --branch v0.3.10 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" 2>/dev/null || \
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"

# ComfyUI venv
COMFY_VENV="$HOME/comfyui-venv"
if [ -d "$COMFY_VENV" ]; then
    rm -rf "$COMFY_VENV"
fi
python3 -m venv "$COMFY_VENV"
source "$COMFY_VENV/bin/activate"
pip install --upgrade pip wheel -q

# Ensure opencv-contrib (not base opencv) — they conflict
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true

if [ "$IS_ROCM" = true ]; then
    echo "[..] Installing ROCm PyTorch for ComfyUI..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 -q
else
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

# Remove comfy-aimdo if it snuck in
pip uninstall -y comfy-aimdo 2>/dev/null || true

cd "$COMFY_DIR"
grep -v "^torch\|^torchvision\|^torchaudio\|comfy.aimdo\|comfy-aimdo" requirements.txt > /tmp/comfy_req.txt
pip install -q -r /tmp/comfy_req.txt
pip install -q torchsde kornia spandrel requests
cd "$SCRIPT_DIR"

# Verify GPU visible
echo ""
echo "[..] Verifying GPU..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU visible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('[OK] GPU ready!')
else:
    print('[WARN] GPU not visible yet — may need logout/login')
"
deactivate

# ── 5. Models ─────────────────────────────────────────────────────────────────
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
        echo "[OK] SDXL-Turbo downloaded"
    else
        echo "[OK] SDXL-Turbo already present"
    fi

    CNET="$MODELS/controlnet/control-lora-canny-rank128.safetensors"
    if [ ! -f "$CNET" ]; then
        echo "[..] Downloading ControlNet (~660 MB)..."
        wget --show-progress -q \
            "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors" \
            -O "$CNET"
        echo "[OK] ControlNet downloaded"
    else
        echo "[OK] ControlNet already present"
    fi
else
    echo "── Step 5: Models skipped (no ROCm) ──"
fi

# ── 6. Launch scripts ─────────────────────────────────────────────────────────
echo ""
echo "── Step 6: Launch scripts ──"
cp "$SCRIPT_DIR/start_comfyui.sh" "$HOME/start_comfyui.sh"
chmod +x "$HOME/start_comfyui.sh"
echo "[OK] ~/start_comfyui.sh ready"

echo ""
echo "======================================================"
echo "  Setup Complete!"
echo "======================================================"
echo ""
if [ "$IS_ROCM" = true ]; then
    echo "  Terminal 1:  ~/start_comfyui.sh"
    echo "  Terminal 2:  cd $SCRIPT_DIR && source venv/bin/activate && python start.py"
else
    echo "  Run bash install_rocm.sh first, then re-run this script"
fi
echo ""
