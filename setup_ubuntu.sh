#!/bin/bash
# setup_ubuntu.sh — AMD Adapt Kiosk full setup
# Ubuntu 22.04 or 24.04 · AMD GPU (W7900 recommended for AI generation)

echo ""
echo "======================================================"
echo "  AMD ADAPT KIOSK — Ubuntu Setup"
echo "======================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IS_ROCM=false
UBUNTU_VER=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
echo "[INFO] Ubuntu $UBUNTU_VER detected"

if [ "$UBUNTU_VER" = "22.04" ]; then
    CODENAME="jammy"
else
    CODENAME="noble"
fi

# ── 0. Fix broken packages ────────────────────────────────────────────────────
echo ""
echo "── Step 0: Fixing package state ──"
sudo apt --fix-broken install -y 2>/dev/null || true
sudo dpkg --configure -a 2>/dev/null || true

# ── 1. Base packages ──────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Base packages ──"
sudo apt update -qq
sudo apt install -y python3-venv python3-pip git wget curl gnupg2

# ── 2. GPU detection ──────────────────────────────────────────────────────────
echo ""
echo "── Step 2: GPU detection ──"

if command -v rocm-smi &>/dev/null; then
    # ROCm already installed — always use ROCm PyTorch
    IS_ROCM=true
    GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | grep -i "card\|product\|series" | head -3)
    echo "[OK] ROCm is installed"
    echo "[OK] GPU info: $GPU_INFO"
    echo "[OK] Will use ROCm PyTorch for full GPU acceleration"
else
    echo "[INFO] ROCm not detected — run bash install_rocm.sh first, then log out/in, then re-run this script"
    echo "[INFO] Continuing with CPU mode for now..."
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

if [ ! -f "venv/bin/activate" ]; then
    echo "[ERR] Failed to create venv. Run: sudo apt install python3-venv"
    exit 1
fi

source venv/bin/activate
pip install --upgrade pip wheel -q

if [ "$IS_ROCM" = true ]; then
    echo "[..] Installing PyTorch with ROCm 6.2 (GPU accelerated)..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 -q
else
    echo "[..] Installing PyTorch CPU..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

echo "[..] Installing kiosk packages..."
pip install -q \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "opencv-python>=4.9.0.80" \
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

# ── 4. ComfyUI ────────────────────────────────────────────────────────────────
echo ""
echo "── Step 4: ComfyUI ──"
COMFY_DIR="$HOME/ComfyUI"

if [ ! -d "$COMFY_DIR" ]; then
    echo "[..] Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
    echo "[OK] ComfyUI already present"
    cd "$COMFY_DIR" && git pull -q && cd "$SCRIPT_DIR"
fi

COMFY_VENV="$HOME/comfyui-venv"
if [ ! -d "$COMFY_VENV" ]; then
    python3 -m venv "$COMFY_VENV"
fi

source "$COMFY_VENV/bin/activate"
pip install --upgrade pip wheel -q

if [ "$IS_ROCM" = true ]; then
    echo "[..] Installing PyTorch ROCm for ComfyUI..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 -q
else
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

cd "$COMFY_DIR"
grep -v "^torch\|^torchvision\|^torchaudio" requirements.txt > /tmp/comfy_req.txt
pip install -q -r /tmp/comfy_req.txt
cd "$SCRIPT_DIR"
echo "[OK] ComfyUI installed"

# ── 5. Models ─────────────────────────────────────────────────────────────────
echo ""
if [ "$IS_ROCM" = true ]; then
    echo "── Step 5: AI Models ──"
    MODELS="$COMFY_DIR/models"
    mkdir -p "$MODELS/checkpoints" "$MODELS/controlnet"

    SDXL="$MODELS/checkpoints/sd_xl_turbo_1.0_fp16.safetensors"
    if [ ! -f "$SDXL" ]; then
        echo "[..] Downloading SDXL-Turbo (~3.1 GB) — this takes a while..."
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
    echo "── Step 5: Models skipped (ROCm not installed) ──"
    echo "[INFO] Run bash install_rocm.sh, log out/in, then re-run this script"
fi

# ── 6. Launch script ──────────────────────────────────────────────────────────
echo ""
echo "── Step 6: Launch scripts ──"
cp "$SCRIPT_DIR/start_comfyui.sh" "$HOME/start_comfyui.sh"
chmod +x "$HOME/start_comfyui.sh"
echo "[OK] ~/start_comfyui.sh created"

deactivate

echo ""
echo "======================================================"
echo "  Setup Complete!"
echo "======================================================"
echo ""
if [ "$IS_ROCM" = true ]; then
    echo "  Full AI generation enabled!"
    echo ""
    echo "  Terminal 1:  ~/start_comfyui.sh"
    echo "  Terminal 2:  cd $SCRIPT_DIR && source venv/bin/activate && python start.py"
else
    echo "  ROCm not found — live transforms only"
    echo "  Run: bash install_rocm.sh, log out/in, then bash setup_ubuntu.sh again"
fi
echo ""
