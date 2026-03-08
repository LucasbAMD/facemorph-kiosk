#!/bin/bash
# setup_ubuntu.sh — AMD Adapt Kiosk full setup
# Ubuntu 22.04 or 24.04 · AMD GPU (W7900 recommended for AI generation)
# Run as normal user (not root) — uses sudo internally where needed

set -e

echo ""
echo "======================================================"
echo "  AMD ADAPT KIOSK — Ubuntu Setup"
echo "======================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IS_W7900=false
UBUNTU_VER=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
echo "[INFO] Ubuntu $UBUNTU_VER detected"

# ── 0. Fix any held broken packages first ─────────────────────────────────────
echo ""
echo "── Step 0: Fixing package state ──"
sudo apt --fix-broken install -y 2>/dev/null || true
sudo dpkg --configure -a 2>/dev/null || true

# ── 1. Python venv support ────────────────────────────────────────────────────
echo ""
echo "── Step 1: Python venv ──"
sudo apt update -qq
sudo apt install -y python3-venv python3-pip git wget curl

# ── 2. Check GPU ──────────────────────────────────────────────────────────────
echo ""
echo "── Step 2: GPU detection ──"

if command -v rocm-smi &>/dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep "Card series" | head -1 | awk -F: '{print $2}' | xargs)
    echo "[OK] ROCm found. GPU: $GPU_NAME"
    if echo "$GPU_NAME" | grep -qi "W7900\|7900 XTX\|7900 XT"; then
        IS_W7900=true
        echo "[OK] W7900/RDNA3 detected — AI generation will be enabled"
    else
        echo "[INFO] Non-W7900 GPU — AI generation will run on CPU (slower but works)"
    fi
else
    echo "[INFO] ROCm not installed"
    # Only install ROCm if this looks like a discrete AMD GPU worth it
    # For laptop APUs (Phoenix, Hawk Point) skip ROCm — CPU is fine
    GPU_INFO=$(lspci 2>/dev/null | grep -i "VGA\|Display\|3D" || echo "unknown")
    echo "[INFO] GPU: $GPU_INFO"
    
    if echo "$GPU_INFO" | grep -qi "Navi\|W7900\|RX 7\|RX 6"; then
        echo "[INFO] Discrete AMD GPU found — installing ROCm..."
        
        sudo apt install -y wget gnupg2
        wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | \
            sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg 2>/dev/null || \
            sudo apt-key add - 2>/dev/null

        if [ "$UBUNTU_VER" = "22.04" ]; then
            CODENAME="jammy"
        else
            CODENAME="noble"
        fi

        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/6.2 $CODENAME main" | \
            sudo tee /etc/apt/sources.list.d/rocm.list

        sudo apt update -qq
        # Install just what we need, not the full HIP SDK which has broken deps
        sudo apt install -y rocm-opencl-runtime 2>/dev/null || \
            echo "[WARN] ROCm install had issues — continuing without it"

        sudo usermod -a -G render,video $USER
        echo "[INFO] Added to render/video groups. May need to log out/in for full effect."
    else
        echo "[INFO] No discrete AMD GPU — will run on CPU (fine for laptop testing)"
    fi
fi

# ── 3. Kiosk Python venv ──────────────────────────────────────────────────────
echo ""
echo "── Step 3: Kiosk Python dependencies ──"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[OK] venv created"
fi

source venv/bin/activate
pip install --upgrade pip wheel -q

# PyTorch — ROCm if W7900, CPU otherwise
if [ "$IS_W7900" = true ]; then
    echo "[..] Installing PyTorch with ROCm 6.2..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 -q
else
    echo "[..] Installing PyTorch (CPU) — fine for testing..."
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

# InsightFace — may fail on some Python versions, that's ok
pip install -q "insightface==0.7.3" 2>/dev/null || \
    echo "[WARN] InsightFace skipped (optional)"

echo "[OK] Kiosk packages installed"

# ── 4. ComfyUI ────────────────────────────────────────────────────────────────
echo ""
echo "── Step 4: ComfyUI ──"
COMFY_DIR="$HOME/ComfyUI"

if [ ! -d "$COMFY_DIR" ]; then
    echo "[..] Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
    echo "[OK] ComfyUI already cloned — pulling latest..."
    cd "$COMFY_DIR" && git pull -q && cd "$SCRIPT_DIR"
fi

# ComfyUI venv
COMFY_VENV="$HOME/comfyui-venv"
if [ ! -d "$COMFY_VENV" ]; then
    python3 -m venv "$COMFY_VENV"
fi

source "$COMFY_VENV/bin/activate"
pip install --upgrade pip wheel -q

if [ "$IS_W7900" = true ]; then
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 -q
else
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

# ComfyUI requirements — skip torch lines to avoid overwrite
cd "$COMFY_DIR"
grep -v "^torch\|^torchvision\|^torchaudio" requirements.txt > /tmp/comfy_req.txt
pip install -q -r /tmp/comfy_req.txt
cd "$SCRIPT_DIR"
echo "[OK] ComfyUI ready at $COMFY_DIR"

# ── 5. Models (W7900 only — too slow on CPU) ──────────────────────────────────
if [ "$IS_W7900" = true ]; then
    echo ""
    echo "── Step 5: AI Models ──"
    MODELS="$COMFY_DIR/models"
    mkdir -p "$MODELS/checkpoints" "$MODELS/controlnet"

    SDXL="$MODELS/checkpoints/sd_xl_turbo_1.0_fp16.safetensors"
    if [ ! -f "$SDXL" ]; then
        echo "[..] Downloading SDXL-Turbo (~3.1 GB)..."
        wget -q --show-progress \
            "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors" \
            -O "$SDXL"
    else
        echo "[OK] SDXL-Turbo already present"
    fi

    CNET="$MODELS/controlnet/control-lora-canny-rank128.safetensors"
    if [ ! -f "$CNET" ]; then
        echo "[..] Downloading ControlNet Canny (~660 MB)..."
        wget -q --show-progress \
            "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors" \
            -O "$CNET"
    else
        echo "[OK] ControlNet already present"
    fi
else
    echo ""
    echo "── Step 5: AI Models (skipped — CPU mode) ──"
    echo "[INFO] AI generation models not downloaded (need W7900 for usable speed)"
    echo "[INFO] Live character transforms will still work fully"
fi

# ── 6. Copy start_comfyui.sh to home ──────────────────────────────────────────
echo ""
echo "── Step 6: Launch scripts ──"
cp "$SCRIPT_DIR/start_comfyui.sh" "$HOME/start_comfyui.sh"
chmod +x "$HOME/start_comfyui.sh"
echo "[OK] ~/start_comfyui.sh created"

deactivate

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Setup Complete!"
echo "======================================================"
echo ""
if [ "$IS_W7900" = true ]; then
    echo "  Full AI generation enabled (W7900 + ROCm)"
    echo ""
    echo "  Terminal 1:  ~/start_comfyui.sh"
    echo "  Terminal 2:  cd $SCRIPT_DIR && source venv/bin/activate && python start.py"
else
    echo "  Live transforms only (no AI generation on this machine)"
    echo "  This is fine for testing — use W7900 machine for the demo"
    echo ""
    echo "  source venv/bin/activate && python start.py"
fi
echo ""
