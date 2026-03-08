#!/bin/bash
# setup_ubuntu.sh — AMD Adapt Kiosk full setup for Ubuntu 22.04/24.04 + W7900
# Run as your normal user (not root), has sudo calls where needed
# Takes ~20-40 min on first run (downloading models)

set -e
echo ""
echo "======================================================"
echo "  AMD ADAPT KIOSK — Ubuntu + ROCm Setup"
echo "======================================================"
echo ""

# ── 1. ROCm ───────────────────────────────────────────────────────────────────
echo "── Step 1: ROCm (skipping if already installed) ──"
if command -v rocm-smi &>/dev/null; then
    echo "[OK] ROCm already installed"
    rocm-smi --showproductname 2>/dev/null | head -5 || true
else
    echo "[..] Installing ROCm..."
    sudo apt update
    sudo apt install -y wget gnupg2

    # Add AMD ROCm repo
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | \
        sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

    # Ubuntu 22.04
    if grep -q "22.04" /etc/os-release; then
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/6.3 jammy main" | \
            sudo tee /etc/apt/sources.list.d/rocm.list
    else
        # Ubuntu 24.04
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/6.3 noble main" | \
            sudo tee /etc/apt/sources.list.d/rocm.list
    fi

    sudo apt update
    sudo apt install -y rocm-hip-sdk rocm-opencl-sdk

    # Add user to render/video groups
    sudo usermod -a -G render,video $USER
    echo "[OK] ROCm installed. Groups updated."
    echo ""
    echo "⚠️  IMPORTANT: Log out and back in for GPU group membership to take effect."
    echo "    Then re-run this script."
    exit 0
fi

# ── 2. Python deps for kiosk ──────────────────────────────────────────────────
echo ""
echo "── Step 2: Kiosk Python dependencies ──"
cd "$(dirname "$0")"  # make sure we're in the kiosk dir

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip wheel

# ROCm PyTorch
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

# Kiosk packages
pip install \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    "opencv-python>=4.9.0.80" \
    "numpy>=2.0.0" \
    "Pillow>=11.0.0" \
    "insightface==0.7.3" \
    "mediapipe>=0.10.30,<0.11" \
    "rembg[gpu]" \
    "python-multipart==0.0.9" \
    "aiofiles==23.2.1" \
    "requests==2.31.0"

echo "[OK] Kiosk packages installed"

# ── 3. ComfyUI ────────────────────────────────────────────────────────────────
echo ""
echo "── Step 3: ComfyUI ──"
COMFY_DIR="$HOME/ComfyUI"

if [ ! -d "$COMFY_DIR" ]; then
    echo "[..] Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi

# Create ComfyUI venv
COMFY_VENV="$HOME/comfyui-venv"
if [ ! -d "$COMFY_VENV" ]; then
    python3 -m venv "$COMFY_VENV"
fi
source "$COMFY_VENV/bin/activate"

pip install --upgrade pip wheel

# ROCm PyTorch for ComfyUI
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

# ComfyUI requirements (skip torch/torchvision/torchaudio — already installed)
cd "$COMFY_DIR"
sed 's/^torch/# torch/;s/^torchvision/# torchvision/;s/^torchaudio/# torchaudio/' \
    requirements.txt > requirements_notorch.txt
pip install -r requirements_notorch.txt
cd -

echo "[OK] ComfyUI installed at $COMFY_DIR"

# ── 4. ComfyUI models ─────────────────────────────────────────────────────────
echo ""
echo "── Step 4: Download AI models ──"
MODELS="$COMFY_DIR/models"

# SDXL-Turbo (fast generation, ~4 steps)
SDXL_PATH="$MODELS/checkpoints/sd_xl_turbo_1.0_fp16.safetensors"
if [ ! -f "$SDXL_PATH" ]; then
    echo "[..] Downloading SDXL-Turbo (~3.1 GB)..."
    mkdir -p "$MODELS/checkpoints"
    wget -q --show-progress \
        "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors" \
        -O "$SDXL_PATH"
    echo "[OK] SDXL-Turbo downloaded"
else
    echo "[OK] SDXL-Turbo already present"
fi

# ControlNet for SDXL (keeps body pose/structure)
CNET_PATH="$MODELS/controlnet/control-lora-canny-rank128.safetensors"
if [ ! -f "$CNET_PATH" ]; then
    echo "[..] Downloading ControlNet Canny (~660 MB)..."
    mkdir -p "$MODELS/controlnet"
    wget -q --show-progress \
        "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors" \
        -O "$CNET_PATH"
    echo "[OK] ControlNet downloaded"
else
    echo "[OK] ControlNet already present"
fi

# ComfyUI-Manager (optional but useful)
MANAGER_DIR="$COMFY_DIR/custom_nodes/ComfyUI-Manager"
if [ ! -d "$MANAGER_DIR" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "$MANAGER_DIR"
fi

# ── 5. Create launch script ───────────────────────────────────────────────────
echo ""
echo "── Step 5: Creating launch scripts ──"

cat > "$HOME/start_comfyui.sh" << 'EOF'
#!/bin/bash
source ~/comfyui-venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_TUNABLEOP_ENABLED=1
export DISABLE_ADDMM_CUDA_LT=1
export COMFYUI_PATH=~/ComfyUI
cd ~/ComfyUI
python main.py \
    --port 8188 \
    --listen 127.0.0.1 \
    --disable-auto-launch \
    --output-directory ~/ComfyUI/output
EOF
chmod +x "$HOME/start_comfyui.sh"
echo "[OK] ~/start_comfyui.sh created"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  ✅  Setup Complete!"
echo "======================================================"
echo ""
echo "TO RUN THE KIOSK:"
echo ""
echo "  Terminal 1 — Start ComfyUI (GPU inference):"
echo "    ~/start_comfyui.sh"
echo ""
echo "  Terminal 2 — Start the kiosk:"
echo "    cd $(pwd)"
echo "    source venv/bin/activate"
echo "    python start.py"
echo ""
echo "  Or use launch_comfy.py to start both:"
echo "    python launch_comfy.py && python start.py"
echo ""
echo "MODELS INSTALLED:"
echo "  ✓ SDXL-Turbo — ~2s per AI transform on W7900"
echo "  ✓ ControlNet Canny — preserves body pose"
echo ""
