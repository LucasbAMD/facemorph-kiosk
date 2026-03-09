#!/bin/bash
# start_comfyui.sh — Launch ComfyUI for AMD Adapt Kiosk
# Run this in Terminal 1 before python start.py

COMFY_DIR="${COMFYUI_PATH:-$HOME/ComfyUI}"
COMFY_VENV="$HOME/comfyui-venv"

if [ ! -d "$COMFY_DIR" ]; then
    echo "[ERR] ComfyUI not found at $COMFY_DIR"
    echo "      Run bash setup_ubuntu.sh first."
    exit 1
fi

if [ ! -d "$COMFY_VENV" ]; then
    echo "[ERR] ComfyUI venv not found. Run bash setup_ubuntu.sh first."
    exit 1
fi

source "$COMFY_VENV/bin/activate"

# AMD W7900 / RDNA3 ROCm environment vars
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_TUNABLEOP_ENABLED=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export AMD_LOG_LEVEL=0

echo ""
echo "======================================================"
echo "  ComfyUI — AMD Adapt Kiosk AI Engine"
echo "  http://127.0.0.1:8188"
echo "======================================================"
echo ""

mkdir -p "$HOME/ComfyUI/output"
cd "$COMFY_DIR"

python main.py \
    --port 8188 \
    --listen 127.0.0.1 \
    --disable-auto-launch \
    --force-fp16 \
    --output-directory "$HOME/ComfyUI/output"
