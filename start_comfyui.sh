#!/bin/bash
# start_comfyui.sh — Launch ComfyUI for AMD Adapt Kiosk (W7900 + ROCm)

COMFY_DIR="${COMFYUI_PATH:-$HOME/ComfyUI}"
COMFY_VENV="$HOME/comfyui-venv"

if [ ! -d "$COMFY_DIR" ]; then
    echo "[ERR] ComfyUI not found. Run bash setup_ubuntu.sh first."
    exit 1
fi

if [ ! -d "$COMFY_VENV" ]; then
    echo "[ERR] ComfyUI venv not found. Run bash setup_ubuntu.sh first."
    exit 1
fi

source "$COMFY_VENV/bin/activate"

# Verify PyTorch sees the GPU
echo "[..] Checking GPU..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('[WARN] GPU not visible to PyTorch — ROCm PyTorch may not be installed')
"

# W7900 = gfx1100 (RDNA3)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_TUNABLEOP_ENABLED=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export AMD_LOG_LEVEL=0
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

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
