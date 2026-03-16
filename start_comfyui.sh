#!/bin/bash
# start_comfyui.sh — ComfyUI for AMD Adapt Kiosk
COMFY_DIR="${COMFYUI_PATH:-$HOME/ComfyUI}"
COMFY_VENV="$HOME/comfyui-venv"
if [ ! -d "$COMFY_DIR" ] || [ ! -d "$COMFY_VENV" ]; then
    echo "[ERR] ComfyUI not installed. Run: bash setup_ubuntu.sh"
    exit 1
fi
# GPU device permissions every launch
sudo chmod 666 /dev/kfd 2>/dev/null || true
sudo chmod 666 /dev/dri/renderD128 2>/dev/null || true
source "$COMFY_VENV/bin/activate"
# Fix any missing packages
python3 -c "import torchvision" 2>/dev/null || {
    echo "[..] Installing torchvision..."
    pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 -q
}
pip install torchsde kornia spandrel requests -q 2>/dev/null || true
# AMD W7900 ROCm env vars
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
python3 -c "
import torch
ok = torch.cuda.is_available()
print(f'[{\"OK\" if ok else \"WARN\"}] GPU: {torch.cuda.get_device_name(0) if ok else \"not visible\"}')
"
mkdir -p "$HOME/ComfyUI/output"
cd "$COMFY_DIR"
python3 main.py \
    --port 8188 \
    --listen 127.0.0.1 \
    --disable-auto-launch \
    --force-fp16 \
    --output-directory "$HOME/ComfyUI/output"
