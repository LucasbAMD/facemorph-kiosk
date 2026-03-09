#!/bin/bash
# fix_comfy_rocm.sh — Force reinstall ROCm PyTorch in ComfyUI venv
# Run this if ComfyUI complains about NVIDIA/CUDA

echo "[..] Reinstalling ROCm PyTorch in ComfyUI venv..."
source "$HOME/comfyui-venv/bin/activate"

# Uninstall any CPU or CUDA pytorch
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install ROCm 6.2 pytorch
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

echo ""
echo "[..] Verifying GPU access..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('[OK] GPU is working!')
else:
    print('[ERR] GPU not visible — check ROCm install with: rocm-smi')
"
deactivate
