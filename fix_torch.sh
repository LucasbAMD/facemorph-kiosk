#!/usr/bin/env bash
# Fix PyTorch for Strix Halo (gfx1151) with ROCm 7.2
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "======================================================="
echo "  Fixing PyTorch for Strix Halo (gfx1151) + ROCm 7.2"
echo "======================================================="
echo ""

killall python python3 2>/dev/null || true
sleep 1

echo "[1/5] Activating venv..."
source kiosk_venv/bin/activate || { echo "ERROR: Could not activate venv"; exit 1; }

echo "[2/5] Uninstalling broken PyTorch..."
pip uninstall -y torch torchvision torchaudio triton 2>/dev/null

echo "[3/5] Downloading official AMD ROCm 7.2 wheels..."
mkdir -p /tmp/rocm_wheels
cd /tmp/rocm_wheels

wget -q --show-progress -N https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
wget -q --show-progress -N https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
wget -q --show-progress -N https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl
wget -q --show-progress -N https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl

echo ""
echo "[4/5] Installing triton first, then torch..."
pip install /tmp/rocm_wheels/triton-*.whl
pip install --no-deps /tmp/rocm_wheels/torch-*.whl /tmp/rocm_wheels/torchvision-*.whl /tmp/rocm_wheels/torchaudio-*.whl

cd "$SCRIPT_DIR"

echo ""
echo "[5/5] Testing GPU compute..."
HSA_OVERRIDE_GFX_VERSION=11.0.0 timeout 30 python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.zeros(512,512).cuda()
    print(f'Sum: {x.sum().item()}')
    print('SUCCESS - GPU works!')
else:
    print('CUDA not available')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================="
    echo "  PyTorch fixed! Run: python start.py"
    echo "======================================================="
else
    echo ""
    echo "GPU compute timed out or failed."
    echo "======================================================="
fi
