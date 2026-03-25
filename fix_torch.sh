#!/usr/bin/env bash
# Fix PyTorch for Strix Halo (gfx1151) with ROCm 7.2
# Run with: bash fix_torch.sh

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "======================================================="
echo "  Fixing PyTorch for Strix Halo (gfx1151) + ROCm 7.2"
echo "======================================================="
echo ""

# Kill any hanging python processes
echo "[1/5] Killing any hanging Python processes..."
killall python 2>/dev/null || true
killall python3 2>/dev/null || true
sleep 2

# Activate venv
echo "[2/5] Activating venv..."
source kiosk_venv/bin/activate || { echo "ERROR: Could not activate venv"; exit 1; }

# Uninstall broken packages
echo "[3/5] Uninstalling broken PyTorch..."
pip uninstall -y torch torchvision torchaudio triton 2>/dev/null
pip cache purge 2>/dev/null

# Download official ROCm 7.2 wheels
echo "[4/5] Downloading official AMD ROCm 7.2 wheels..."
echo "      This will download ~4 GB..."
mkdir -p /tmp/rocm_wheels
cd /tmp/rocm_wheels

wget -q --show-progress -N https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
wget -q --show-progress -N https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl
wget -q --show-progress -N https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl

echo ""
echo "[5/5] Installing wheels..."
pip install /tmp/rocm_wheels/torch-*.whl /tmp/rocm_wheels/torchvision-*.whl /tmp/rocm_wheels/torchaudio-*.whl

cd "$SCRIPT_DIR"

echo ""
echo "Testing GPU compute..."
echo "(Each test has a 30s timeout)"
echo ""

# Test without override first
HSA_OVERRIDE_GFX_VERSION="" python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
x = torch.zeros(512,512).cuda()
print(f'Sum: {x.sum().item()}')
print('SUCCESS - GPU works!')
" && echo "No HSA override needed!" && exit 0

# Test with 11.0.0 override
echo "Trying with HSA_OVERRIDE_GFX_VERSION=11.0.0..."
HSA_OVERRIDE_GFX_VERSION=11.0.0 timeout 30 python -c "
import torch
x = torch.zeros(512,512).cuda()
print(f'Sum: {x.sum().item()}')
print('SUCCESS with 11.0.0!')
" && exit 0

echo ""
echo "GPU compute test failed. You may need a different ROCm version."
echo "======================================================="
