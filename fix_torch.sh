#!/usr/bin/env bash
# Fix PyTorch for Strix Halo (gfx1151)
#
# This script:
#   1. Removes any existing PyTorch installation
#   2. Installs AMD's TheRock wheel built specifically for gfx1151
#   3. Sets the correct environment variables for APU compute
#   4. Tests GPU compute with a simple tensor operation
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "======================================================="
echo "  Fixing PyTorch for Strix Halo (gfx1151)"
echo "======================================================="
echo ""

killall python python3 2>/dev/null || true
sleep 1

echo "[1/4] Activating venv..."
source kiosk_venv/bin/activate || { echo "ERROR: Could not activate venv"; exit 1; }

echo "[2/4] Uninstalling current PyTorch..."
pip uninstall -y torch torchvision torchaudio triton 2>/dev/null
pip cache purge 2>/dev/null

echo "[3/4] Installing TheRock PyTorch for gfx1151..."
echo "      (This is the AMD-specific build for Strix Halo)"
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio

cd "$SCRIPT_DIR"

echo ""
echo "[4/4] Testing GPU compute..."
echo "      (Setting HSA_ENABLE_SDMA=0 for APU stability)"

# Critical env vars for Strix Halo APU
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.9"
# Do NOT set HSA_OVERRIDE_GFX_VERSION — TheRock wheel has native gfx1151 support
unset HSA_OVERRIDE_GFX_VERSION 2>/dev/null

timeout 30 python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.zeros(512,512).cuda()
    print(f'Sum: {x.sum().item()}')
    print('SUCCESS - GPU compute works!')
else:
    print('ERROR: No HIP GPUs found')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================="
    echo "  PyTorch fixed! Run: python start.py"
    echo "======================================================="
else
    echo ""
    echo "GPU test failed or timed out."
    echo ""
    echo "Troubleshooting steps:"
    echo ""
    echo "  1. GRUB kernel parameter (required):"
    echo "     sudo nano /etc/default/grub"
    echo "     Add amdgpu.cwsr_enable=0 to GRUB_CMDLINE_LINUX_DEFAULT"
    echo "     sudo update-grub && sudo reboot"
    echo ""
    echo "  2. GPU permissions:"
    echo "     sudo chmod 666 /dev/kfd /dev/dri/renderD*"
    echo "     sudo usermod -a -G render,video \$(whoami)"
    echo "     (log out and back in for group changes)"
    echo ""
    echo "  3. Update firmware:"
    echo "     sudo apt install linux-firmware"
    echo ""
    echo "  4. After fixing, re-run: bash fix_torch.sh"
    echo "======================================================="
fi
