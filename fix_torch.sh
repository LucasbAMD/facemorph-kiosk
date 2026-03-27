#!/usr/bin/env bash
# Fix PyTorch for Strix Halo (gfx1151)
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
timeout 30 python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.zeros(512,512).cuda()
    print(f'Sum: {x.sum().item()}')
    print('SUCCESS - GPU works!')
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
    echo "Make sure you added the GRUB fix and rebooted:"
    echo "  1. sudo nano /etc/default/grub"
    echo "  2. Add amdgpu.cwsr_enable=0 to GRUB_CMDLINE_LINUX_DEFAULT"
    echo "  3. sudo update-grub && sudo reboot"
    echo "  4. Then re-run: bash fix_torch.sh"
    echo "======================================================="
fi
