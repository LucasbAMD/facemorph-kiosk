#!/usr/bin/env bash
# Fix PyTorch for Strix Halo (gfx1151)
# Run with: bash fix_torch.sh

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "======================================================="
echo "  Fixing PyTorch for Strix Halo (gfx1151)"
echo "======================================================="
echo ""

# Kill any hanging python processes
echo "[1/4] Killing any hanging Python processes..."
killall python 2>/dev/null || true
killall python3 2>/dev/null || true
sleep 2

# Activate venv
echo "[2/4] Activating venv..."
source kiosk_venv/bin/activate || { echo "ERROR: Could not activate venv"; exit 1; }

# Uninstall broken packages
echo "[3/4] Uninstalling broken PyTorch and triton..."
pip uninstall -y torch torchvision torchaudio triton 2>/dev/null
pip cache purge 2>/dev/null

# Install correct PyTorch for gfx1151
echo "[4/4] Installing AMD TheRock PyTorch for gfx1151..."
echo "      This may take a few minutes..."
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch

echo ""
echo "Testing GPU compute..."
python check_compute.py

echo ""
echo "If it says WORKING, run: python start.py"
echo "======================================================="
