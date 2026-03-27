#!/usr/bin/env bash
echo "Fixing GPU permissions..."

sudo chmod 666 /dev/kfd /dev/dri/renderD*
sudo usermod -a -G render,video $(whoami)

echo ""
echo "Testing GPU access..."
source kiosk_venv/bin/activate
python check_gpu.py

echo ""
echo "If it says 'GPU NOT available', log out and back in"
echo "(or reboot) for group changes to take effect, then"
echo "run: python check_gpu.py"
