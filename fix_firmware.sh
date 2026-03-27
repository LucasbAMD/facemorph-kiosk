#!/usr/bin/env bash
# Fix broken MES firmware (0x83) on Strix Halo (gfx1151)
#
# The amdgpu-dkms-firmware package from ROCm ships MES firmware 0x83
# which causes GPU compute hangs. This script downloads correct firmware
# from the upstream linux-firmware repo and rebuilds initramfs.
set -o pipefail

echo ""
echo "======================================================="
echo "  Fixing MES firmware for Strix Halo (gfx1151)"
echo "======================================================="
echo ""

# Remove broken firmware package if still installed
if dpkg -l amdgpu-dkms-firmware 2>/dev/null | grep -q '^ii'; then
    echo "[1/4] Removing amdgpu-dkms-firmware..."
    sudo apt autoremove --purge -y amdgpu-dkms-firmware
else
    echo "[1/4] amdgpu-dkms-firmware already removed (good)"
fi

echo ""
echo "[2/4] Downloading latest firmware from kernel.org..."
cd /tmp
rm -rf /tmp/linux-firmware 2>/dev/null
git clone --depth 1 https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git || {
    echo "ERROR: Failed to clone firmware repo. Check internet connection."
    exit 1
}

echo ""
echo "[3/4] Installing gfx1151 firmware files..."
sudo cp /tmp/linux-firmware/amdgpu/gc_11_5_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
sudo cp /tmp/linux-firmware/amdgpu/mes_11_5_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
sudo cp /tmp/linux-firmware/amdgpu/sdma_6_1_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
sudo cp /tmp/linux-firmware/amdgpu/vcn_5_0_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
sudo cp /tmp/linux-firmware/amdgpu/jpeg_5_0_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
echo "  [OK] Firmware files copied"

echo ""
echo "[4/4] Rebuilding initramfs..."
sudo update-initramfs -u || {
    echo "ERROR: Failed to update initramfs"
    exit 1
}
echo "  [OK] initramfs updated"

# Cleanup
rm -rf /tmp/linux-firmware 2>/dev/null

echo ""
echo "======================================================="
echo "  Firmware updated! Now reboot:"
echo ""
echo "    sudo reboot"
echo ""
echo "  After reboot, test with:"
echo "    sudo bash check_system.sh"
echo "======================================================="
echo ""
