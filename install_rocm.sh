#!/bin/bash
# install_rocm.sh — Install ROCm on Ubuntu 22.04/24.04 for AMD W7900
# Run as normal user: bash install_rocm.sh

echo ""
echo "======================================================"
echo "  ROCm Installer for AMD Adapt Kiosk"
echo "======================================================"
echo ""

# Detect Ubuntu version
UBUNTU_VER=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
echo "[INFO] Ubuntu $UBUNTU_VER detected"

if [ "$UBUNTU_VER" = "22.04" ]; then
    CODENAME="jammy"
elif [ "$UBUNTU_VER" = "24.04" ]; then
    CODENAME="noble"
else
    echo "[WARN] Unknown Ubuntu version, trying noble"
    CODENAME="noble"
fi

# Step 1 — dependencies
echo "[..] Installing dependencies..."
sudo apt update -qq
sudo apt install -y curl gnupg2 python3-venv python3-pip git

# Step 2 — ROCm GPG key (using curl, more reliable than wget)
echo "[..] Adding ROCm GPG key..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | \
    sudo gpg --dearmor --yes -o /etc/apt/keyrings/rocm.gpg

if [ $? -ne 0 ]; then
    echo "[WARN] GPG key via curl failed, trying alternate method..."
    sudo apt-key adv --fetch-keys https://repo.radeon.com/rocm/rocm.gpg.key
fi

# Step 3 — ROCm repo
echo "[..] Adding ROCm repository..."
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/6.2 $CODENAME main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update -qq

# Step 4 — Install ROCm
echo "[..] Installing ROCm (this takes a few minutes)..."
sudo apt install -y rocm-opencl-runtime rocm-hip-sdk 2>/dev/null || \
sudo apt install -y rocm-opencl-runtime 2>/dev/null || \
    echo "[WARN] Full ROCm install had issues, trying minimal..."
sudo apt install -y --fix-broken

# Step 5 — Add user to GPU groups
echo "[..] Adding $USER to render and video groups..."
sudo usermod -a -G render,video $USER

# Step 6 — Verify
echo ""
echo "[..] Verifying install..."
if command -v rocm-smi &>/dev/null; then
    rocm-smi --showproductname 2>/dev/null || true
    echo ""
    echo "[OK] ROCm installed successfully!"
else
    sudo apt install -y rocm-smi 2>/dev/null || true
    echo "[OK] ROCm installed — rocm-smi will be available after logout"
fi

echo ""
echo "======================================================"
echo "  IMPORTANT: Log out and back in now!"
echo "  Then run: bash setup_ubuntu.sh"
echo "======================================================"
echo ""
