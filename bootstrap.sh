#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  bootstrap.sh — One-command setup for AI Scene Style Kiosk
#
#  Usage:
#    git clone <repo-url> && cd facemorph-kiosk && bash bootstrap.sh
#
#  What it does:
#    1. Detects Linux distro and installs system packages
#    2. Checks for ROCm drivers — installs them if missing (Ubuntu)
#    3. Detects AMD GPU architecture automatically
#    4. Creates a Python venv (kiosk_venv)
#    5. Installs PyTorch with the correct ROCm version
#    6. Installs all Python dependencies
#    7. Downloads all AI models (~15 GB total)
#    8. Sets GPU device permissions
#
#  Requirements:
#    - Linux (Ubuntu 22.04/24.04 for auto ROCm install, others manual)
#    - AMD GPU/APU (RDNA 1-4, CDNA 1-4, or APU with ROCm support)
#    - ~20 GB free disk space for models
#    - Internet connection for downloads
# ──────────────────────────────────────────────────────────────────────
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/kiosk_venv"

# ── Error handler ────────────────────────────────────────────────────
fail() {
    echo ""
    echo "  [ERROR] $1"
    echo "  Bootstrap failed. Fix the issue above and re-run: bash bootstrap.sh"
    echo ""
    exit 1
}

echo ""
echo "======================================================="
echo "  AI Scene Style Kiosk — Bootstrap Setup"
echo "======================================================="
echo ""

# ── Helper: detect Linux distro ──────────────────────────────────────
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif command -v lsb_release &>/dev/null; then
        lsb_release -si | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}

DISTRO=$(detect_distro)
echo "[INFO] Detected Linux distro: $DISTRO"

# ── Helper: detect AMD GPU GFX version ───────────────────────────────
detect_gfx() {
    local kfd_nodes="/sys/class/kfd/kfd/topology/nodes"
    if [ ! -d "$kfd_nodes" ]; then
        echo ""
        return
    fi
    for node in "$kfd_nodes"/*/properties; do
        [ -f "$node" ] || continue
        local ver
        ver=$(grep "^gfx_target_version" "$node" 2>/dev/null | awk '{print $2}')
        if [ -n "$ver" ] && [ "$ver" != "0" ]; then
            # Decode MMMNNRR numeric format
            local major=$((ver / 10000))
            local minor=$(( (ver % 10000) / 100 ))
            local patch=$((ver % 100))
            # gfx9 series uses hex for some variants
            if [ "$major" -eq 9 ] && [ "$minor" -eq 0 ] && [ "$patch" -eq 10 ]; then
                echo "gfx90a"
            elif [ "$major" -eq 9 ] && [ "$minor" -eq 0 ] && [ "$patch" -ge 10 ]; then
                printf "gfx%d0%x\n" "$major" "$patch"
            else
                printf "gfx%d%d%02d\n" "$major" "$minor" "$patch"
            fi
            return
        fi
    done
    echo ""
}

# ── Helper: detect installed ROCm version ────────────────────────────
detect_rocm_version() {
    # Try the .info file first (most reliable)
    if [ -f /opt/rocm/.info/version ]; then
        cat /opt/rocm/.info/version | head -1 | cut -d'-' -f1
        return
    fi
    # Try rocm-smi
    if command -v rocm-smi &>/dev/null; then
        rocm-smi --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1
        return
    fi
    # Try dpkg
    if command -v dpkg &>/dev/null; then
        dpkg -l 2>/dev/null | grep "rocm-core" | awk '{print $3}' | cut -d'-' -f1 | head -1
        return
    fi
    # Try rpm
    if command -v rpm &>/dev/null; then
        rpm -q rocm-core 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1
        return
    fi
    echo ""
}

# ── Helper: detect Ubuntu codename ────────────────────────────────────
# On Linux Mint, VERSION_CODENAME is the Mint codename (e.g. "xia");
# UBUNTU_CODENAME holds the actual Ubuntu base (e.g. "noble"). Prefer that.
detect_ubuntu_codename() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
    else
        echo ""
    fi
}

# ── Helper: install ROCm on Ubuntu ───────────────────────────────────
install_rocm_ubuntu() {
    local codename
    codename=$(detect_ubuntu_codename)

    # Map codename to ROCm package URL
    local rocm_url=""
    case "$codename" in
        jammy)
            rocm_url="https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb"
            ;;
        noble)
            rocm_url="https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb"
            ;;
        *)
            echo "  [ERR] Unsupported Ubuntu version: $codename"
            echo "         ROCm auto-install supports Ubuntu 22.04 (jammy) and 24.04 (noble)"
            echo "         Install ROCm manually: https://rocm.docs.amd.com/en/latest/deploy/linux/install.html"
            return 1
            ;;
    esac

    echo ""
    echo "  Installing ROCm 7.2 for Ubuntu $codename..."
    echo "  This will install the AMD GPU kernel driver and ROCm runtime."
    echo "  (A reboot will be required after installation)"
    echo ""

    # Download and install the amdgpu-install package
    local tmp_deb="/tmp/amdgpu-install.deb"
    echo "  [..] Downloading ROCm installer..."
    wget -q -O "$tmp_deb" "$rocm_url" || fail "Failed to download ROCm installer from $rocm_url"
    sudo apt-get install -y "$tmp_deb" || fail "Failed to install amdgpu-install package"
    rm -f "$tmp_deb"

    # Install AMDGPU driver + ROCm
    echo "  [..] Installing AMD GPU driver and ROCm (this takes several minutes)..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get update || fail "apt-get update failed during ROCm install"
    sudo DEBIAN_FRONTEND=noninteractive amdgpu-install -y --usecase=rocm --no-32 --no-dkms || fail "amdgpu-install failed"

    # Add user to required groups
    sudo usermod -a -G video,render ${SUDO_USER:-$USER} 2>/dev/null || true

    echo "  [OK] ROCm installed successfully"
    return 0
}

# ── Helper: map ROCm version to PyTorch wheel URL ────────────────────
get_pytorch_rocm_url() {
    local rocm_ver="$1"
    local gfx_ver="$2"
    local major minor
    major=$(echo "$rocm_ver" | cut -d'.' -f1)
    minor=$(echo "$rocm_ver" | cut -d'.' -f2)

    # gfx1150/gfx1151 (Strix Point/Halo) need TheRock wheels with native support.
    # Standard PyTorch nightly does NOT support these architectures and causes hangs.
    if [ "$gfx_ver" = "gfx1150" ] || [ "$gfx_ver" = "gfx1151" ]; then
        echo "https://rocm.nightlies.amd.com/v2/$gfx_ver/"
        return
    fi

    if [ "$major" -ge 7 ]; then
        # ROCm 7.x — use nightly with rocm7.1 wheels
        echo "https://download.pytorch.org/whl/nightly/rocm7.1"
    elif [ "$major" -eq 6 ] && [ "$minor" -ge 2 ]; then
        echo "https://download.pytorch.org/whl/rocm6.2"
    elif [ "$major" -eq 6 ]; then
        echo "https://download.pytorch.org/whl/rocm6.1"
    else
        # ROCm 5.x or older
        echo "https://download.pytorch.org/whl/rocm5.7"
    fi
}

# ── 1. System packages ──────────────────────────────────────────────
echo "[1/7] Installing system packages..."

# Check for apt lock (e.g. unattended-upgrades running)
if command -v fuser &>/dev/null; then
    if fuser /var/lib/apt/lists/lock &>/dev/null || fuser /var/lib/dpkg/lock-frontend &>/dev/null; then
        echo "  [WAIT] Another package manager is running. Waiting up to 120s..."
        for i in $(seq 1 24); do
            sleep 5
            if ! fuser /var/lib/apt/lists/lock &>/dev/null && ! fuser /var/lib/dpkg/lock-frontend &>/dev/null; then
                echo "  [OK] Package manager lock released"
                break
            fi
            if [ "$i" -eq 24 ]; then
                fail "Package manager is still locked after 120s. Try: sudo kill \$(fuser /var/lib/apt/lists/lock 2>/dev/null) then re-run bootstrap.sh"
            fi
        done
    fi
fi

case "$DISTRO" in
    ubuntu|debian|linuxmint|pop)
        sudo apt-get update -qq || fail "apt-get update failed. Check your internet connection."

        # libgl1-mesa-glx was removed in Ubuntu 24.04+. libgl1 exists on
        # both 22.04 and 24.04 and provides libGL.so.1 (all OpenCV needs).
        # Use apt-cache policy to check for an actual install candidate,
        # not just package metadata (transitional dummies confuse apt-cache show).
        if apt-cache policy libgl1-mesa-glx 2>/dev/null | grep -qE "Candidate: [0-9]"; then
            GL_PKG="libgl1-mesa-glx"
        else
            GL_PKG="libgl1"
        fi

        sudo apt-get install -y \
            python3 python3-venv python3-dev python3-pip \
            build-essential cmake \
            "$GL_PKG" libglib2.0-0 libsm6 libxrender1 libxext6 \
            libopencv-dev \
            wget curl git \
            || fail "apt-get install failed. See errors above."
        ;;
    fedora|rhel|centos|rocky|almalinux)
        sudo dnf install -y \
            python3 python3-devel python3-pip \
            gcc gcc-c++ cmake make \
            mesa-libGL glib2 libSM libXrender libXext \
            opencv-devel \
            wget curl git \
            || fail "dnf install failed. See errors above."
        ;;
    arch|manjaro|endeavouros)
        sudo pacman -Syu --noconfirm --needed \
            python python-pip \
            base-devel cmake \
            mesa glib2 libsm libxrender libxext \
            opencv \
            wget curl git \
            || fail "pacman install failed. See errors above."
        ;;
    opensuse*|sles)
        sudo zypper install -y \
            python3 python3-devel python3-pip \
            gcc gcc-c++ cmake make \
            Mesa-libGL1 glib2-devel libSM6 libXrender1 libXext6 \
            opencv-devel \
            wget curl git \
            || fail "zypper install failed. See errors above."
        ;;
    *)
        echo "  [WARN] Unknown distro: $DISTRO"
        echo "         Please install manually: python3, python3-venv, cmake,"
        echo "         build-essential, OpenCV deps, wget, curl, git"
        echo "         Press Enter to continue or Ctrl+C to abort..."
        read -r
        ;;
esac
echo "  [OK] System packages installed"

# ── 2. Check / Install ROCm drivers ──────────────────────────────────
echo ""
echo "[2/7] Checking ROCm drivers..."
ROCM_VER=$(detect_rocm_version)
NEEDS_REBOOT=false
if [ -z "$ROCM_VER" ]; then
    echo "  [WARN] ROCm drivers not found"
    case "$DISTRO" in
        ubuntu|linuxmint|pop)
            echo "  [..] Attempting automatic ROCm installation..."
            if install_rocm_ubuntu; then
                ROCM_VER=$(detect_rocm_version)
                if [ -z "$ROCM_VER" ]; then
                    ROCM_VER="7.2.0"  # Just installed it
                fi
                NEEDS_REBOOT=true
                echo "  [OK] ROCm $ROCM_VER installed"
            else
                echo "  [ERR] ROCm auto-install failed"
                echo "         Install manually: https://rocm.docs.amd.com/en/latest/deploy/linux/install.html"
                exit 1
            fi
            ;;
        *)
            echo "  [ERR] ROCm auto-install is only supported on Ubuntu"
            echo ""
            echo "  Install ROCm manually for your distro:"
            echo "    https://rocm.docs.amd.com/en/latest/deploy/linux/install.html"
            echo ""
            echo "  After installing ROCm, reboot and re-run this script."
            exit 1
            ;;
    esac
else
    echo "  [OK] ROCm $ROCM_VER detected"
fi

# ── 3. Detect GPU ────────────────────────────────────────────────────
echo ""
echo "[3/7] Detecting AMD GPU..."
GFX_VER=$(detect_gfx)
if [ -n "$GFX_VER" ]; then
    echo "  [OK] GPU architecture: $GFX_VER"
else
    echo "  [WARN] Could not detect GPU architecture from sysfs"
    echo "         /dev/kfd may need permissions — continuing anyway"
fi

# Determine PyTorch wheel URL based on ROCm version and GPU architecture
ROCM_TORCH_URL=$(get_pytorch_rocm_url "$ROCM_VER" "$GFX_VER")
echo "  [OK] PyTorch ROCm URL: $ROCM_TORCH_URL"

# ── 4. Python venv ───────────────────────────────────────────────────
echo ""
echo "[4/7] Setting up Python virtual environment..."

# Check for python3-venv (common missing package on Debian/Ubuntu)
if ! python3 -m venv --help &>/dev/null; then
    echo "  [INFO] Installing python3-venv..."
    case "$DISTRO" in
        ubuntu|debian|linuxmint|pop)
            sudo apt-get install -y python3-venv || fail "Failed to install python3-venv"
            ;;
    esac
fi

if [ -d "$VENV_DIR" ]; then
    echo "  [OK] venv already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR" || fail "Failed to create Python virtual environment"
    echo "  [OK] Created venv at $VENV_DIR"
fi

# Activate venv for the rest of the script
source "$VENV_DIR/bin/activate" || fail "Failed to activate virtual environment at $VENV_DIR"
pip install --upgrade pip setuptools wheel -q || fail "Failed to upgrade pip/setuptools. Check your internet connection."

# ── 5. PyTorch with ROCm ─────────────────────────────────────────────
echo ""
echo "[5/7] Installing PyTorch with ROCm $ROCM_VER support..."
echo "       (This may take a few minutes)"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  [OK] PyTorch with GPU support already installed"
    python -c "import torch; print(f'  PyTorch {torch.__version__} — GPU: {torch.cuda.get_device_name(0)}')"
else
    pip install --pre torch torchvision torchaudio --index-url "$ROCM_TORCH_URL" -q || fail "Failed to install PyTorch. Check your internet connection."
    echo "  [OK] PyTorch installed"
    python -c "import torch; print(f'  PyTorch {torch.__version__}')"
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
    else
        echo "  [WARN] GPU not detected by PyTorch — check ROCm drivers"
        echo "         You may need to reboot after ROCm installation"
    fi
fi

# ── 6. Python dependencies ───────────────────────────────────────────
echo ""
echo "[6/7] Installing Python dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" -q || fail "Failed to install Python dependencies"
# huggingface-hub is needed by setup_models.py but not in requirements.txt
pip install huggingface-hub -q || fail "Failed to install huggingface-hub"
echo "  [OK] All Python packages installed"

# ── 7. Download AI models ────────────────────────────────────────────
echo ""
echo "[7/7] Downloading AI models (~15 GB, may take a while)..."
echo "       Models are cached in ~/.cache/huggingface/ and ~/kiosk_models/"
echo ""
python "$SCRIPT_DIR/setup_models.py" || fail "Model download failed. Check your internet connection and disk space."

# ── GPU permissions ──────────────────────────────────────────────────
echo ""
echo "[*] Setting GPU device permissions..."
if [ -e /dev/kfd ]; then
    sudo chmod 666 /dev/kfd 2>/dev/null && echo "  [OK] /dev/kfd" || echo "  [WARN] Could not set /dev/kfd permissions"
fi
for dev in /dev/dri/renderD*; do
    if [ -e "$dev" ]; then
        sudo chmod 666 "$dev" 2>/dev/null && echo "  [OK] $dev" || echo "  [WARN] Could not set $dev permissions"
    fi
done

# ── Strix Halo / Strix Point specific fixes ─────────────────────────
if [ "$GFX_VER" = "gfx1151" ] || [ "$GFX_VER" = "gfx1150" ]; then
    echo ""
    echo "[*] Strix Halo/Point APU detected — applying required fixes..."

    # ── Fix 1: Remove broken amdgpu-dkms-firmware (MES 0x83 causes hangs) ──
    if dpkg -l amdgpu-dkms-firmware 2>/dev/null | grep -q '^ii'; then
        echo "  [..] Removing broken amdgpu-dkms-firmware package..."
        sudo apt-get autoremove --purge -y amdgpu-dkms-firmware 2>/dev/null
        echo "  [OK] Removed amdgpu-dkms-firmware"
    fi

    # ── Fix 2: Download latest gfx1151 firmware from kernel.org ──
    # Always update firmware on Strix Halo to ensure we have non-broken MES
    FW_NEEDS_UPDATE=false
    if [ ! -f /usr/lib/firmware/amdgpu/gc_11_5_1_me.bin ]; then
        FW_NEEDS_UPDATE=true
    fi
    if [ "$FW_NEEDS_UPDATE" = true ]; then
        echo "  [..] Downloading latest gfx1151 firmware from kernel.org..."
        fw_dir="/tmp/linux-firmware-$$"
        if git clone --depth 1 --filter=blob:none --sparse \
               https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git \
               "$fw_dir" 2>/dev/null; then
            cd "$fw_dir" && git sparse-checkout set amdgpu 2>/dev/null && cd "$SCRIPT_DIR"
            sudo cp "$fw_dir"/amdgpu/gc_11_5_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
            sudo cp "$fw_dir"/amdgpu/mes_11_5_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
            sudo cp "$fw_dir"/amdgpu/sdma_6_1_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
            sudo cp "$fw_dir"/amdgpu/vcn_5_0_1_* /usr/lib/firmware/amdgpu/ 2>/dev/null
            rm -rf "$fw_dir"
            echo "  [OK] Firmware updated"
        else
            echo "  [WARN] Could not download firmware — run bash fix_firmware.sh after bootstrap"
        fi
    else
        echo "  [OK] gfx1151 firmware files present"
    fi

    # ── Fix 3: Add amdgpu.cwsr_enable=0 to GRUB ──
    if ! grep -q "amdgpu.cwsr_enable=0" /proc/cmdline 2>/dev/null; then
        echo "  [..] Adding amdgpu.cwsr_enable=0 to GRUB..."
        if [ -f /etc/default/grub ]; then
            # Only add if not already in the GRUB config file
            if ! grep -q "cwsr_enable=0" /etc/default/grub 2>/dev/null; then
                sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 amdgpu.cwsr_enable=0"/' /etc/default/grub
                sudo update-grub 2>/dev/null || sudo grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null
                echo "  [OK] GRUB updated — reboot required"
            fi
        fi
        NEEDS_REBOOT=true
    else
        echo "  [OK] amdgpu.cwsr_enable=0 is set"
    fi

    # ── Fix 4: Rebuild initramfs with correct firmware ──
    echo "  [..] Rebuilding initramfs..."
    sudo update-initramfs -u 2>/dev/null
    echo "  [OK] initramfs updated"

    # ── Check MES firmware version if debugfs is available ──
    if [ -r /sys/kernel/debug/dri/1/amdgpu_firmware_info ]; then
        MES_VER=$(grep "MES feature" /sys/kernel/debug/dri/1/amdgpu_firmware_info 2>/dev/null | head -1 | grep -oP '0x[0-9a-f]+' | tail -1)
        if [ "$MES_VER" = "0x00000083" ]; then
            echo "  [WARN] MES firmware 0x83 still active — reboot required to load new firmware"
            NEEDS_REBOOT=true
        elif [ -n "$MES_VER" ]; then
            echo "  [OK] MES firmware: $MES_VER"
        fi
    fi

    echo "  [OK] Strix Halo fixes applied"
fi

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Setup complete!"
if [ -n "$GFX_VER" ]; then
    echo "  GPU: $GFX_VER | ROCm: $ROCM_VER"
fi
if [ "$NEEDS_REBOOT" = true ]; then
    echo ""
    echo "  ** REBOOT REQUIRED **"
    echo "  ROCm was just installed. You must reboot before starting."
    echo ""
    echo "  After reboot:"
    echo "    cd $(pwd)"
    echo "    source kiosk_venv/bin/activate"
    echo "    python start.py"
else
    echo ""
    echo "  To start the kiosk:"
    echo ""
    echo "    source kiosk_venv/bin/activate"
    echo "    python start.py"
fi
echo ""
echo "  Then open http://localhost:8000 in a browser."
echo "======================================================="
echo ""
