#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  bootstrap.sh — One-command setup for AI Scene Style Kiosk
#
#  Usage:
#    git clone <repo-url> && cd facemorph-kiosk && bash bootstrap.sh
#
#  What it does:
#    1. Detects Linux distro and installs system packages
#    2. Checks for ROCm drivers (required, not installed by this script)
#    3. Detects AMD GPU architecture automatically
#    4. Creates a Python venv (kiosk_venv)
#    5. Installs PyTorch with the correct ROCm version
#    6. Installs all Python dependencies
#    7. Downloads all AI models (~15 GB total)
#    8. Sets GPU device permissions
#
#  Requirements:
#    - Linux (Ubuntu/Debian, Fedora/RHEL, Arch, or openSUSE)
#    - AMD GPU/APU with ROCm drivers pre-installed
#    - ~20 GB free disk space for models
#    - Internet connection for downloads
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/kiosk_venv"

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

# ── Helper: map ROCm version to PyTorch wheel URL ────────────────────
get_pytorch_rocm_url() {
    local rocm_ver="$1"
    local major minor
    major=$(echo "$rocm_ver" | cut -d'.' -f1)
    minor=$(echo "$rocm_ver" | cut -d'.' -f2)

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
case "$DISTRO" in
    ubuntu|debian|linuxmint|pop)
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            python3 python3-venv python3-dev python3-pip \
            build-essential cmake \
            libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
            libopencv-dev \
            wget curl git \
            2>/dev/null
        ;;
    fedora|rhel|centos|rocky|almalinux)
        sudo dnf install -y -q \
            python3 python3-devel python3-pip \
            gcc gcc-c++ cmake make \
            mesa-libGL glib2 libSM libXrender libXext \
            opencv-devel \
            wget curl git \
            2>/dev/null
        ;;
    arch|manjaro|endeavouros)
        sudo pacman -Syu --noconfirm --needed \
            python python-pip \
            base-devel cmake \
            mesa glib2 libsm libxrender libxext \
            opencv \
            wget curl git \
            2>/dev/null
        ;;
    opensuse*|sles)
        sudo zypper install -y -n \
            python3 python3-devel python3-pip \
            gcc gcc-c++ cmake make \
            Mesa-libGL1 glib2-devel libSM6 libXrender1 libXext6 \
            opencv-devel \
            wget curl git \
            2>/dev/null
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

# ── 2. Check ROCm drivers ────────────────────────────────────────────
echo ""
echo "[2/7] Checking ROCm drivers..."
ROCM_VER=$(detect_rocm_version)
if [ -z "$ROCM_VER" ]; then
    echo "  [ERR] ROCm drivers not found!"
    echo ""
    echo "  ROCm must be installed before running this script."
    echo "  Install ROCm from: https://rocm.docs.amd.com/en/latest/deploy/linux/install.html"
    echo ""
    echo "  Or use the automated installer:"
    echo "  https://github.com/JoergR75/rocm-7.2.0-pytorch-docker-cdna-rdna-automated-deployment"
    echo ""
    echo "  After installing ROCm, reboot and re-run this script."
    exit 1
fi
echo "  [OK] ROCm $ROCM_VER detected"

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

# Determine PyTorch wheel URL based on ROCm version
ROCM_TORCH_URL=$(get_pytorch_rocm_url "$ROCM_VER")
echo "  [OK] PyTorch ROCm URL: $ROCM_TORCH_URL"

# ── 4. Python venv ───────────────────────────────────────────────────
echo ""
echo "[4/7] Setting up Python virtual environment..."

# Check for python3-venv (common missing package on Debian/Ubuntu)
if ! python3 -m venv --help &>/dev/null; then
    echo "  [INFO] Installing python3-venv..."
    case "$DISTRO" in
        ubuntu|debian|linuxmint|pop)
            sudo apt-get install -y -qq python3-venv 2>/dev/null
            ;;
    esac
fi

if [ -d "$VENV_DIR" ]; then
    echo "  [OK] venv already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "  [OK] Created venv at $VENV_DIR"
fi

# Activate venv for the rest of the script
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ── 5. PyTorch with ROCm ─────────────────────────────────────────────
echo ""
echo "[5/7] Installing PyTorch with ROCm $ROCM_VER support..."
echo "       (This may take a few minutes)"
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  [OK] PyTorch with GPU support already installed"
    python -c "import torch; print(f'  PyTorch {torch.__version__} — GPU: {torch.cuda.get_device_name(0)}')"
else
    pip install torch torchvision torchaudio --index-url "$ROCM_TORCH_URL" -q
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
pip install -r "$SCRIPT_DIR/requirements.txt" -q
# huggingface-hub is needed by setup_models.py but not in requirements.txt
pip install huggingface-hub -q
echo "  [OK] All Python packages installed"

# ── 7. Download AI models ────────────────────────────────────────────
echo ""
echo "[7/7] Downloading AI models (~15 GB, may take a while)..."
echo "       Models are cached in ~/.cache/huggingface/ and ~/kiosk_models/"
echo ""
python "$SCRIPT_DIR/setup_models.py"

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

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Setup complete!"
if [ -n "$GFX_VER" ]; then
echo "  GPU: $GFX_VER | ROCm: $ROCM_VER"
fi
echo ""
echo "  To start the kiosk:"
echo ""
echo "    source kiosk_venv/bin/activate"
echo "    python start.py"
echo ""
echo "  Then open http://localhost:8000 in a browser."
echo "======================================================="
echo ""
