#!/bin/bash
# download_sdxl_base.sh -- Download SDXL Base 1.0 for AMD Adapt Kiosk
# Run this once from any terminal (ComfyUI does NOT need to be running)

DEST="$HOME/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors"
URL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"

echo ""
echo "======================================================"
echo "  AMD Adapt Kiosk -- Download SDXL Base 1.0"
echo "======================================================"
echo ""

if [ -f "$DEST" ]; then
    SIZE=$(du -sh "$DEST" | cut -f1)
    echo "[OK] Already downloaded ($SIZE) -- nothing to do"
    echo "     $DEST"
    exit 0
fi

mkdir -p "$HOME/ComfyUI/models/checkpoints"
echo "[..] Downloading SDXL Base 1.0 (~6.9 GB) -- this takes 5-15 minutes..."
echo ""

pip install huggingface_hub --break-system-packages -q 2>/dev/null || true

python3 - <<'PYEOF'
import sys, os

dest = os.path.expanduser("~/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors")
url  = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"

try:
    from huggingface_hub import hf_hub_download
    import shutil
    print("[..] Using huggingface_hub...")
    path = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0.safetensors",
        local_dir=os.path.dirname(dest)
    )
    print(f"[OK] Downloaded to {path}")
    sys.exit(0)
except Exception as e:
    print(f"[..] huggingface_hub failed ({e}), falling back to urllib...")

import urllib.request

def progress(count, block, total):
    pct = min(count * block / total * 100, 100)
    mb  = count * block / 1024 / 1024
    tot = total / 1024 / 1024
    print(f"\r    {pct:.1f}%  {mb:.0f} / {tot:.0f} MB", end="", flush=True)

try:
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\n[OK] Download complete!")
except Exception as e:
    print(f"\n[ERROR] Download failed: {e}")
    sys.exit(1)
PYEOF

if [ -f "$DEST" ]; then
    SIZE=$(du -sh "$DEST" | cut -f1)
    echo ""
    echo "[OK] SDXL Base 1.0 ready ($SIZE)"
    echo "     $DEST"
    echo ""
    echo "  Now restart ComfyUI and the kiosk:"
    echo "  Terminal 1: ~/start_comfyui.sh"
    echo "  Terminal 2: bash ~/facemorph-kiosk/start.sh"
else
    echo "[ERROR] File not found after download -- something went wrong"
    exit 1
fi
