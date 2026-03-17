#!/usr/bin/env python3
"""
setup_models.py — Download IP-Adapter model weights for AMD Adapt Kiosk.
Run this once before starting the kiosk for the first time.

Usage:
    source ~/facemorph-kiosk/venv/bin/activate
    python setup_models.py
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR    = Path.home() / "kiosk_models"
IP_ADAPTER_DIR = MODELS_DIR / "ip_adapter" / "sdxl_models"

# IP-Adapter SDXL model — preserves facial identity in generated images
# Source: https://huggingface.co/h94/IP-Adapter
IP_ADAPTER_FILES = {
    "ip-adapter_sdxl.bin": (
        "https://huggingface.co/h94/IP-Adapter/resolve/main/"
        "sdxl_models/ip-adapter_sdxl.bin"
    ),
    "image_encoder/config.json": (
        "https://huggingface.co/h94/IP-Adapter/resolve/main/"
        "models/image_encoder/config.json"
    ),
    "image_encoder/pytorch_model.bin": (
        "https://huggingface.co/h94/IP-Adapter/resolve/main/"
        "models/image_encoder/pytorch_model.bin"
    ),
}

def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [OK] Already exists: {dest.name}")
        return
    print(f"  [..] Downloading {dest.name}...")
    try:
        def progress(block, block_size, total):
            if total > 0:
                pct = min(100, int(block * block_size * 100 / total))
                print(f"       {pct}%\r", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print(f"  [OK] {dest.name} ({dest.stat().st_size // 1024**2}MB)")
    except Exception as e:
        print(f"  [ERR] Failed to download {dest.name}: {e}")
        sys.exit(1)

print("\n" + "="*55)
print("  AMD Adapt Kiosk — Model Setup")
print("="*55 + "\n")

# Check SDXL base model exists
sdxl_path = Path.home() / "ComfyUI" / "models" / "checkpoints" / "sd_xl_base_1.0.safetensors"
if sdxl_path.exists():
    print(f"[OK] SDXL base 1.0 found at {sdxl_path}")
else:
    print(f"[ERR] SDXL base 1.0 not found at {sdxl_path}")
    print("      Make sure sd_xl_base_1.0.safetensors is in ~/ComfyUI/models/checkpoints/")
    sys.exit(1)

# Download IP-Adapter
print("\n[..] Downloading IP-Adapter SDXL weights (~600MB total)...")
for filename, url in IP_ADAPTER_FILES.items():
    dest = IP_ADAPTER_DIR / filename
    download(url, dest)

# Verify diffusers is installed
print("\n[..] Checking Python packages...")
missing = []
for pkg in ["diffusers", "transformers", "accelerate", "safetensors", "cv2"]:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError:
        print(f"  [ERR] {pkg} not installed")
        missing.append(pkg)

if missing:
    print(f"\n[..] Installing missing packages...")
    import subprocess
    pkgs = []
    for p in missing:
        if p == "cv2":
            pkgs.append("opencv-python")
        else:
            pkgs.append(p)
    subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs, check=True)
    print("[OK] Packages installed")

print("\n[OK] Setup complete — run: python start.py\n")
