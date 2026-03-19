#!/usr/bin/env python3
"""
start.py — AI Avatar Kiosk Launcher
Run with: python start.py
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def main():
    print("\n" + "="*55)
    print("  AI AVATAR KIOSK")
    print("  http://localhost:8000")
    print("="*55 + "\n")

    # ROCm environment
    os.environ.update({
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
        "AMD_LOG_LEVEL": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    })

    # GPU permissions
    for dev in ["/dev/kfd", "/dev/dri/renderD128"]:
        try:
            subprocess.run(["sudo", "chmod", "666", dev],
                           capture_output=True, timeout=5)
        except Exception:
            pass

    # Check required packages
    missing = []
    for pkg in ("uvicorn", "fastapi", "cv2", "diffusers", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERR] Missing packages: {', '.join(missing)}")
        print("      Run: pip install -r requirements.txt")
        sys.exit(1)

    # Check model
    sdxl = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
            "sd_xl_turbo_1.0_fp16.safetensors")
    if not sdxl.exists():
        print(f"[ERR] SDXL model not found at {sdxl}")
        sys.exit(1)

    # Check IP-Adapter
    ip_bin = Path.home() / "kiosk_models" / "ip_adapter" / "sdxl_models" / "ip-adapter_sdxl.bin"
    ip_enc = Path.home() / "kiosk_models" / "ip_adapter" / "models" / "image_encoder"
    if ip_bin.exists() and ip_enc.exists():
        print("[OK] IP-Adapter found - face identity preservation active")
    else:
        print("[WARN] IP-Adapter not found - avatars will be style-only (no face matching)")
        print("       Run: python setup_models.py")

    print("\n[OK] Starting kiosk at http://localhost:8000")
    print("     AI pipeline loading in background...\n")

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)


if __name__ == "__main__":
    main()
