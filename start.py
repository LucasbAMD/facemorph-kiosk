#!/usr/bin/env python3
"""
start.py — AMD Adapt Kiosk
Run with: python start.py

No ComfyUI needed — uses diffusers directly.
First time setup: python setup_models.py
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

def main():
    print("\n" + "="*55)
    print("  AMD ADAPT KIOSK")
    print("  http://localhost:8000")
    print("="*55 + "\n")

    # Set ROCm env vars
    os.environ.update({
        "HSA_OVERRIDE_GFX_VERSION":  "11.0.0",
        "AMD_LOG_LEVEL":             "0",
        "HIP_VISIBLE_DEVICES":       "0",
        "ROCR_VISIBLE_DEVICES":      "0",
        "PYTORCH_HIP_ALLOC_CONF":    "expandable_segments:True",
    })

    # GPU permissions
    for dev in ["/dev/kfd", "/dev/dri/renderD128"]:
        try:
            subprocess.run(["sudo", "chmod", "666", dev],
                           capture_output=True, timeout=5)
        except Exception:
            pass

    # Check packages
    missing = []
    for pkg in ("uvicorn", "fastapi", "cv2", "diffusers", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERR] Missing packages: {', '.join(missing)}")
        print("      Run: pip install uvicorn fastapi opencv-python diffusers transformers accelerate safetensors")
        sys.exit(1)

    # Check model exists
    sdxl = Path.home() / "ComfyUI" / "models" / "checkpoints" / "sd_xl_base_1.0.safetensors"
    if not sdxl.exists():
        print(f"[ERR] SDXL model not found at {sdxl}")
        sys.exit(1)

    ip_adapter = Path.home() / "kiosk_models" / "ip_adapter" / "sdxl_models" / "ip-adapter_sdxl.bin"
    if not ip_adapter.exists():
        print("[WARN] IP-Adapter not found — face identity preservation disabled")
        print("       Run: python setup_models.py")
    else:
        print("[OK] IP-Adapter found — face identity preservation active")

    print("\n[OK] Starting kiosk at http://localhost:8000")
    print("     AI pipeline loading in background...\n")

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)

if __name__ == "__main__":
    main()
