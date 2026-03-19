#!/usr/bin/env python3
"""
start.py — AI Scene Style Kiosk Launcher
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
    print("  AMD-ADAPT  |  Scene Style Transfer Kiosk")
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

    # Check models
    sdxl_turbo = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                  "sd_xl_turbo_1.0_fp16.safetensors")

    # Check for ControlNet (HF cache)
    try:
        from diffusers import ControlNetModel
        ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            local_files_only=True,
        )
        print("[OK] ControlNet Depth SDXL — found in cache")
        print("[OK] Mode: ControlNet + SDXL Base (best quality)")
    except Exception:
        if sdxl_turbo.exists():
            print("[OK] SDXL Turbo found — using fast mode")
            print("[INFO] For best quality, run: python setup_models.py")
        else:
            print("[ERR] No AI models found!")
            print("      Run: python setup_models.py")
            sys.exit(1)

    print(f"\n[OK] Starting kiosk at http://localhost:8000")
    print("     AI pipeline loading in background...\n")

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)


if __name__ == "__main__":
    main()
