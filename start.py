#!/usr/bin/env python3
"""
start.py — AI Scene Style Kiosk Launcher
Run with: python start.py

Auto-detects AMD GPU architecture and sets the correct ROCm environment
variables. Works on any AMD dGPU or APU with ROCm-compatible Linux drivers.
"""

import glob
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


# ── GFX version → HSA_OVERRIDE_GFX_VERSION mapping ──────────────────────────
# Maps detected gfx_target_version (from sysfs) to the nearest ROCm-supported
# architecture. Format in sysfs is numeric MMMNNRR (e.g. 110000 = gfx1100).
_GFX_OVERRIDE_MAP = {
    # GFX9 — Vega (GCN 5)
    "gfx900":  "9.0.0",
    "gfx902":  "9.0.0",
    "gfx904":  "9.0.0",
    "gfx906":  "9.0.6",
    "gfx908":  "9.0.8",
    "gfx90a":  "9.0.10",
    "gfx90c":  "9.0.0",    # Renoir/Cezanne APU → nearest supported
    "gfx940":  None,        # MI300 — officially supported, no override needed
    "gfx941":  None,
    "gfx942":  None,
    # GFX10 — RDNA 1 (Navi 1x)
    "gfx1010": "10.1.0",
    "gfx1011": "10.1.0",
    "gfx1012": "10.1.0",
    "gfx1013": "10.1.0",
    # GFX10.3 — RDNA 2 (Navi 2x)
    "gfx1030": "10.3.0",
    "gfx1031": "10.3.0",
    "gfx1032": "10.3.0",
    "gfx1033": "10.3.0",
    "gfx1034": "10.3.0",
    "gfx1035": "10.3.0",   # Rembrandt APU
    "gfx1036": "10.3.0",   # Raphael/Phoenix APU iGPU
    # GFX11 — RDNA 3 (Navi 3x)
    "gfx1100": "11.0.0",
    "gfx1101": "11.0.0",
    "gfx1102": "11.0.0",
    "gfx1103": "11.0.0",   # Phoenix/Hawk Point APU
    # GFX11.5 — RDNA 3.5 (Strix Point APU)
    "gfx1150": "11.0.0",
    "gfx1151": "11.0.0",
    # GFX12 — RDNA 4 (Navi 4x)
    "gfx1200": None,        # Officially supported in ROCm 7+
    "gfx1201": None,
}


def _detect_gfx_version():
    """Detect AMD GPU GFX version from KFD sysfs topology.
    Returns (gfx_string, node_path) or (None, None) if no GPU found."""
    kfd_nodes = "/sys/class/kfd/kfd/topology/nodes"
    if not os.path.isdir(kfd_nodes):
        return None, None

    for node_dir in sorted(glob.glob(os.path.join(kfd_nodes, "*"))):
        props_file = os.path.join(node_dir, "properties")
        if not os.path.isfile(props_file):
            continue
        try:
            with open(props_file) as f:
                for line in f:
                    if line.startswith("gfx_target_version"):
                        val = int(line.split()[1])
                        if val == 0:
                            continue  # CPU node, skip
                        # Decode MMMNNRR → gfxMMMNN (hex-style for gfx9)
                        major = val // 10000
                        minor = (val % 10000) // 100
                        patch = val % 100
                        # gfx9 series uses hex for minor (e.g. 90a = 9.0.10)
                        if major == 9 and minor == 0 and patch == 10:
                            gfx = "gfx90a"
                        elif major == 9 and minor == 0 and patch >= 10:
                            gfx = f"gfx{major:d}0{patch:x}"
                        else:
                            gfx = f"gfx{major}{minor:01d}{patch:02d}"
                        return gfx, node_dir
        except (IOError, ValueError):
            continue
    return None, None


def _detect_render_devices():
    """Find all /dev/dri/renderD* devices."""
    devices = sorted(glob.glob("/dev/dri/renderD*"))
    return devices if devices else ["/dev/dri/renderD128"]


def _setup_gpu_env():
    """Auto-detect GPU and set ROCm environment variables."""
    gfx, node_path = _detect_gfx_version()

    if gfx:
        print(f"[OK] Detected AMD GPU: {gfx}")
        override = _GFX_OVERRIDE_MAP.get(gfx)
        if override is None and gfx in _GFX_OVERRIDE_MAP:
            # Officially supported — no override needed
            print(f"     Officially supported by ROCm — no override needed")
        elif override:
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = override
            print(f"     HSA_OVERRIDE_GFX_VERSION={override}")
        else:
            # Unknown GFX — try to guess nearest supported version
            print(f"[WARN] Unknown GFX version: {gfx}")
            print(f"       You may need to set HSA_OVERRIDE_GFX_VERSION manually")
    else:
        print("[WARN] Could not detect AMD GPU from sysfs")
        print("       Falling back to HSA_OVERRIDE_GFX_VERSION=11.0.0")
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    # Common ROCm environment
    os.environ.update({
        "AMD_LOG_LEVEL": "0",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    })

    # Auto-detect render devices
    render_devs = _detect_render_devices()
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"

    # Set permissions on all detected render devices + kfd
    perm_devices = ["/dev/kfd"] + render_devs
    for dev in perm_devices:
        if os.path.exists(dev):
            try:
                subprocess.run(["sudo", "chmod", "666", dev],
                               capture_output=True, timeout=5)
            except Exception:
                pass


def main():
    print("\n" + "="*55)
    print("  AMD-ADAPT  |  Scene Style Transfer Kiosk")
    print("  http://localhost:8000")
    print("="*55 + "\n")

    # ── GPU auto-detection ────────────────────────────────────────────
    _setup_gpu_env()

    # ── Check required packages ───────────────────────────────────────
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

    # ── Check models ──────────────────────────────────────────────────
    sdxl_turbo = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                  "sd_xl_turbo_1.0_fp16.safetensors")

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
