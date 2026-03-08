#!/usr/bin/env python3
"""
launch_comfy.py — Start ComfyUI as a background service for AMD Adapt Kiosk
Run this BEFORE python start.py

Detects OS and launches ComfyUI with correct ROCm/DirectML flags.
ComfyUI must be installed at ~/ComfyUI or set COMFYUI_PATH env var.
"""
import os
import sys
import subprocess
import time
import urllib.request
from pathlib import Path

COMFY_PORT = 8188
COMFY_URL  = f"http://127.0.0.1:{COMFY_PORT}"

def find_comfyui():
    candidates = [
        os.environ.get("COMFYUI_PATH",""),
        Path.home() / "ComfyUI",
        Path.home() / "comfyui",
        Path("../ComfyUI"),
        Path("C:/ComfyUI"),
    ]
    for p in candidates:
        p = Path(p)
        if p.exists() and (p / "main.py").exists():
            return p
    return None

def is_running():
    try:
        urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=2)
        return True
    except: return False

def launch():
    comfy_dir = find_comfyui()
    if not comfy_dir:
        print("\n[ERR] ComfyUI not found!")
        print("      Install it first:")
        print("      git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI")
        print("      Then run setup_comfy.sh\n")
        sys.exit(1)

    print(f"[OK] Found ComfyUI at: {comfy_dir}")

    if is_running():
        print(f"[OK] ComfyUI already running on port {COMFY_PORT}")
        return

    # Determine launch args based on OS and GPU
    import platform
    is_win = platform.system() == "Windows"

    if is_win:
        # Windows: use DirectML
        python = comfy_dir.parent / "comfyui-venv" / "Scripts" / "python.exe"
        if not python.exists():
            python = sys.executable
        extra_flags = ["--directml"]
    else:
        # Linux: ROCm
        python = comfy_dir.parent / "comfyui-venv" / "bin" / "python"
        if not python.exists():
            python = sys.executable
        extra_flags = []

    env = os.environ.copy()
    if not is_win:
        env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"   # W7900 = gfx1100
        env["PYTORCH_TUNABLEOP_ENABLED"] = "1"
        env["DISABLE_ADDMM_CUDA_LT"]    = "1"

    cmd = [
        str(python), str(comfy_dir / "main.py"),
        "--port", str(COMFY_PORT),
        "--listen", "127.0.0.1",
        "--output-directory", str(Path("comfy_outputs").absolute()),
        "--disable-auto-launch",
    ] + extra_flags

    print(f"[..] Launching ComfyUI (port {COMFY_PORT})...")
    log_f = open("comfy.log", "w")
    subprocess.Popen(cmd, cwd=str(comfy_dir), env=env,
                     stdout=log_f, stderr=log_f)

    # Wait for it to come up
    for i in range(60):
        time.sleep(1)
        if is_running():
            print(f"[OK] ComfyUI ready! ({i+1}s)")
            print(f"     UI available at http://localhost:{COMFY_PORT}")
            return
        if i % 5 == 4:
            print(f"[..] Waiting for ComfyUI... ({i+1}s)")

    print("[WARN] ComfyUI taking longer than expected — check comfy.log")

if __name__ == "__main__":
    launch()
    print("\n[OK] ComfyUI is running. Now start the kiosk:")
    print("     python start.py\n")
