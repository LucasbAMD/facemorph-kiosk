#!/usr/bin/env python3
"""
start.py — AMD Adapt Kiosk
Single command to launch everything:
  1. Starts ComfyUI in the background (same flags as start_comfyui.sh)
  2. Waits for ComfyUI to come online
  3. Starts the FastAPI kiosk server

Usage:
    source ~/comfyui-venv/bin/activate   # only needed once per shell session
    python start.py
"""

import os
import sys
import time
import signal
import subprocess
import urllib.request
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
COMFY_DIR = Path(os.environ.get("COMFYUI_PATH", Path.home() / "ComfyUI"))
COMFY_VENV = Path.home() / "comfyui-venv"
COMFY_URL  = "http://127.0.0.1:8188"

os.chdir(ROOT)

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_comfy_online():
    try:
        urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=2)
        return True
    except Exception:
        return False

def fix_gpu_permissions():
    """Same permission fix as start_comfyui.sh — needed every launch on Ubuntu."""
    for dev in ["/dev/kfd", "/dev/dri/renderD128"]:
        try:
            subprocess.run(["sudo", "chmod", "666", dev],
                           capture_output=True, timeout=5)
        except Exception:
            pass

def start_comfyui():
    """
    Launch ComfyUI as a background subprocess using the comfyui-venv Python.
    Mirrors all flags and env vars from start_comfyui.sh.
    Returns the Popen object so we can terminate it on exit.
    """
    if not COMFY_DIR.exists():
        print("[ERR] ComfyUI not found. Run: bash setup_ubuntu.sh")
        sys.exit(1)
    if not COMFY_VENV.exists():
        print("[ERR] ComfyUI venv not found. Run: bash setup_ubuntu.sh")
        sys.exit(1)

    python_bin = COMFY_VENV / "bin" / "python3"
    output_dir = Path.home() / "ComfyUI" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "HSA_OVERRIDE_GFX_VERSION":              "11.0.0",
        "PYTORCH_TUNABLEOP_ENABLED":             "1",
        "DISABLE_ADDMM_CUDA_LT":                 "1",
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "AMD_LOG_LEVEL":                         "0",
        "HIP_VISIBLE_DEVICES":                   "0",
        "ROCR_VISIBLE_DEVICES":                  "0",
    })

    cmd = [
        str(python_bin), "main.py",
        "--port",             "8188",
        "--listen",           "127.0.0.1",
        "--disable-auto-launch",
        "--force-fp16",
        "--lowvram",
        "--output-directory", str(output_dir),
    ]

    print("[..] Starting ComfyUI in background...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(COMFY_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc

def stream_comfy_logs(proc):
    """Forward ComfyUI stdout to our console with a prefix, in a daemon thread."""
    import threading
    def _read():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [ComfyUI] {line}")
    t = threading.Thread(target=_read, daemon=True)
    t.start()

def wait_for_comfy(proc, timeout=120):
    """
    Poll until ComfyUI responds or times out.
    Exits early if the subprocess dies unexpectedly.
    """
    print(f"[..] Waiting for ComfyUI to come online (up to {timeout}s)...")
    start = time.time()
    dots  = 0
    while time.time() - start < timeout:
        if proc.poll() is not None:
            print(f"\n[ERR] ComfyUI exited unexpectedly (code {proc.returncode})")
            print("      Check the output above for errors.")
            sys.exit(1)
        if is_comfy_online():
            elapsed = int(time.time() - start)
            print(f"\n[OK] ComfyUI online ({elapsed}s)")
            return
        time.sleep(1)
        dots += 1
        print("." * dots + "\r", end="", flush=True)
    print(f"\n[WARN] ComfyUI did not respond in {timeout}s — kiosk will start anyway.")
    print("       AI generation may fail until ComfyUI finishes loading.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  AMD ADAPT KIOSK")
    print("  Kiosk  → http://localhost:8000")
    print("  AI     → http://127.0.0.1:8188")
    print("="*60 + "\n")

    fix_gpu_permissions()

    # If ComfyUI is already running (e.g. from a previous session), skip launch
    if is_comfy_online():
        print("[OK] ComfyUI already running — skipping launch")
        comfy_proc = None
    else:
        comfy_proc = start_comfyui()
        stream_comfy_logs(comfy_proc)
        wait_for_comfy(comfy_proc)

    # Graceful shutdown: kill ComfyUI when the kiosk exits
    def _shutdown(sig, frame):
        print("\n[..] Shutting down...")
        if comfy_proc and comfy_proc.poll() is None:
            print("[..] Stopping ComfyUI...")
            comfy_proc.terminate()
            try:
                comfy_proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                comfy_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("\n[OK] Starting kiosk at http://localhost:8000\n")

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)


if __name__ == "__main__":
    main()
