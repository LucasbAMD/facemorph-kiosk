#!/usr/bin/env python3
"""
start.py — AMD Adapt Kiosk
Single command to launch everything:
  1. Starts ComfyUI in the background (same flags as start_comfyui.sh)
  2. Waits for ComfyUI to come online
  3. Starts the FastAPI kiosk server

Usage:
    python start.py              # normal start
    python start.py --reinstall  # reinstall ComfyUI keeping your models
"""

import os
import sys
import time
import signal
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
COMFY_DIR  = Path(os.environ.get("COMFYUI_PATH", Path.home() / "ComfyUI"))
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
    for dev in ["/dev/kfd", "/dev/dri/renderD128"]:
        try:
            subprocess.run(["sudo", "chmod", "666", dev],
                           capture_output=True, timeout=5)
        except Exception:
            pass

def disable_broken_custom_nodes():
    """
    Disable custom nodes that crash on import and poison the node registry.
    ComfyUI_InstantID requires insightface which is not available on ROCm.
    When it fails to import it causes ALL workflows to fail instantly.
    """
    custom_nodes = COMFY_DIR / "custom_nodes"
    if not custom_nodes.exists():
        return
    for node in ["ComfyUI_InstantID"]:
        src  = custom_nodes / node
        dest = custom_nodes / f"{node}_disabled"
        if src.exists() and not dest.exists():
            try:
                src.rename(dest)
                print(f"[OK] Disabled broken custom node: {node}")
            except Exception as e:
                print(f"[WARN] Could not disable {node}: {e}")
        elif dest.exists():
            print(f"[OK] {node} already disabled")

def run(cmd, cwd=None):
    print(f"[..] {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERR] Command failed (code {result.returncode}): {cmd}")
        sys.exit(1)

def reinstall_comfyui():
    """
    Reinstall ComfyUI cleanly while preserving all model files.
      1. Back up models folder (~20GB, kept on same disk so it's instant)
      2. Remove broken ComfyUI directory
      3. Clone fresh ComfyUI from GitHub
      4. Reinstall Python dependencies in existing comfyui-venv
      5. Restore models
    """
    print("\n" + "="*60)
    print("  ComfyUI Reinstall — preserving your models")
    print("="*60 + "\n")

    models_dir    = COMFY_DIR / "models"
    models_backup = Path.home() / "comfyui_models_backup"

    # Step 1 — back up models (same filesystem = fast rename, not copy)
    if models_dir.exists():
        if models_backup.exists():
            print(f"[..] Removing old backup...")
            shutil.rmtree(models_backup)
        print(f"[..] Moving models to backup (fast — same disk)...")
        shutil.move(str(models_dir), str(models_backup))
        print(f"[OK] Models safely backed up to {models_backup}")
    else:
        print("[WARN] No models directory found")
        models_backup = None

    # Step 2 — remove broken install
    if COMFY_DIR.exists():
        print(f"[..] Removing broken ComfyUI...")
        shutil.rmtree(COMFY_DIR)
        print(f"[OK] Removed {COMFY_DIR}")

    # Step 3 — fresh clone
    print("[..] Cloning fresh ComfyUI from GitHub...")
    run(f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}",
        cwd=Path.home())
    print("[OK] ComfyUI cloned")

    # Step 4 — reinstall deps in existing venv
    if not COMFY_VENV.exists():
        print("[..] Creating comfyui-venv...")
        run(f"python3 -m venv {COMFY_VENV}")

    pip = COMFY_VENV / "bin" / "pip"
    print("[..] Installing PyTorch for ROCm 6.2 (this takes a few minutes)...")
    run(f"{pip} install torch torchvision torchaudio "
        f"--index-url https://download.pytorch.org/whl/rocm6.2 -q")
    print("[..] Installing ComfyUI requirements...")
    run(f"{pip} install -r {COMFY_DIR}/requirements.txt -q")
    print("[..] Installing extra packages...")
    run(f"{pip} install torchsde kornia spandrel requests -q")

    # Step 5 — restore models
    if models_backup and models_backup.exists():
        print(f"[..] Restoring models...")
        restore_dir = COMFY_DIR / "models"
        if restore_dir.exists():
            shutil.rmtree(restore_dir)
        shutil.move(str(models_backup), str(restore_dir))
        print(f"[OK] Models restored")

    # Output dir
    (COMFY_DIR / "output").mkdir(exist_ok=True)

    print("\n[OK] ComfyUI reinstalled successfully — starting kiosk...\n")


def start_comfyui():
    if not COMFY_DIR.exists():
        print("[ERR] ComfyUI not found. Run: python start.py --reinstall")
        sys.exit(1)
    if not COMFY_VENV.exists():
        print("[ERR] ComfyUI venv not found. Run: python start.py --reinstall")
        sys.exit(1)

    python_bin = COMFY_VENV / "bin" / "python3"
    output_dir = COMFY_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "HSA_OVERRIDE_GFX_VERSION":                "11.0.0",
        "PYTORCH_TUNABLEOP_ENABLED":               "1",
        "DISABLE_ADDMM_CUDA_LT":                   "1",
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "AMD_LOG_LEVEL":                           "0",
        "HIP_VISIBLE_DEVICES":                     "0",
        "ROCR_VISIBLE_DEVICES":                    "0",
        "PYTORCH_HIP_ALLOC_CONF":                  "expandable_segments:True",
    })

    cmd = [
        str(python_bin), "main.py",
        "--port",             "8188",
        "--listen",           "127.0.0.1",
        "--disable-auto-launch",
        "--force-fp16",
        "--bf16-vae",
        "--lowvram",
        "--output-directory", str(output_dir),
    ]

    print("[..] Starting ComfyUI in background...")
    return subprocess.Popen(
        cmd, cwd=str(COMFY_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

def stream_comfy_logs(proc):
    import threading
    def _read():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [ComfyUI] {line}")
    threading.Thread(target=_read, daemon=True).start()

def wait_for_comfy(proc, timeout=180):
    print(f"[..] Waiting for ComfyUI (up to {timeout}s)...")
    start = time.time()
    dots  = 0
    while time.time() - start < timeout:
        if proc.poll() is not None:
            print(f"\n[ERR] ComfyUI exited unexpectedly (code {proc.returncode})")
            sys.exit(1)
        if is_comfy_online():
            print(f"\n[OK] ComfyUI online ({int(time.time()-start)}s)")
            return
        time.sleep(1)
        dots += 1
        print("." * (dots % 40) + "\r", end="", flush=True)
    print(f"\n[WARN] ComfyUI did not respond in {timeout}s — starting kiosk anyway.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    reinstall = "--reinstall" in sys.argv

    print("\n" + "="*60)
    print("  AMD ADAPT KIOSK")
    print("  Kiosk  → http://localhost:8000")
    print("  AI     → http://127.0.0.1:8188")
    print("="*60 + "\n")

    if reinstall:
        reinstall_comfyui()

    fix_gpu_permissions()
    disable_broken_custom_nodes()

    if is_comfy_online():
        print("[OK] ComfyUI already running — skipping launch")
        comfy_proc = None
    else:
        comfy_proc = start_comfyui()
        stream_comfy_logs(comfy_proc)
        wait_for_comfy(comfy_proc)

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

    missing = []
    for pkg in ("uvicorn", "fastapi", "cv2"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERR] Missing packages: {', '.join(missing)}")
        print("      Activate the kiosk venv first:")
        print("        source ~/facemorph-kiosk/venv/bin/activate")
        print("        pip install uvicorn fastapi opencv-python")
        sys.exit(1)

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)


if __name__ == "__main__":
    main()
