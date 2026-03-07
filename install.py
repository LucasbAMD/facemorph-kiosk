#!/usr/bin/env python3
"""
install.py — AMD Adapt Kiosk one-shot installer
Run this ONCE on a fresh machine:

    python install.py

Auto-detects your OS and GPU:
  Windows 11  → DirectML (AMD/NVIDIA/Intel GPU acceleration)
  Linux       → ROCm (AMD) or CUDA (NVIDIA) or CPU fallback

After install, every future start is just:
    python start.py
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────
def c(code, txt): return f"\033[{code}m{txt}\033[0m" if sys.platform != "win32" else txt
ok   = lambda t: print(c("32",   f"  [OK]  {t}"))
info = lambda t: print(c("36",   f"  [..]  {t}"))
warn = lambda t: print(c("33",   f"  [!!]  {t}"))
err  = lambda t: print(c("31",   f"  [ERR] {t}"))
hdr  = lambda t: print(c("1;36", f"\n── {t} " + "─" * max(0, 50 - len(t))))
bold = lambda t: print(c("1",    t))

def run(cmd, **kwargs):
    return subprocess.run(cmd, shell=True, check=False, **kwargs)

def run_check(cmd, **kwargs):
    r = run(cmd, **kwargs)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return r

def pip(packages, extra_args=""):
    cmd = f'"{VENV_PIP}" install {packages} {extra_args} --quiet'
    r = run(cmd)
    if r.returncode != 0:
        warn(f"pip install had issues: {packages}")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.resolve()
VENV_DIR = ROOT / "venv"
IS_WIN   = sys.platform == "win32"
VENV_PY  = VENV_DIR / ("Scripts/python.exe" if IS_WIN else "bin/python")
VENV_PIP = VENV_DIR / ("Scripts/pip.exe"    if IS_WIN else "bin/pip")

# ═════════════════════════════════════════════════════════════════════════════
print()
bold("═" * 56)
bold("   🎭  AMD Adapt Kiosk — Installer")
bold("═" * 56)

# ── Step 1: Python version ────────────────────────────────────────────────────
hdr("Step 1  Python Version")
major, minor = sys.version_info[:2]
if major < 3 or minor < 10:
    err(f"Python 3.10+ required. You have {major}.{minor}.")
    sys.exit(1)
ok(f"Python {major}.{minor} ✓")

# ── Step 2: OS + GPU detection ────────────────────────────────────────────────
hdr("Step 2  OS + GPU Detection")
gpu_mode = "cpu"
pt_index = None

if IS_WIN:
    ok("Windows detected → using DirectML (AMD/NVIDIA/Intel GPU acceleration)")
    gpu_mode = "directml"
else:
    # Linux — check ROCm first, then CUDA, then CPU
    rocm_found = (
        run("rocm-smi --version", capture_output=True).returncode == 0 or
        Path("/dev/kfd").exists() or
        Path("/opt/rocm").exists()
    )
    if rocm_found:
        ok("Linux + AMD ROCm detected")
        gpu_mode = "rocm"

        r = run("rocm-smi --version", capture_output=True, text=True)
        rocm_ver = "6.0"
        if r.returncode == 0:
            for token in r.stdout.split():
                if token.count(".") == 1:
                    try:
                        float(token); rocm_ver = token; break
                    except ValueError:
                        pass
        ok(f"ROCm version: {rocm_ver}")
        if   rocm_ver.startswith("6.2"): pt_index = "rocm6.2"
        elif rocm_ver.startswith("6.1"): pt_index = "rocm6.1"
        else:                             pt_index = "rocm6.0"

    elif run("nvidia-smi", capture_output=True).returncode == 0:
        r = run("nvidia-smi --query-gpu=name --format=csv,noheader",
                capture_output=True, text=True)
        ok(f"Linux + NVIDIA CUDA detected: {r.stdout.strip()}")
        gpu_mode = "cuda"

    else:
        warn("No GPU detected — CPU mode (still works well on Threadripper)")
        gpu_mode = "cpu"

print(f"     GPU mode: {c('1', gpu_mode.upper())}")

# ── Step 3: Virtual environment ───────────────────────────────────────────────
hdr("Step 3  Virtual Environment")
if not VENV_DIR.exists():
    info("Creating venv...")
    run_check(f'"{sys.executable}" -m venv "{VENV_DIR}"')
ok("venv ready")
run(f'"{VENV_PIP}" install --upgrade pip --quiet')

# ── Step 4: GPU compute stack ─────────────────────────────────────────────────
hdr("Step 4  GPU Compute Stack")

if gpu_mode == "directml":
    info("Installing PyTorch (CPU base) + onnxruntime-directml for Windows GPU...")
    pip("torch torchvision",
        "--index-url https://download.pytorch.org/whl/cpu")
    run(f'"{VENV_PIP}" uninstall -y onnxruntime onnxruntime-gpu --quiet')
    pip("onnxruntime-directml")
    ok("DirectML stack installed (AMD W7900 GPU acceleration active)")

elif gpu_mode == "rocm":
    info(f"Installing PyTorch for ROCm ({pt_index}) — ~2.5 GB, this will take a few minutes...")
    pip("torch torchvision",
        f"--index-url https://download.pytorch.org/whl/{pt_index}")
    info("Installing onnxruntime-rocm...")
    run(f'"{VENV_PIP}" uninstall -y onnxruntime onnxruntime-gpu --quiet')
    r = run(f'"{VENV_PIP}" install onnxruntime-rocm --quiet')
    if r.returncode != 0:
        r2 = run(f'"{VENV_PIP}" install onnxruntime-rocm '
                 f'--index-url https://download.pytorch.org/whl/{pt_index} --quiet')
        if r2.returncode != 0:
            warn("onnxruntime-rocm not found on PyPI — falling back to CPU ONNX")
            pip("onnxruntime")
    ok("ROCm stack installed")

elif gpu_mode == "cuda":
    info("Installing PyTorch for CUDA 12.1...")
    pip("torch torchvision",
        "--index-url https://download.pytorch.org/whl/cu121")
    run(f'"{VENV_PIP}" uninstall -y onnxruntime --quiet')
    pip("onnxruntime-gpu")
    ok("CUDA stack installed")

else:
    info("Installing PyTorch (CPU)...")
    pip("torch torchvision",
        "--index-url https://download.pytorch.org/whl/cpu")
    pip("onnxruntime")
    ok("CPU stack installed")

# ── Step 5: Core packages ─────────────────────────────────────────────────────
hdr("Step 5  Core Packages")
info("Installing FastAPI, OpenCV, MediaPipe, InsightFace...")
pip(" ".join([
    "fastapi==0.111.0",
    "uvicorn[standard]==0.29.0",
    "opencv-python>=4.9.0.80",
    "numpy>=2.0.0",
    "Pillow>=11.0.0",
    "insightface==0.7.3",
    "mediapipe>=0.10.30",
    "python-multipart==0.0.9",
    "aiofiles==23.2.1",
    "requests==2.31.0",
]))
ok("All packages installed")

# ── Step 6: Folders ───────────────────────────────────────────────────────────
hdr("Step 6  Project Folders")
for d in ["faces/celebrity", "faces/fantasy", "faces/custom", "models", "static"]:
    Path(d).mkdir(parents=True, exist_ok=True)
ok("Directories ready")

# ── Step 7: GPU verification ──────────────────────────────────────────────────
hdr("Step 7  GPU Verification")
verify = subprocess.run(
    [str(VENV_PY), "-c",
     "import onnxruntime as o; p=o.get_available_providers(); "
     "print('DML ACTIVE'  if 'DmlExecutionProvider'  in p else "
     "'ROCm ACTIVE' if 'ROCMExecutionProvider' in p else "
     "'CUDA ACTIVE' if 'CUDAExecutionProvider' in p else 'CPU ONLY'); print(p)"],
    capture_output=True, text=True
)
if verify.returncode == 0:
    out = verify.stdout.strip()
    if "ACTIVE" in out:
        ok(f"GPU verified: {out.split()[0]} {out.split()[1]}")
    else:
        warn(f"Running CPU only — {out}")
else:
    warn("Could not verify ONNX runtime")

# ── Done ──────────────────────────────────────────────────────────────────────
print()
bold("═" * 56)
bold("   ✅  Install Complete!")
bold("═" * 56)
print("""
NEXT STEPS:
──────────────────────────────────────────────────────
1. ADD FACE PHOTOS (for Face Swap mode):
   Drop JPG/PNG files into:
     faces/celebrity/   e.g. Lisa_Su.jpg
     faces/fantasy/     e.g. Thanos.jpg
   Use clear, front-facing photos.

2. DOWNLOAD FACE SWAP MODEL (optional):
   Windows (PowerShell):
     Invoke-WebRequest -Uri https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx -OutFile models\\inswapper_128.onnx
   Linux:
     wget -P models/ https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx

   Na'vi, Hulk, Thanos, Predator, Ghost, Groot work WITHOUT this model.

3. START THE KIOSK:
   python start.py
   Browser opens automatically at http://localhost:8000

KEYBOARD SHORTCUTS:
  1-6   Pick character
  F     Open Face Swap panel
  ESC   Return to live view
──────────────────────────────────────────────────────
""")
