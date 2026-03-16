#!/usr/bin/env python3
"""
diagnose.py — AMD Adapt Kiosk diagnostic
Checks everything and reports exactly what is broken.
Run with: python diagnose.py
"""
import os
import sys
import json
import subprocess
import urllib.request
from pathlib import Path

COMFY_URL  = "http://127.0.0.1:8188"
COMFY_DIR  = Path.home() / "ComfyUI"
COMFY_VENV = Path.home() / "comfyui-venv"

SEP = "=" * 55

def ok(msg):  print(f"  [OK]   {msg}")
def err(msg): print(f"  [ERR]  {msg}")
def warn(msg):print(f"  [WARN] {msg}")
def info(msg):print(f"  [INFO] {msg}")

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ── 1. Python + venv ──────────────────────────────────────────────────────────
section("1. Python environment")
info(f"Running Python: {sys.executable}")
info(f"Python version: {sys.version.split()[0]}")

if "comfyui-venv" in sys.executable:
    warn("Running inside comfyui-venv — kiosk packages may be missing")
elif "facemorph" in sys.executable or "venv" in sys.executable:
    ok("Running inside kiosk venv")
else:
    warn(f"Not in a venv — using system Python")

# ── 2. PyTorch in comfyui-venv ────────────────────────────────────────────────
section("2. PyTorch / ROCm (in comfyui-venv)")
python_bin = COMFY_VENV / "bin" / "python3"
if not python_bin.exists():
    err(f"comfyui-venv not found at {COMFY_VENV}")
else:
    ok(f"comfyui-venv exists at {COMFY_VENV}")
    result = subprocess.run(
        [str(python_bin), "-c",
         "import torch; "
         "print('TORCH_VERSION:', torch.__version__); "
         "print('CUDA_AVAILABLE:', torch.cuda.is_available()); "
         "print('DEVICE_COUNT:', torch.cuda.device_count()); "
         "name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; "
         "print('DEVICE_NAME:', name); "
         "print('ROCM:', hasattr(torch.version, 'hip') and torch.version.hip is not None); "
         "print('HIP_VERSION:', getattr(torch.version, 'hip', 'N/A')); "
        ],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        err(f"PyTorch import failed in comfyui-venv:")
        print(f"    {result.stderr.strip()[:300]}")
    else:
        for line in result.stdout.strip().split("\n"):
            key, _, val = line.partition(":")
            val = val.strip()
            if key == "TORCH_VERSION":
                ok(f"PyTorch version: {val}")
                if "rocm" not in val.lower():
                    err(f"PyTorch is NOT a ROCm build! Got: {val}")
                    err("This is likely the root cause. PyTorch needs to be reinstalled for ROCm.")
            elif key == "CUDA_AVAILABLE":
                if val == "True":
                    ok("CUDA/ROCm device visible: True")
                else:
                    err("CUDA/ROCm device NOT visible — GPU won't work")
                    err("Check that ROCm is installed: rocm-smi")
            elif key == "DEVICE_NAME":
                ok(f"GPU: {val}")
            elif key == "HIP_VERSION":
                if val != "N/A":
                    ok(f"HIP version: {val}")

# ── 3. ComfyUI installation ───────────────────────────────────────────────────
section("3. ComfyUI installation")
if not COMFY_DIR.exists():
    err(f"ComfyUI not found at {COMFY_DIR}")
else:
    ok(f"ComfyUI directory exists: {COMFY_DIR}")
    main_py = COMFY_DIR / "main.py"
    if main_py.exists():
        ok("main.py found")
    else:
        err("main.py NOT found — ComfyUI is incomplete")

    # Check custom nodes
    custom = COMFY_DIR / "custom_nodes"
    nodes = [p.name for p in custom.iterdir()] if custom.exists() else []
    info(f"Custom nodes: {nodes}")
    for node in nodes:
        if not node.endswith("_disabled") and node not in ("__pycache__",):
            if node.endswith(".py"):
                ok(f"  {node} (single file node)")
            else:
                node_init = custom / node / "__init__.py"
                if node_init.exists():
                    ok(f"  {node}")
                else:
                    warn(f"  {node} — no __init__.py, may cause issues")

# ── 4. Models ─────────────────────────────────────────────────────────────────
section("4. Models")
checkpoints = COMFY_DIR / "models" / "checkpoints"
controlnet  = COMFY_DIR / "models" / "controlnet"

if checkpoints.exists():
    models = [f.name for f in checkpoints.iterdir() if not f.name.startswith("put_")]
    if models:
        ok(f"Checkpoints: {models}")
    else:
        err("No checkpoint models found")
else:
    err("Checkpoints directory missing")

if controlnet.exists():
    cnets = [f.name for f in controlnet.iterdir() if not f.name.startswith("put_")]
    if cnets:
        ok(f"ControlNet: {cnets}")
    else:
        warn("No ControlNet models found")

# ── 5. ComfyUI running ────────────────────────────────────────────────────────
section("5. ComfyUI server")
try:
    with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3) as r:
        stats = json.loads(r.read())
    ok("ComfyUI is running and responding")
    devices = stats.get("devices", [{}])
    if devices:
        d = devices[0]
        vram = d.get("vram_total", 0) // 1024**2
        vram_free = d.get("vram_free", 0) // 1024**2
        ok(f"VRAM: {vram_free}MB free / {vram}MB total")
except Exception as e:
    err(f"ComfyUI is NOT running: {e}")
    info("Start it first with: python start.py")
    info("Then run this script again for a full report")

# ── 6. Quick workflow test ────────────────────────────────────────────────────
section("6. Workflow test (simple txt2img, no ControlNet)")
try:
    urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=2)
    comfy_running = True
except Exception:
    comfy_running = False

if not comfy_running:
    warn("Skipping workflow test — ComfyUI not running")
else:
    import time, uuid
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "a red apple", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "ugly, blurry", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0],
                         "seed": 42, "steps": 4, "cfg": 1.0,
                         "sampler_name": "euler_ancestral",
                         "scheduler": "sgm_uniform", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "diag_test"}},
    }
    try:
        body = json.dumps({"prompt": workflow, "client_id": uuid.uuid4().hex}).encode()
        req  = urllib.request.Request(
            f"{COMFY_URL}/prompt", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            result    = json.loads(r.read())
            prompt_id = result.get("prompt_id")
        ok(f"Workflow submitted — prompt_id={prompt_id}")
        info("Waiting up to 60s for result...")
        start = time.time()
        success = False
        while time.time() - start < 60:
            time.sleep(2)
            elapsed = int(time.time() - start)
            try:
                with urllib.request.urlopen(
                        f"{COMFY_URL}/history/{prompt_id}", timeout=10) as r:
                    history = json.loads(r.read())
                entry = history.get(prompt_id)
                if entry:
                    outputs = entry.get("outputs", {})
                    status  = entry.get("status", {})
                    msgs    = status.get("messages", [])
                    if outputs:
                        ok(f"SUCCESS — image generated in {elapsed}s!")
                        ok("ComfyUI is working. The kiosk workflow code needs fixing.")
                        success = True
                    else:
                        err(f"FAILED — workflow ran but produced no output")
                        err(f"Status messages: {msgs}")
                        # Try to get error details
                        for msg in msgs:
                            if isinstance(msg, list) and len(msg) > 1:
                                err(f"  Error detail: {msg}")
                    break
                else:
                    print(f"  waiting... {elapsed}s\r", end="", flush=True)
            except Exception as poll_err:
                print(f"  poll error at {elapsed}s: {poll_err}\r", end="", flush=True)
        if not success and time.time() - start >= 60:
            err("Timed out after 60s — ComfyUI accepted job but never finished")
    except Exception as e:
        err(f"Failed to submit workflow: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary — next steps")
print("""
  Run this script and paste the full output.
  The [ERR] lines tell you exactly what to fix.

  Common fixes:
  - PyTorch not ROCm build → python start.py --reinstall
  - GPU not visible        → check ROCm: rocm-smi
  - Workflow failed        → paste the error detail lines above
""")
