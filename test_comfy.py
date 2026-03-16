#!/usr/bin/env python3
"""
test_comfy.py — Quick ComfyUI diagnostic
Run this to check if ComfyUI itself works independently of the kiosk.

Usage:
    python test_comfy.py
"""
import json
import time
import urllib.request

COMFY_URL = "http://127.0.0.1:8188"

def api(endpoint, data=None, method="GET"):
    url  = f"{COMFY_URL}/{endpoint}"
    body = json.dumps(data).encode() if data else None
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

# Simplest possible workflow — no ControlNet, no img2img, just text → image
WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"}
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a red apple, photorealistic", "clip": ["1", 1]}
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "ugly, blurry", "clip": ["1", 1]}
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1}
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "model":        ["1", 0],
            "positive":     ["2", 0],
            "negative":     ["3", 0],
            "latent_image": ["4", 0],
            "seed":         42,
            "steps":        4,
            "cfg":          1.0,
            "sampler_name": "euler_ancestral",
            "scheduler":    "sgm_uniform",
            "denoise":      1.0,
        }
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": "kiosk_test"}
    },
}

print("\n" + "="*50)
print("  ComfyUI Diagnostic Test")
print("="*50 + "\n")

# Check ComfyUI is running
try:
    stats = api("system_stats")
    print(f"[OK] ComfyUI is running")
    print(f"     PyTorch: {stats.get('system', {}).get('pytorch_version', 'unknown')}")
    print(f"     VRAM:    {stats.get('devices', [{}])[0].get('vram_total', 0) // 1024**2} MB")
except Exception as e:
    print(f"[ERR] ComfyUI not reachable: {e}")
    exit(1)

# Submit workflow
print("\n[..] Submitting simple text-to-image workflow (no ControlNet)...")
try:
    result    = api("prompt", {"prompt": WORKFLOW}, "POST")
    prompt_id = result["prompt_id"]
    print(f"[OK] Accepted — prompt_id={prompt_id}")
except Exception as e:
    print(f"[ERR] Failed to submit workflow: {e}")
    exit(1)

# Poll for result
print("[..] Waiting for result (up to 60s)...")
start = time.time()
while time.time() - start < 60:
    time.sleep(2)
    elapsed = int(time.time() - start)
    try:
        history = api(f"history/{prompt_id}")
        entry   = history.get(prompt_id)
        if entry:
            outputs = entry.get("outputs", {})
            if outputs:
                print(f"\n[OK] SUCCESS — ComfyUI generated an image in {elapsed}s!")
                print(f"     Output nodes: {list(outputs.keys())}")
                print("\n     The kiosk workflow code is the problem, not ComfyUI.")
                print("     Share this result and we can fix comfy_bridge.py.\n")
            else:
                status = entry.get("status", {})
                msgs   = status.get("messages", [])
                print(f"\n[ERR] Workflow completed but no output — failed in ComfyUI.")
                print(f"      Status messages: {msgs}")
                print("\n      This means the ComfyUI installation has an issue.")
                print("      Most likely fix: reinstall ComfyUI or check model files.\n")
            exit(0)
        else:
            print(f"  {elapsed}s — waiting...", end="\r")
    except Exception as e:
        print(f"  {elapsed}s — poll error: {e}", end="\r")

print(f"\n[ERR] Timed out after 60s — ComfyUI accepted the job but never finished.")
print("      Check the ComfyUI terminal for errors.\n")
