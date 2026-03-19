#!/usr/bin/env python3
"""
setup_models.py — Download all model weights for the AI Avatar Kiosk.
Run this once before starting the kiosk for the first time.

Downloads:
  1. IP-Adapter FaceID SDXL — face identity preservation (uses face recognition embeddings)
  2. IP-Adapter FaceID LoRA — improves generation quality with FaceID
  3. insightface antelopev2 — face detection + embedding extraction (auto-downloaded)

Usage:
    python setup_models.py
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path.home() / "kiosk_models"

# ── IP-Adapter FaceID SDXL ────────────────────────────────────────────────────
# From https://huggingface.co/h94/IP-Adapter-FaceID
# These use actual face recognition embeddings (512-dim from insightface)
# instead of generic CLIP features, giving much stronger identity preservation.
FACEID_DIR = MODELS_DIR / "ip_adapter_faceid"
FACEID_FILES = {
    "ip-adapter-faceid_sdxl.bin": (
        "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/"
        "ip-adapter-faceid_sdxl.bin"
    ),
    "ip-adapter-faceid_sdxl_lora.safetensors": (
        "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/"
        "ip-adapter-faceid_sdxl_lora.safetensors"
    ),
}

# ── Regular IP-Adapter SDXL (fallback) ────────────────────────────────────────
# From https://huggingface.co/h94/IP-Adapter
IP_ADAPTER_DIR = MODELS_DIR / "ip_adapter"
IP_ADAPTER_FILES = {
    "sdxl_models/ip-adapter_sdxl.bin": (
        "https://huggingface.co/h94/IP-Adapter/resolve/main/"
        "sdxl_models/ip-adapter_sdxl.safetensors"
    ),
    "models/image_encoder/config.json": (
        "https://huggingface.co/h94/IP-Adapter/resolve/main/"
        "models/image_encoder/config.json"
    ),
    "models/image_encoder/model.safetensors": (
        "https://huggingface.co/h94/IP-Adapter/resolve/main/"
        "models/image_encoder/model.safetensors"
    ),
}


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        size_mb = dest.stat().st_size // (1024 * 1024)
        print(f"  [OK] Already exists: {dest.name} ({size_mb}MB)")
        return True
    print(f"  [..] Downloading {dest.name}...")
    try:
        def progress(block, block_size, total):
            if total > 0:
                done = min(block * block_size, total)
                pct = int(done * 100 / total)
                mb = done // (1024 * 1024)
                total_mb = total // (1024 * 1024)
                print(f"\r       {pct}%  ({mb}/{total_mb} MB)", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()
        size_mb = dest.stat().st_size // (1024 * 1024)
        print(f"  [OK] {dest.name} ({size_mb}MB)")
        return True
    except Exception as e:
        print(f"\n  [ERR] Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    print("\n" + "=" * 55)
    print("  AI Avatar Kiosk — Model Setup")
    print("=" * 55)

    # ── Check SDXL model ──────────────────────────────────────────────────
    print("\n[1/4] Checking SDXL model...")
    sdxl_path = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                 "sd_xl_turbo_1.0_fp16.safetensors")
    if sdxl_path.exists():
        print(f"  [OK] SDXL found at {sdxl_path}")
    else:
        print(f"  [ERR] SDXL not found at {sdxl_path}")
        print("        Download it first, then re-run this script.")
        sys.exit(1)

    # ── Download IP-Adapter FaceID (primary) ──────────────────────────────
    print("\n[2/4] Downloading IP-Adapter FaceID SDXL...")
    print("       (This gives strong face identity preservation)")
    faceid_ok = True
    for filename, url in FACEID_FILES.items():
        if not download(url, FACEID_DIR / filename):
            faceid_ok = False

    # ── Download regular IP-Adapter (fallback) ────────────────────────────
    print("\n[3/4] Downloading regular IP-Adapter SDXL (fallback)...")
    for filename, url in IP_ADAPTER_FILES.items():
        download(url, IP_ADAPTER_DIR / filename)

    # ── Initialize insightface (downloads antelopev2 model) ───────────────
    print("\n[4/4] Setting up insightface face analysis...")
    try:
        from insightface.app import FaceAnalysis
        print("  [..] Downloading antelopev2 model (first time only)...")
        app = FaceAnalysis(
            name="antelopev2",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("  [OK] insightface ready")
    except ImportError:
        print("  [ERR] insightface not installed!")
        print("        Run: pip install insightface onnxruntime")
    except Exception as e:
        print(f"  [WARN] insightface setup issue: {e}")
        print("         Face detection will fall back to OpenCV Haar cascades")

    # ── Check Python packages ─────────────────────────────────────────────
    print("\n[..] Checking Python packages...")
    pkgs = {
        "torch": "torch",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "cv2": "opencv-python",
        "insightface": "insightface",
    }
    missing = []
    for mod, pip_name in pkgs.items():
        try:
            __import__(mod)
            print(f"  [OK] {mod}")
        except ImportError:
            print(f"  [--] {mod} (not installed)")
            missing.append(pip_name)

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install {' '.join(missing)}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if faceid_ok:
        print("  Setup complete!")
        print("  FaceID mode: ENABLED (avatars will look like you)")
    else:
        print("  Setup partially complete.")
        print("  FaceID: FAILED (avatars may not preserve identity)")
        print("  Regular IP-Adapter may still work as fallback.")
    print("\n  Start the kiosk: python start.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
