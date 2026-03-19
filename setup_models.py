#!/usr/bin/env python3
"""
setup_models.py — Download all model weights for the AI Scene Style Kiosk.
Run this once before starting the kiosk for the first time.

Downloads:
  1. SDXL Base 1.0 — high-quality diffusion model
  2. ControlNet Depth SDXL — preserves scene structure during style transfer
  3. DPT depth estimator — extracts depth maps from camera frames

Usage:
    python setup_models.py
"""
import os
import sys
from pathlib import Path

MODELS_DIR = Path.home() / "kiosk_models"


def check_package(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def main():
    print("\n" + "=" * 55)
    print("  AI Scene Style Kiosk — Model Setup")
    print("=" * 55)

    # ── Check required packages ────────────────────────────────────────────
    print("\n[1/8] Checking Python packages...")
    required = {
        "torch": "torch (with ROCm)",
        "diffusers": "diffusers>=0.27.0",
        "transformers": "transformers>=4.36.0",
        "accelerate": "accelerate>=0.25.0",
        "cv2": "opencv-python",
    }
    missing = []
    for mod, desc in required.items():
        if check_package(mod):
            print(f"  [OK] {desc}")
        else:
            print(f"  [--] {desc} NOT INSTALLED")
            missing.append(mod)

    if "torch" in missing or "diffusers" in missing:
        print("\n  [ERR] Core packages missing. Install them first:")
        print("        pip install -r requirements.txt")
        sys.exit(1)

    import torch
    print(f"\n  PyTorch: {torch.__version__}")
    print(f"  CUDA/ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM: {vram:.1f} GB")

    # ── Check/download SDXL Turbo (existing) ──────────────────────────────
    print("\n[2/8] Checking SDXL Turbo model...")
    sdxl_turbo = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                  "sd_xl_turbo_1.0_fp16.safetensors")
    if sdxl_turbo.exists():
        size_mb = sdxl_turbo.stat().st_size // (1024 * 1024)
        print(f"  [OK] SDXL Turbo found ({size_mb} MB)")
    else:
        print(f"  [WARN] SDXL Turbo not found at {sdxl_turbo}")
        print("         Turbo fallback mode will not be available.")

    # ── Download ControlNet Depth SDXL ────────────────────────────────────
    print("\n[3/8] Downloading ControlNet Depth for SDXL...")
    print("       (This preserves your pose and scene structure)")
    print("       Model: diffusers/controlnet-depth-sdxl-1.0")
    try:
        from diffusers import ControlNetModel
        controlnet_id = "diffusers/controlnet-depth-sdxl-1.0"
        print(f"  [..] Downloading {controlnet_id}...")
        ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        print(f"  [OK] ControlNet Depth SDXL cached")
    except Exception as e:
        print(f"  [WARN] Could not download ControlNet: {e}")
        print("         Will fall back to Turbo-only mode.")

    # ── Download Depth-Anything-V2 depth estimator ─────────────────────────
    print("\n[4/8] Downloading depth estimator (Depth-Anything-V2)...")
    print("       Model: depth-anything/Depth-Anything-V2-Small-hf")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        depth_id = "depth-anything/Depth-Anything-V2-Small-hf"
        print(f"  [..] Downloading {depth_id}...")
        AutoImageProcessor.from_pretrained(depth_id)
        AutoModelForDepthEstimation.from_pretrained(depth_id)
        print(f"  [OK] Depth-Anything-V2 cached")
    except Exception as e:
        print(f"  [WARN] Could not download depth estimator: {e}")
        print("         Will fall back to Turbo-only mode.")

    # ── Download IP-Adapter FaceID for SDXL ─────────────────────────────
    # ── Download Real-ESRGAN 2x upscaler ─────────────────────────────
    print("\n[5/8] Downloading Real-ESRGAN 2x upscaler...")
    print("       (Upscales output from 1024 to 2048 for sharper results)")
    models_dir = Path.home() / "kiosk_models"
    models_dir.mkdir(exist_ok=True)
    esrgan_path = models_dir / "RealESRGAN_x2plus.pth"
    if esrgan_path.exists():
        size_mb = esrgan_path.stat().st_size // (1024 * 1024)
        print(f"  [OK] RealESRGAN_x2plus.pth already exists ({size_mb} MB)")
    else:
        try:
            import urllib.request
            url = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                   "v0.2.1/RealESRGAN_x2plus.pth")
            print(f"  [..] Downloading from GitHub releases...")
            urllib.request.urlretrieve(url, str(esrgan_path))
            size_mb = esrgan_path.stat().st_size // (1024 * 1024)
            print(f"  [OK] RealESRGAN_x2plus.pth downloaded ({size_mb} MB)")
        except Exception as e:
            print(f"  [WARN] Could not download Real-ESRGAN: {e}")
            print("         Output will not be upscaled (still works, just lower res).")

    print("\n[6/8] Downloading IP-Adapter FaceID for SDXL...")
    print("       (This preserves your face identity during avatar style)")
    print("       Model: h94/IP-Adapter-FaceID")
    try:
        from huggingface_hub import hf_hub_download
        print("  [..] Downloading ip-adapter-faceid_sdxl.bin...")
        hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID",
            filename="ip-adapter-faceid_sdxl.bin",
        )
        print("  [OK] IP-Adapter FaceID SDXL cached")
    except Exception as e:
        print(f"  [WARN] Could not download IP-Adapter FaceID: {e}")
        print("         Avatar style will work without face ID preservation.")

    # ── Download InsightFace buffalo_l model ───────────────────────────
    print("\n[7/8] Checking InsightFace face analysis model...")
    try:
        from insightface.app import FaceAnalysis
        print("  [..] Downloading buffalo_l model (if needed)...")
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(160, 160))
        print("  [OK] InsightFace buffalo_l cached")
    except ImportError:
        print("  [WARN] InsightFace not installed.")
        print("         Install with: pip install insightface onnxruntime-gpu")
    except Exception as e:
        print(f"  [WARN] Could not download InsightFace model: {e}")

    # ── Download Juggernaut XL v9 base model ─────────────────────────────
    print("\n[Bonus] Pre-caching Juggernaut XL v9 pipeline components...")
    print("        (Much higher quality than vanilla SDXL Base)")
    print("        This may take a while on first run (~6.5 GB).")
    try:
        from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
        base_id = "RunDiffusion/Juggernaut-XL-v9"
        print(f"  [..] Downloading {base_id}...")
        try:
            StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                base_id,
                torch_dtype=torch.float16,
                variant="fp16",
                controlnet=ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                ),
            )
        except OSError:
            StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                base_id,
                torch_dtype=torch.float16,
                controlnet=ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                ),
            )
        print("  [OK] Juggernaut XL v9 + ControlNet pipeline cached")
    except Exception as e:
        print(f"  [WARN] Could not pre-cache full pipeline: {e}")
        print("         It will download on first start instead.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Setup complete!")
    print()
    print("  Modes available:")
    print("    - ControlNet + Juggernaut XL v9: Best quality (~25-40s)")
    print("    - SDXL Turbo fallback:           Fast (~5s)")
    print()
    print("  Start the kiosk: python start.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
