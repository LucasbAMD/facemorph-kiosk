"""
generator.py — AMD-Adapt AI Scene Style Transfer

Best quality mode: SDXL Base + ControlNet Depth
  - Extracts depth map from camera frame to preserve structure
  - Uses SDXL Base (not Turbo) for highest quality output
  - ~20-30s per generation but dramatically better results

Fallback mode: SDXL Turbo img2img
  - Fast (~5s) but lower quality
  - Used when ControlNet models are not available
"""

import cv2
import numpy as np
import threading
import time
import warnings
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore", message=".*upcast_vae.*")
warnings.filterwarnings("ignore", message=".*enable_vae_slicing.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*The following part of your input was truncated.*")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ── Model paths ───────────────────────────────────────────────────────────────
SDXL_TURBO_PATH = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                   "sd_xl_turbo_1.0_fp16.safetensors")

# HuggingFace model IDs (downloaded to cache by setup_models.py)
SDXL_BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0"
DEPTH_ESTIMATOR_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# ── Common negative terms — split across dual CLIP encoders ───────────────────
# neg1 = identity/body issues (CLIP-L, <=77 tokens)
# neg2 = quality/content issues (CLIP-G, <=77 tokens)
_NEG_IDENTITY = (
    "beard, mustache, facial hair, goatee, stubble, "
    "mask, helmet, face covering, face paint, "
    "different face, different person, different race, different gender, "
    "nudity, exposed chest, bare chest, cleavage, topless, nsfw"
)
_NEG_QUALITY = (
    "deformed hands, extra fingers, fused fingers, missing fingers, "
    "bad anatomy, blurry, low quality, lowres, "
    "text, watermark, deformed, ugly"
)

# ── Style definitions ─────────────────────────────────────────────────────────
# Each style uses dual SDXL prompts to stay within 77-token CLIP limits:
#   prompt   = subject/identity (CLIP-L, 77 tokens)
#   prompt_2 = scene/style/background (CLIP-G, 77 tokens)
# controlnet_scale ~0.50-0.58 preserves face/pose/layout from depth map.
# strength ~0.78-0.85 controls how much the image deviates from original.

STYLES = {
    "avatar": {
        "label": "Avatar",
        "prompt": (
            "Na'vi alien from Avatar movie, same person's face shape and expression, "
            "entire body covered in vivid deep blue skin, glowing cyan bioluminescent "
            "freckles and dots across face and arms, golden cat-slit eyes, "
            "tall pointed elf ears, full coverage warrior armor with leather chest plate"
        ),
        "prompt_2": (
            "cinematic still from James Cameron Avatar on planet Pandora, "
            "lush alien bioluminescent jungle, massive glowing trees, "
            "floating jellyfish seeds, blue misty atmosphere, "
            "movie quality, ultra detailed, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", human skin, pink skin, white skin, normal skin color, pale",
        "negative_2": _NEG_QUALITY + ", realistic photo, office, indoor room, plain wall",
        "controlnet": {"strength": 0.90, "guidance": 14.0, "steps": 35,
                        "controlnet_scale": 0.38},
        "turbo":      {"strength": 0.92, "guidance": 0.0, "steps": 7},
    },
    "claymation": {
        "label": "Claymation",
        "prompt": (
            "same person's identical face sculpted in smooth clay, "
            "preserving exact jaw nose eyes and expression, clean-shaven, "
            "round chunky clay body, visible fingerprint textures, "
            "big expressive clay eyes, smooth clay hair matching their hairstyle"
        ),
        "prompt_2": (
            "incredible stop-motion claymation scene, "
            "miniature clay furniture and props on detailed tabletop set, "
            "painted cardboard sky backdrop, cotton ball clouds, "
            "warm studio spotlight, soft shadows, "
            "Aardman animations, Wallace and Gromit quality, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", photorealistic, real human, real skin, photograph",
        "negative_2": _NEG_QUALITY + ", office, plain wall",
        "controlnet": {"strength": 0.82, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.52},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "anime": {
        "label": "Anime",
        "prompt": (
            "same person redrawn as anime character, identical face shape and features, "
            "same hairstyle same hair color same expression, "
            "thick black ink outlines, flat cel-shaded coloring, "
            "sparkling anime eyes with light reflections"
        ),
        "prompt_2": (
            "stunning 2D anime illustration, NOT a photograph, "
            "vibrant anime cityscape at sunset background, "
            "cherry blossom trees, glowing lanterns, dramatic clouds, "
            "Studio Ghibli quality, manga illustration, vibrant colors, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", photorealistic, real person, photograph, real skin texture",
        "negative_2": _NEG_QUALITY + ", 3d render, realistic lighting, office, plain background",
        "controlnet": {"strength": 0.83, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.50},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    "cyberpunk": {
        "label": "Cyberpunk",
        "prompt": (
            "same person identical face preserved exactly, same eyes nose jaw, "
            "subtle neon circuit tattoo lines on cheeks, "
            "chrome cybernetic accent near temple, LED strips in hair, "
            "futuristic jacket with illuminated trim"
        ),
        "prompt_2": (
            "incredible cyberpunk scene at night, "
            "neon-lit rain-soaked city alley, massive holographic billboards, "
            "pink and cyan neon signs, puddles reflecting neon, "
            "steam vents, flying vehicles, "
            "Blade Runner cinematic style, neon noir, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", natural lighting, daytime, sunny",
        "negative_2": _NEG_QUALITY + ", office, plain room",
        "controlnet": {"strength": 0.80, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.53},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "oilpainting": {
        "label": "Oil Painting",
        "prompt": (
            "same person's identical face painted with visible brushstrokes, "
            "preserving exact facial features eyes nose mouth and expression, "
            "rich thick impasto technique, warm golden Rembrandt lighting, "
            "dramatic chiaroscuro shadows and glowing highlights"
        ),
        "prompt_2": (
            "magnificent classical oil painting on canvas, "
            "grand Renaissance palace interior background, "
            "rich velvet curtains, ornate gold frames, marble columns, "
            "candlelight flickering, old master museum painting, "
            "baroque masterpiece, gallery quality artwork"
        ),
        "negative": _NEG_IDENTITY + ", photograph, digital art, modern",
        "negative_2": _NEG_QUALITY + ", plain background, office",
        "controlnet": {"strength": 0.80, "guidance": 10.0, "steps": 35,
                        "controlnet_scale": 0.55},
        "turbo":      {"strength": 0.86, "guidance": 0.0, "steps": 7},
    },
    "pixelart": {
        "label": "Pixel Art",
        "prompt": (
            "same person as pixel art character, identical face shape and features, "
            "same hairstyle and hair color, "
            "visible square pixels, limited retro color palette, dithering"
        ),
        "prompt_2": (
            "retro 16-bit pixel art video game screenshot, "
            "colorful RPG game level background, pixel art trees, 8-bit clouds, "
            "tiled ground, HUD elements and health bar, "
            "classic SNES Final Fantasy style, nostalgic pixel art, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", photorealistic, smooth, high resolution, photograph",
        "negative_2": _NEG_QUALITY,
        "controlnet": {"strength": 0.83, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.50},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    "comicbook": {
        "label": "Comic Book",
        "prompt": (
            "same person's identical face drawn with ink, no mask, face fully visible, "
            "preserving exact eyes nose mouth jaw and expression, "
            "heavy black ink outlines, halftone dot shading on skin, "
            "bright saturated primary colors"
        ),
        "prompt_2": (
            "bold dynamic comic book illustration, "
            "dramatic speed lines and pop art bursts background, "
            "Ben-Day dots pattern, bold color blocks, "
            "action comic panel layout, "
            "classic comic art style, dynamic composition, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", superhero mask, domino mask, eye mask, face covered",
        "negative_2": _NEG_QUALITY + ", photorealistic, photograph, plain background",
        "controlnet": {"strength": 0.80, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.55},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "steampunk": {
        "label": "Steampunk",
        "prompt": (
            "same person identical face preserved exactly, same eyes nose jaw mouth, "
            "clean-shaven face matching the real person precisely, "
            "ornate brass goggles pushed up on forehead, "
            "Victorian leather coat with copper gears and clockwork accessories"
        ),
        "prompt_2": (
            "incredible steampunk scene, "
            "grand Victorian workshop background, "
            "massive spinning brass gears, steam pipes, pressure gauges, "
            "leather-bound books, amber gaslight glow, "
            "copper and bronze palette, warm dramatic lighting, "
            "industrial revolution aesthetic, masterpiece"
        ),
        "negative": _NEG_IDENTITY + ", added beard, added facial hair, rugged face",
        "negative_2": _NEG_QUALITY + ", modern, futuristic, neon, office",
        "controlnet": {"strength": 0.78, "guidance": 10.0, "steps": 35,
                        "controlnet_scale": 0.58},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
}

STYLE_ORDER = [
    "avatar", "anime", "cyberpunk", "claymation",
    "oilpainting", "comicbook", "pixelart", "steampunk",
]


# ── Pipeline state ────────────────────────────────────────────────────────────
_pipe = None
_depth_estimator = None
_depth_processor = None
_pipe_lock = threading.Lock()
_pipe_ready = False
_pipe_mode = "none"  # "controlnet" or "turbo"


def _load_pipeline():
    """Load best available pipeline. Tries ControlNet first, falls back to Turbo."""
    global _pipe, _pipe_ready, _pipe_mode
    global _depth_estimator, _depth_processor

    try:
        import torch

        # ── Try ControlNet + SDXL Base (best quality) ─────────────────────
        try:
            from diffusers import (
                StableDiffusionXLControlNetImg2ImgPipeline,
                ControlNetModel,
            )
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            print("[Generator] Loading ControlNet Depth + SDXL Base...")
            print("[Generator]   Loading depth estimator (Depth-Anything-V2)...")
            depth_proc = AutoImageProcessor.from_pretrained(DEPTH_ESTIMATOR_ID)
            depth_model = AutoModelForDepthEstimation.from_pretrained(
                DEPTH_ESTIMATOR_ID)
            depth_model = depth_model.to("cuda")
            depth_model.eval()

            print("[Generator]   Loading ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_DEPTH_ID,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            print("[Generator]   Loading SDXL Base pipeline...")
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                SDXL_BASE_ID,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe = pipe.to("cuda")

            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

            with _pipe_lock:
                _pipe = pipe
                _depth_estimator = depth_model
                _depth_processor = depth_proc
            _pipe_mode = "controlnet"
            _pipe_ready = True
            print("[Generator] Ready! mode=ControlNet+SDXL_Base (best quality)")
            return

        except Exception as e:
            print(f"[Generator] ControlNet not available: {e}")
            print("[Generator] Falling back to SDXL Turbo...")

        # ── Fallback: SDXL Turbo ──────────────────────────────────────────
        if not SDXL_TURBO_PATH.exists():
            print(f"[Generator] ERROR: No models found!")
            print(f"[Generator]   Run: python setup_models.py")
            return

        from diffusers import StableDiffusionXLImg2ImgPipeline

        print(f"[Generator] Loading SDXL Turbo from {SDXL_TURBO_PATH}...")
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(SDXL_TURBO_PATH),
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")

        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        with _pipe_lock:
            _pipe = pipe
        _pipe_mode = "turbo"
        _pipe_ready = True
        print("[Generator] Ready! mode=SDXL_Turbo (fast)")

    except Exception as e:
        print(f"[Generator] ERROR loading pipeline: {e}")
        import traceback
        traceback.print_exc()


def is_ready():
    return _pipe_ready and _pipe is not None


def get_mode():
    return _pipe_mode


# ── Depth estimation ──────────────────────────────────────────────────────────
def _estimate_depth(pil_image):
    """Extract depth map from image using Depth-Anything-V2."""
    import torch

    with _pipe_lock:
        proc = _depth_processor
        model = _depth_estimator

    inputs = proc(images=pil_image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        depth = outputs.predicted_depth

    # Normalize to 0-255
    depth = depth.squeeze().float()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_np = depth.cpu().numpy()
    depth_np = (depth_np * 255).astype(np.uint8)

    # Resize to match input image size
    w, h = pil_image.size
    depth_resized = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Convert to 3-channel RGB PIL image for ControlNet
    depth_3ch = np.stack([depth_resized] * 3, axis=-1)
    return Image.fromarray(depth_3ch)


# ── Generation ────────────────────────────────────────────────────────────────
def generate_scene(frame, style_key):
    """
    Transform entire camera frame into the given style.
    Uses ControlNet+depth if available, otherwise Turbo img2img.
    Returns (cv2_image, error_string).
    """
    if not is_ready():
        return None, "AI pipeline loading — please wait a moment"

    style = STYLES.get(style_key)
    if not style:
        return None, f"Unknown style: {style_key}"

    try:
        import torch

        mode = _pipe_mode
        params = style.get(mode, style["turbo"])
        prompt = style["prompt"]
        prompt_2 = style.get("prompt_2", prompt)
        neg_prompt = style["negative"]
        neg_prompt_2 = style.get("negative_2", neg_prompt)

        # Prepare source image at 1024x1024 for SDXL
        h, w = frame.shape[:2]
        source_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        source_pil = source_pil.resize((1024, 1024), Image.LANCZOS)

        with _pipe_lock:
            pipe = _pipe

        print(f"[Generator] Style={style_key} Mode={mode} "
              f"Steps={params['steps']} Strength={params['strength']} "
              f"Frame={w}x{h}")

        start = time.time()
        generator = torch.Generator(device="cuda").manual_seed(
            int(time.time()) % 2**32)

        with torch.inference_mode():
            if mode == "controlnet":
                # Extract depth map for structural preservation
                print("[Generator]   Estimating depth...")
                depth_image = _estimate_depth(source_pil)

                result = pipe(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=neg_prompt,
                    negative_prompt_2=neg_prompt_2,
                    image=source_pil,
                    control_image=depth_image,
                    strength=params["strength"],
                    num_inference_steps=params["steps"],
                    guidance_scale=params["guidance"],
                    controlnet_conditioning_scale=params["controlnet_scale"],
                    generator=generator,
                ).images[0]
            else:
                # Turbo fallback
                result = pipe(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=neg_prompt,
                    negative_prompt_2=neg_prompt_2,
                    image=source_pil,
                    strength=params["strength"],
                    num_inference_steps=params["steps"],
                    guidance_scale=params["guidance"],
                    generator=generator,
                ).images[0]

        elapsed = time.time() - start
        print(f"[Generator] Done in {elapsed:.1f}s")

        # Resize back to original aspect ratio
        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        result_cv = cv2.resize(result_cv, (w, h),
                               interpolation=cv2.INTER_LANCZOS4)
        return result_cv, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Generation failed: {e}"


# ── ComfyBridge-compatible wrapper ────────────────────────────────────────────
class ComfyBridge:
    """Drop-in replacement maintaining the same interface as main.py expects."""

    def __init__(self):
        self._result = None
        self._status = "idle"
        self._message = ""
        self._thread = None
        self.available = False
        threading.Thread(target=self._init_pipeline, daemon=True).start()

    def _init_pipeline(self):
        _load_pipeline()
        self.available = is_ready()
        if self.available:
            print(f"[Generator] AI engine ready (mode={get_mode()})")
        else:
            print("[Generator] AI engine failed to load")

    def check_available(self):
        self.available = is_ready()
        return self.available

    def get_mode(self):
        return get_mode()

    def generate(self, frame, style_key):
        if self._status == "generating":
            return False
        self._status = "generating"
        self._result = None
        self._message = "Transforming scene..."
        self._thread = threading.Thread(
            target=self._run,
            args=(frame.copy(), style_key),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, frame, style_key):
        img, err = generate_scene(frame, style_key)
        if img is not None:
            self._result = img
            self._status = "done"
            self._message = "Done!"
        else:
            self._status = "error"
            self._message = err or "Generation failed"

    def get_status(self):
        return {"status": self._status, "message": self._message}

    def get_result(self):
        if self._result is not None:
            img = self._result.copy()
            self._status = "idle"
            self._result = None
            return img
        return None

    def reset(self):
        self._status = "idle"
        self._result = None
        self._message = ""
