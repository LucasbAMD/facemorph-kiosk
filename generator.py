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

# ── Model paths ───────────────────────────────────────────────────────────────
SDXL_TURBO_PATH = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                   "sd_xl_turbo_1.0_fp16.safetensors")

# HuggingFace model IDs (downloaded to cache by setup_models.py)
SDXL_BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0"
DEPTH_ESTIMATOR_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# ── Style definitions ─────────────────────────────────────────────────────────
# Each style has prompts and params for both ControlNet and Turbo modes.
# ControlNet mode uses lower strength since depth map preserves structure.

STYLES = {
    "avatar": {
        "label": "Avatar",
        "positive": (
            "incredible cinematic still from Avatar movie on planet Pandora, "
            "person completely transformed into a Na'vi alien with vivid deep blue skin, "
            "glowing cyan bioluminescent freckles and markings across face and body, "
            "large golden cat-slit eyes, tall pointed elf ears, flat wide nose, "
            "tribal bone necklace and woven leaf clothing, "
            "lush alien jungle background with massive glowing trees, "
            "bioluminescent plants, floating jellyfish seeds, blue misty atmosphere, "
            "James Cameron Avatar film style, ultra detailed, masterpiece"
        ),
        "negative": (
            "human skin, white skin, pink skin, pale skin, normal skin color, "
            "office, indoor room, plain wall, realistic photo, "
            "blurry, deformed, text, watermark"
        ),
        "controlnet": {"strength": 0.92, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.35},
        "turbo":      {"strength": 0.92, "guidance": 0.0, "steps": 7},
    },
    "claymation": {
        "label": "Claymation",
        "positive": (
            "incredible stop-motion claymation scene, "
            "person sculpted entirely from colorful plasticine clay, "
            "round chunky body with visible fingerprint textures in the clay, "
            "oversized head with big clay eyes, smooth clay hair, "
            "miniature clay furniture and props on a detailed tabletop set, "
            "painted cardboard sky backdrop with cotton ball clouds, "
            "warm studio spotlight casting soft shadows, "
            "Aardman animations style, Wallace and Gromit quality, masterpiece"
        ),
        "negative": (
            "photorealistic, real human, real skin, photograph, "
            "blurry, deformed, text, watermark"
        ),
        "controlnet": {"strength": 0.90, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.35},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    "anime": {
        "label": "Anime",
        "positive": (
            "stunning 2D anime illustration, NOT a photograph, "
            "person completely redrawn as anime character with thick black ink outlines, "
            "flat cel-shaded coloring with no realistic shading, "
            "huge sparkling anime eyes with light reflections, "
            "colorful stylized spiky hair, small pointed chin, "
            "background completely replaced with vibrant anime cityscape at sunset, "
            "cherry blossom trees, glowing lanterns, dramatic clouds, "
            "Studio Ghibli quality, manga illustration, vibrant colors, masterpiece"
        ),
        "negative": (
            "photorealistic, real person, photograph, real skin texture, "
            "3d render, realistic lighting, office, plain background, "
            "blurry, deformed, text, watermark"
        ),
        "controlnet": {"strength": 0.93, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.30},
        "turbo":      {"strength": 0.92, "guidance": 0.0, "steps": 7},
    },
    "cyberpunk": {
        "label": "Cyberpunk",
        "positive": (
            "incredible cyberpunk scene at night, "
            "person with glowing neon circuit tattoos across face and arms, "
            "chrome cybernetic eye implant, LED strips in hair, "
            "wearing futuristic jacket with illuminated trim, "
            "background completely transformed into neon-lit rain-soaked city alley, "
            "massive holographic billboards, pink and cyan neon signs, "
            "puddles reflecting neon lights, steam vents, flying vehicles above, "
            "Blade Runner cinematic style, neon noir, masterpiece"
        ),
        "negative": (
            "natural lighting, daytime, sunny, office, plain room, "
            "blurry, deformed, text, watermark"
        ),
        "controlnet": {"strength": 0.90, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.35},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    "oilpainting": {
        "label": "Oil Painting",
        "positive": (
            "magnificent classical oil painting on canvas, "
            "person painted with rich thick visible brushstrokes and heavy impasto, "
            "warm golden Rembrandt lighting with dramatic chiaroscuro, "
            "deep shadows and glowing highlights on skin, "
            "background transformed into grand Renaissance palace interior, "
            "rich velvet curtains, ornate gold frames, marble columns, "
            "candlelight flickering, old master museum painting, "
            "baroque masterpiece, gallery quality artwork"
        ),
        "negative": (
            "photograph, digital art, modern, plain background, office, "
            "blurry, deformed, text, watermark"
        ),
        "controlnet": {"strength": 0.88, "guidance": 10.0, "steps": 35,
                        "controlnet_scale": 0.35},
        "turbo":      {"strength": 0.88, "guidance": 0.0, "steps": 7},
    },
    "pixelart": {
        "label": "Pixel Art",
        "positive": (
            "retro 16-bit pixel art video game screenshot, "
            "person as chunky pixel art character with visible square pixels, "
            "limited retro color palette with dithering patterns, "
            "background as colorful retro RPG game level, "
            "pixel art trees, 8-bit clouds, tiled ground, "
            "HUD elements and health bar at top, "
            "classic SNES Final Fantasy style, nostalgic pixel art, masterpiece"
        ),
        "negative": (
            "photorealistic, smooth, high resolution, photograph, "
            "blurry, deformed, text, watermark"
        ),
        "controlnet": {"strength": 0.93, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.30},
        "turbo":      {"strength": 0.92, "guidance": 0.0, "steps": 7},
    },
    "comicbook": {
        "label": "Comic Book",
        "positive": (
            "bold dynamic comic book illustration, "
            "person drawn with heavy black ink outlines and halftone dot shading, "
            "bright saturated primary colors, dramatic heroic pose, "
            "background with dramatic speed lines, POW burst effects, "
            "Ben-Day dots pattern everywhere, bold color blocks, "
            "action comic panel layout, speech bubble in corner, "
            "Jack Kirby Marvel comic art style, dynamic composition, masterpiece"
        ),
        "negative": (
            "photorealistic, photograph, real skin, plain background, "
            "blurry, deformed, text, watermark"
        ),
        "controlnet": {"strength": 0.92, "guidance": 12.0, "steps": 35,
                        "controlnet_scale": 0.30},
        "turbo":      {"strength": 0.90, "guidance": 0.0, "steps": 7},
    },
    "steampunk": {
        "label": "Steampunk",
        "positive": (
            "incredible steampunk scene, "
            "person wearing ornate brass goggles on forehead, "
            "Victorian leather coat with copper gears and clockwork accessories, "
            "mechanical arm with visible brass pistons, "
            "background transformed into grand Victorian workshop, "
            "massive spinning brass gears on walls, steam pipes everywhere, "
            "pressure gauges, leather-bound books, amber gaslight glow, "
            "copper and bronze color palette, warm dramatic lighting, "
            "industrial revolution aesthetic, masterpiece"
        ),
        "negative": (
            "modern, futuristic, neon, office, plain room, "
            "blurry, deformed, text, watermark"
        ),
        "controlnet": {"strength": 0.90, "guidance": 11.0, "steps": 35,
                        "controlnet_scale": 0.35},
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
def _build_prompts(style, gender):
    """Build gender-aware positive and negative prompts."""
    prompt = style["positive"]
    neg = style["negative"]
    if gender == "male":
        prompt = f"male man, {prompt}"
        neg = f"female, woman, {neg}"
    elif gender == "female":
        prompt = f"female woman, {prompt}"
        neg = f"male, man, beard, {neg}"
    return prompt, neg


def generate_scene(frame, style_key, gender="unknown"):
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
        prompt, neg_prompt = _build_prompts(style, gender)

        # Prepare source image at 1024x1024 for SDXL
        h, w = frame.shape[:2]
        source_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        source_pil = source_pil.resize((1024, 1024), Image.LANCZOS)

        with _pipe_lock:
            pipe = _pipe

        print(f"[Generator] Style={style_key} Mode={mode} "
              f"Steps={params['steps']} Strength={params['strength']} "
              f"Gender={gender} Frame={w}x{h}")

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
                    negative_prompt=neg_prompt,
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
                    negative_prompt=neg_prompt,
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

    def generate(self, frame, style_key, gender="unknown"):
        if self._status == "generating":
            return False
        self._status = "generating"
        self._result = None
        self._message = "Transforming scene..."
        self._thread = threading.Thread(
            target=self._run,
            args=(frame.copy(), style_key, gender),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, frame, style_key, gender="unknown"):
        img, err = generate_scene(frame, style_key, gender=gender)
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
