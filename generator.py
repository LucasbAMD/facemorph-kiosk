"""
generator.py — AMD-Adapt AI Scene Style Transfer
Uses SDXL Turbo img2img on ROCm to transform the entire camera scene
(person + background) into different artistic styles.
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
SDXL_PATH = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
             "sd_xl_turbo_1.0_fp16.safetensors")

# ── Detect model type from filename ───────────────────────────────────────────
_is_turbo = "turbo" in SDXL_PATH.name.lower()

# ── Style definitions ─────────────────────────────────────────────────────────
# Full-scene prompts: describe how the PERSON and ENVIRONMENT should look.
# Strength ~0.80 preserves layout, ~0.90+ for dramatic transformation.

STYLES = {
    "avatar": {
        "label": "Avatar",
        "positive": (
            "scene from alien planet Pandora, "
            "person with deep vivid blue skin and glowing cyan bioluminescent dots, "
            "golden cat-slit eyes, pointed elf ears, tribal bone jewelry, "
            "lush bioluminescent jungle background with towering alien trees, "
            "hanging vines, floating glowing seeds, fireflies in misty air, "
            "cinematic film still, masterpiece"
        ),
        "negative": (
            "human skin, white skin, pink skin, normal skin, office, "
            "blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.88, "guidance": 0.0, "steps": 7},
        "base":   {"strength": 0.82, "guidance": 7.5, "steps": 20},
    },
    "claymation": {
        "label": "Claymation",
        "positive": (
            "entire scene made of plasticine clay and stop-motion miniatures, "
            "clay person with round chunky proportions and thumbprint texture, "
            "miniature clay furniture and props on a tabletop set, "
            "painted cardboard backdrop, warm studio spotlight, "
            "Wallace and Gromit style stop-motion frame, masterpiece"
        ),
        "negative": (
            "photorealistic, real human, CGI, 3D render, "
            "blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.85, "guidance": 0.0, "steps": 6},
        "base":   {"strength": 0.80, "guidance": 8.0, "steps": 20},
    },
    "anime": {
        "label": "Anime",
        "positive": (
            "2D anime illustration of entire scene, "
            "person drawn with thick black ink outlines and flat cel-shaded colors, "
            "huge sparkling anime eyes, colorful stylized hair, "
            "background transformed into anime cityscape at golden hour, "
            "cherry blossom petals drifting through air, "
            "manga style, vibrant saturated colors, masterpiece"
        ),
        "negative": (
            "photorealistic, real person, photograph, 3d render, "
            "realistic skin, blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.87, "guidance": 0.0, "steps": 7},
        "base":   {"strength": 0.82, "guidance": 8.0, "steps": 20},
    },
    "cyberpunk": {
        "label": "Cyberpunk",
        "positive": (
            "cyberpunk scene, person with chrome cybernetic implants and "
            "glowing circuit-pattern tattoos, neon pink and cyan lighting, "
            "environment transformed into rain-soaked neon alley, "
            "holographic advertisements, kanji signs, laser beams, "
            "wet reflective surfaces with puddles, steam rising, "
            "cinematic neon noir, masterpiece"
        ),
        "negative": (
            "natural lighting, daytime, sunny, medieval, fantasy, "
            "blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.85, "guidance": 0.0, "steps": 7},
        "base":   {"strength": 0.80, "guidance": 7.5, "steps": 20},
    },
    "oilpainting": {
        "label": "Oil Painting",
        "positive": (
            "classical oil painting of entire scene, "
            "person painted with rich visible brushstrokes and impasto technique, "
            "warm golden Rembrandt lighting with dramatic chiaroscuro shadows, "
            "background as ornate Renaissance interior with velvet drapes, "
            "gilded picture frames on dark walls, candelabra, "
            "museum masterpiece on canvas, baroque style"
        ),
        "negative": (
            "photograph, digital art, modern, cartoon, anime, "
            "blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.82, "guidance": 0.0, "steps": 6},
        "base":   {"strength": 0.78, "guidance": 7.5, "steps": 20},
    },
    "pixelart": {
        "label": "Pixel Art",
        "positive": (
            "retro 16-bit pixel art scene, "
            "person as pixel art character with blocky square pixels, "
            "limited color palette, dithering shading, "
            "background as retro video game level with pixel clouds and tiles, "
            "8-bit aesthetic, classic SNES RPG style, "
            "crisp pixel edges, nostalgic retro gaming art"
        ),
        "negative": (
            "photorealistic, smooth, high resolution, 3d render, "
            "blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.88, "guidance": 0.0, "steps": 7},
        "base":   {"strength": 0.82, "guidance": 8.0, "steps": 20},
    },
    "comicbook": {
        "label": "Comic Book",
        "positive": (
            "bold comic book illustration of entire scene, "
            "person drawn with heavy black ink outlines and halftone dot shading, "
            "bright primary colors, dramatic action pose, "
            "background with comic panel speed lines and pop art bursts, "
            "Ben-Day dots pattern, speech bubble space, "
            "Marvel comic style, dynamic composition, masterpiece"
        ),
        "negative": (
            "photorealistic, photograph, 3d render, anime, "
            "blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.86, "guidance": 0.0, "steps": 7},
        "base":   {"strength": 0.80, "guidance": 8.0, "steps": 20},
    },
    "steampunk": {
        "label": "Steampunk",
        "positive": (
            "steampunk scene, person wearing brass goggles and Victorian gear, "
            "copper mechanical parts and clockwork accessories, "
            "environment as Victorian workshop with spinning brass gears, "
            "steam pipes, pressure gauges, leather-bound books, "
            "warm amber gaslight illumination, copper and bronze tones, "
            "industrial revolution aesthetic, masterpiece"
        ),
        "negative": (
            "modern, futuristic, neon, digital, cartoon, anime, "
            "blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.84, "guidance": 0.0, "steps": 7},
        "base":   {"strength": 0.78, "guidance": 7.5, "steps": 20},
    },
}

STYLE_ORDER = [
    "avatar", "anime", "cyberpunk", "claymation",
    "oilpainting", "comicbook", "pixelart", "steampunk",
]


# ── Pipeline state ────────────────────────────────────────────────────────────
_pipe = None
_pipe_lock = threading.Lock()
_pipe_ready = False


def _load_pipeline():
    """Load SDXL pipeline. Runs in background thread."""
    global _pipe, _pipe_ready

    try:
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        model_type = "Turbo" if _is_turbo else "Base"
        print(f"[Generator] Loading SDXL {model_type}...")
        if not SDXL_PATH.exists():
            print(f"[Generator] ERROR: Model not found at {SDXL_PATH}")
            return

        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(SDXL_PATH),
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
        _pipe_ready = True
        print(f"[Generator] Ready! model={model_type}")

    except Exception as e:
        print(f"[Generator] ERROR loading pipeline: {e}")
        import traceback
        traceback.print_exc()


def is_ready():
    return _pipe_ready and _pipe is not None


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
    Returns (cv2_image, error_string).
    """
    if not is_ready():
        return None, "AI pipeline loading — please wait a moment"

    style = STYLES.get(style_key)
    if not style:
        return None, f"Unknown style: {style_key}"

    try:
        import torch

        params = style["turbo"] if _is_turbo else style["base"]
        prompt, neg_prompt = _build_prompts(style, gender)

        # Use full frame, resize to 1024x1024 for SDXL
        h, w = frame.shape[:2]
        source_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        source_pil = source_pil.resize((1024, 1024), Image.LANCZOS)

        with _pipe_lock:
            pipe = _pipe

        model_type = "Turbo" if _is_turbo else "Base"
        print(f"[Generator] Style={style_key} Model={model_type} "
              f"Steps={params['steps']} Strength={params['strength']} "
              f"Gender={gender} Frame={w}x{h}")

        start = time.time()
        generator = torch.Generator(device="cuda").manual_seed(
            int(time.time()) % 2**32)

        with torch.inference_mode():
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
        result_cv = cv2.resize(result_cv, (w, h), interpolation=cv2.INTER_LANCZOS4)
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
            print("[Generator] AI engine ready")
        else:
            print("[Generator] AI engine failed to load")

    def check_available(self):
        self.available = is_ready()
        return self.available

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
