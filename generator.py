"""
generator.py — AI Avatar Style Generator
Uses SDXL + IP-Adapter on ROCm for high-quality stylized avatar generation.

Pipeline:
  1. Extracts face from webcam frame
  2. Uses IP-Adapter to condition on the person's actual face
  3. Generates a fully stylized avatar (not just a filter)
  4. Styles: Avatar (blue alien), Claymation, Anime, Ghost

Key improvements over previous version:
  - 20 inference steps (was 4) for actual quality
  - guidance_scale 7.5 (was 0.0) for prompt adherence
  - High denoise (0.80-0.90) to fully restyle, not just tint
  - IP-Adapter enabled for face identity preservation
  - Face-centered crop as input instead of full body frame
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from PIL import Image

# ── Model paths ───────────────────────────────────────────────────────────────
SDXL_PATH = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
             "sd_xl_turbo_1.0_fp16.safetensors")

IP_ADAPTER_DIR = Path.home() / "kiosk_models" / "ip_adapter"
IP_ADAPTER_BIN = IP_ADAPTER_DIR / "sdxl_models" / "ip-adapter_sdxl.bin"
IMAGE_ENCODER  = IP_ADAPTER_DIR / "models" / "image_encoder"

# ── Style definitions ─────────────────────────────────────────────────────────
STYLES = {
    "avatar": {
        "label": "Avatar",
        "positive": (
            "cinematic portrait of a Na'vi alien from Avatar movie, "
            "deep blue skin with cyan bioluminescent freckle patterns, "
            "large luminous amber-gold eyes with cat-like pupils, "
            "pointed elongated ears, flat wide nose, "
            "braided dark hair with glowing beads, "
            "lush bioluminescent Pandora jungle background, "
            "volumetric fog, firefly particles, "
            "masterpiece, ultra detailed, photorealistic, 8k, cinematic lighting"
        ),
        "negative": (
            "human skin, pink skin, normal ears, small eyes, "
            "cartoon, anime, painting, sketch, drawing, "
            "blurry, low quality, deformed, ugly, mutation, extra limbs, "
            "text, watermark, signature, multiple people"
        ),
        "strength": 0.82,
        "guidance": 7.5,
        "steps": 20,
        "ip_weight": 0.5,
    },
    "claymation": {
        "label": "Claymation",
        "positive": (
            "adorable claymation character figurine, "
            "smooth matte plasticine clay texture, round soft features, "
            "chunky proportions, visible fingerprint marks on clay, "
            "Aardman Animations style like Wallace and Gromit, "
            "soft studio lighting, miniature set background, "
            "stop-motion animation still, handcrafted look, "
            "masterpiece, highly detailed, professional photography of clay figure"
        ),
        "negative": (
            "photorealistic skin, real human, CGI, 3D render, smooth plastic, "
            "blurry, low quality, deformed, ugly, "
            "text, watermark, multiple people, anime, cartoon drawing"
        ),
        "strength": 0.85,
        "guidance": 8.0,
        "steps": 20,
        "ip_weight": 0.55,
    },
    "anime": {
        "label": "Anime",
        "positive": (
            "stunning anime character portrait, "
            "clean sharp cel-shaded art style, vibrant saturated colors, "
            "large expressive detailed eyes with light reflections, "
            "smooth flawless anime skin, defined jawline, "
            "stylized colorful hair, dynamic lighting, "
            "soft pastel bokeh background with cherry blossoms, "
            "Studio Ghibli meets Makoto Shinkai style, "
            "masterpiece, best quality, ultra detailed anime illustration"
        ),
        "negative": (
            "photorealistic, photograph, 3d render, CGI, "
            "blurry, low quality, bad anatomy, deformed eyes, extra fingers, "
            "western cartoon, chibi, sketch, rough lines, "
            "text, watermark, multiple people, ugly"
        ),
        "strength": 0.88,
        "guidance": 8.0,
        "steps": 20,
        "ip_weight": 0.45,
    },
    "ghost": {
        "label": "Ghost",
        "positive": (
            "ethereal spectral ghost portrait, "
            "translucent pale glowing skin with blue-white luminescence, "
            "wispy transparent edges dissolving into smoke and mist, "
            "hollow glowing eyes with otherworldly light, "
            "flowing ghostly hair floating weightlessly, "
            "dark haunted atmosphere with fog and moonlight, "
            "spectral aura and particle effects, ectoplasm wisps, "
            "masterpiece, ultra detailed, cinematic horror lighting, 8k"
        ),
        "negative": (
            "solid opaque body, normal skin color, warm colors, "
            "happy, bright, sunny, cartoon, anime, "
            "blurry, low quality, deformed, ugly, "
            "text, watermark, multiple people, costume, mask"
        ),
        "strength": 0.80,
        "guidance": 7.5,
        "steps": 20,
        "ip_weight": 0.5,
    },
}


# ── Pipeline state ────────────────────────────────────────────────────────────
_pipe = None
_pipe_lock = threading.Lock()
_pipe_ready = False
_has_ip_adapter = False


def _load_pipeline():
    """Load SDXL + optional IP-Adapter. Runs in background thread."""
    global _pipe, _pipe_ready, _has_ip_adapter

    try:
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        print("[Generator] Loading SDXL pipeline...")
        print(f"[Generator] Model: {SDXL_PATH}")

        if not SDXL_PATH.exists():
            print(f"[Generator] ERROR: Model not found at {SDXL_PATH}")
            return

        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(SDXL_PATH),
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")

        # Try loading IP-Adapter for face identity preservation
        if IP_ADAPTER_BIN.exists() and IMAGE_ENCODER.exists():
            try:
                pipe.load_ip_adapter(
                    str(IP_ADAPTER_DIR),
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.bin",
                    image_encoder_folder=str(IMAGE_ENCODER),
                )
                _has_ip_adapter = True
                print("[Generator] IP-Adapter loaded - face identity preservation ON")
            except Exception as e:
                print(f"[Generator] IP-Adapter failed ({e}) - using img2img only")
                _has_ip_adapter = False
        else:
            print("[Generator] IP-Adapter not found - using img2img only")
            if not IP_ADAPTER_BIN.exists():
                print(f"  Missing: {IP_ADAPTER_BIN}")
            if not IMAGE_ENCODER.exists():
                print(f"  Missing: {IMAGE_ENCODER}")

        # Enable memory optimizations for AMD
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        with _pipe_lock:
            _pipe = pipe
        _pipe_ready = True
        print("[Generator] Pipeline ready!")

    except Exception as e:
        print(f"[Generator] ERROR loading pipeline: {e}")
        import traceback
        traceback.print_exc()


def is_ready():
    return _pipe_ready and _pipe is not None


# ── Face extraction ───────────────────────────────────────────────────────────
_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _face_cascade


def extract_face_crop(frame, padding_ratio=0.65):
    """
    Extract the largest face from frame with generous padding.
    Returns a square-ish PIL Image centered on the face.
    This becomes the source image for img2img, so the AI
    focuses on restyling the face rather than the background.
    """
    detector = _get_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        # Pick largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

        # Add padding to include hair, ears, neck
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        fh, fw = frame.shape[:2]

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)

        # Make it roughly square for better generation
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w > crop_h:
            diff = crop_w - crop_h
            y1 = max(0, y1 - diff // 2)
            y2 = min(fh, y2 + diff // 2)
        elif crop_h > crop_w:
            diff = crop_h - crop_w
            x1 = max(0, x1 - diff // 2)
            x2 = min(fw, x2 + diff // 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    # Fallback: center crop of frame
    h, w = frame.shape[:2]
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    crop = frame[y1:y1+size, x1:x1+size]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def extract_face_only(frame):
    """Extract a tight face crop for IP-Adapter conditioning."""
    detector = _get_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(max(w, h) * 0.25)
        fh, fw = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ── Generation ────────────────────────────────────────────────────────────────
def generate_avatar(frame, style_key):
    """
    Generate a stylized avatar from a webcam frame.

    Args:
        frame:     BGR webcam frame
        style_key: One of 'avatar', 'claymation', 'anime', 'ghost'

    Returns:
        (bgr_image, None) on success, (None, error_string) on failure
    """
    if not is_ready():
        return None, "AI pipeline loading - please wait a moment"

    style = STYLES.get(style_key)
    if not style:
        return None, f"Unknown style: {style_key}"

    try:
        import torch

        # Extract face-centered crop as source image
        source_pil = extract_face_crop(frame)

        # Resize to 1024x1024 for SDXL
        source_pil = source_pil.resize((1024, 1024), Image.LANCZOS)

        # Extract tight face crop for IP-Adapter
        face_pil = None
        if _has_ip_adapter:
            face_pil = extract_face_only(frame)
            face_pil = face_pil.resize((224, 224), Image.LANCZOS)

        with _pipe_lock:
            pipe = _pipe

        print(f"[Generator] Generating {style_key} avatar...")
        print(f"[Generator] Steps={style['steps']} Strength={style['strength']} "
              f"Guidance={style['guidance']} IP-Adapter={_has_ip_adapter}")
        start = time.time()

        generator = torch.Generator(device="cuda").manual_seed(
            int(time.time()) % 2**32)

        # Set IP-Adapter scale if available
        if _has_ip_adapter and face_pil is not None:
            try:
                pipe.set_ip_adapter_scale(style["ip_weight"])
            except Exception:
                pass

        # Build generation kwargs
        gen_kwargs = {
            "prompt": style["positive"],
            "negative_prompt": style["negative"],
            "image": source_pil,
            "strength": style["strength"],
            "num_inference_steps": style["steps"],
            "guidance_scale": style["guidance"],
            "generator": generator,
        }

        # Add IP-Adapter image if available
        if _has_ip_adapter and face_pil is not None:
            gen_kwargs["ip_adapter_image"] = face_pil

        with torch.inference_mode():
            result = pipe(**gen_kwargs).images[0]

        elapsed = time.time() - start
        print(f"[Generator] Done in {elapsed:.1f}s")

        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return result_bgr, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)


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

    def generate(self, frame, style_key, selection_mask=None, face_boxes=None):
        if self._status == "generating":
            return False
        self._status = "generating"
        self._result = None
        self._message = "Generating avatar..."
        self._thread = threading.Thread(
            target=self._run,
            args=(frame.copy(), style_key),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, frame, style_key):
        img, err = generate_avatar(frame, style_key)
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
