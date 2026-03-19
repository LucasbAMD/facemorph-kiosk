"""
generator.py — AI Avatar Style Generator
Uses SDXL + IP-Adapter FaceID on ROCm for identity-preserving stylized avatars.

Pipeline:
  1. insightface extracts a 512-dim face embedding (your actual facial identity)
  2. IP-Adapter FaceID projects that embedding into the diffusion process
  3. SDXL generates a fully stylized avatar that genuinely looks like you
  4. Styles: Avatar (blue alien), Claymation, Anime, Ghost

Why FaceID matters:
  Regular IP-Adapter uses generic CLIP image features — it captures "a person"
  but not YOUR specific face. FaceID uses actual face recognition embeddings
  from insightface, the same tech used in face unlock. This means the generated
  avatar preserves your actual facial structure, nose shape, eye spacing, etc.
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

# FaceID adapter (primary — strong identity preservation)
FACEID_DIR = Path.home() / "kiosk_models" / "ip_adapter_faceid"
FACEID_BIN = FACEID_DIR / "ip-adapter-faceid_sdxl.bin"
FACEID_LORA = FACEID_DIR / "ip-adapter-faceid_sdxl_lora.safetensors"

# Regular IP-Adapter (fallback — weaker identity)
IP_ADAPTER_DIR = Path.home() / "kiosk_models" / "ip_adapter"
IP_ADAPTER_BIN = IP_ADAPTER_DIR / "sdxl_models" / "ip-adapter_sdxl.bin"
IMAGE_ENCODER = IP_ADAPTER_DIR / "models" / "image_encoder"

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
        "strength": 0.78,
        "guidance": 7.5,
        "steps": 25,
        "ip_scale": 0.7,
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
        "strength": 0.82,
        "guidance": 8.0,
        "steps": 25,
        "ip_scale": 0.7,
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
        "strength": 0.82,
        "guidance": 8.0,
        "steps": 25,
        "ip_scale": 0.65,
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
        "strength": 0.75,
        "guidance": 7.5,
        "steps": 25,
        "ip_scale": 0.7,
    },
}


# ── Pipeline state ────────────────────────────────────────────────────────────
_pipe = None
_pipe_lock = threading.Lock()
_pipe_ready = False
_face_app = None
_face_app_lock = threading.Lock()
_identity_mode = "none"  # "faceid", "ip_adapter", or "none"


def _init_insightface():
    """Initialize insightface for face embedding extraction."""
    global _face_app
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="antelopev2",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        print("[Generator] insightface ready (antelopev2)")
        return True
    except Exception as e:
        print(f"[Generator] insightface not available: {e}")
        return False


def _load_pipeline():
    """Load SDXL + best available identity adapter. Runs in background thread."""
    global _pipe, _pipe_ready, _identity_mode

    try:
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        print("[Generator] Loading SDXL pipeline...")
        if not SDXL_PATH.exists():
            print(f"[Generator] ERROR: Model not found at {SDXL_PATH}")
            return

        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            str(SDXL_PATH),
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")

        # Try enabling memory optimizations
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        # ── Try IP-Adapter FaceID (best identity preservation) ────────────
        faceid_loaded = False
        if FACEID_BIN.exists():
            try:
                print(f"[Generator] Loading IP-Adapter FaceID from {FACEID_BIN}...")
                pipe.load_ip_adapter(
                    str(FACEID_DIR),
                    subfolder="",
                    weight_name="ip-adapter-faceid_sdxl.bin",
                )

                # Load FaceID LoRA for better quality
                if FACEID_LORA.exists():
                    try:
                        pipe.load_lora_weights(
                            str(FACEID_DIR),
                            weight_name="ip-adapter-faceid_sdxl_lora.safetensors",
                        )
                        pipe.fuse_lora(lora_scale=0.7)
                        print("[Generator] FaceID LoRA loaded and fused")
                    except Exception as e:
                        print(f"[Generator] FaceID LoRA failed (non-critical): {e}")

                # Initialize insightface for face embeddings
                if _init_insightface():
                    _identity_mode = "faceid"
                    faceid_loaded = True
                    print("[Generator] IP-Adapter FaceID active — strong identity preservation")
                else:
                    print("[Generator] insightface failed — FaceID adapter loaded but unusable")
                    # Unload the adapter since we can't use it without face embeddings
                    pipe.unload_ip_adapter()
            except Exception as e:
                print(f"[Generator] IP-Adapter FaceID failed: {e}")
                import traceback
                traceback.print_exc()

        # ── Fallback: regular IP-Adapter (weaker identity) ────────────────
        if not faceid_loaded and IP_ADAPTER_BIN.exists() and IMAGE_ENCODER.exists():
            try:
                print("[Generator] Falling back to regular IP-Adapter...")
                pipe.load_ip_adapter(
                    str(IP_ADAPTER_DIR),
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.bin",
                    image_encoder_folder=str(IMAGE_ENCODER),
                )
                _identity_mode = "ip_adapter"
                print("[Generator] Regular IP-Adapter loaded — moderate identity preservation")
            except Exception as e:
                print(f"[Generator] Regular IP-Adapter also failed: {e}")

        if _identity_mode == "none":
            print("[Generator] WARNING: No identity adapter loaded!")
            print("           Avatars will NOT look like you.")
            print("           Run: python setup_models.py")

        with _pipe_lock:
            _pipe = pipe
        _pipe_ready = True
        print(f"[Generator] Pipeline ready! (identity_mode={_identity_mode})")

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
    """
    # Try insightface first (more reliable detection)
    if _face_app is not None:
        try:
            with _face_app_lock:
                faces = _face_app.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                w, h = x2 - x1, y2 - y1
                pad_x = int(w * padding_ratio)
                pad_y = int(h * padding_ratio)
                fh, fw = frame.shape[:2]
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(fw, x2 + pad_x)
                y2 = min(fh, y2 + pad_y)

                # Make roughly square
                crop_w, crop_h = x2 - x1, y2 - y1
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
        except Exception:
            pass

    # Fallback to Haar cascade
    detector = _get_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        fh, fw = frame.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)

        crop_w, crop_h = x2 - x1, y2 - y1
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

    # Last resort: center crop
    h, w = frame.shape[:2]
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    crop = frame[y1:y1+size, x1:x1+size]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _get_face_embedding(frame):
    """
    Extract face embedding using insightface.
    Returns (embedding_tensor, face_info) or (None, None).
    """
    if _face_app is None:
        return None, None

    try:
        import torch
        with _face_app_lock:
            faces = _face_app.get(frame)
        if not faces:
            print("[Generator] No face found by insightface")
            return None, None

        # Pick largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding  # numpy (512,)

        # Convert to tensor: shape (1, 1, 512) for IP-Adapter FaceID
        emb_tensor = torch.from_numpy(emb).unsqueeze(0).to(
            device="cuda", dtype=torch.float16)

        print(f"[Generator] Face embedding extracted (norm={np.linalg.norm(emb):.2f})")
        return emb_tensor, face

    except Exception as e:
        print(f"[Generator] Face embedding extraction failed: {e}")
        return None, None


def _get_face_image_for_ip(frame):
    """Extract a tight face crop as PIL Image for regular IP-Adapter."""
    detector = _get_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(max(w, h) * 0.3)
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            return pil.resize((512, 512), Image.LANCZOS)

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
        (512, 512), Image.LANCZOS)


# ── Generation ────────────────────────────────────────────────────────────────
def generate_avatar(frame, style_key):
    """
    Generate a stylized avatar from a webcam frame.

    Uses 3-tier identity preservation:
      1. FaceID (best) — actual face recognition embeddings
      2. Regular IP-Adapter (fallback) — CLIP image features
      3. Plain img2img (minimal) — structure from low denoise only

    Returns: (bgr_image, None) on success, (None, error_string) on failure
    """
    if not is_ready():
        return None, "AI pipeline loading — please wait a moment"

    style = STYLES.get(style_key)
    if not style:
        return None, f"Unknown style: {style_key}"

    try:
        import torch

        # Extract face-centered crop as source image
        source_pil = extract_face_crop(frame)
        source_pil = source_pil.resize((1024, 1024), Image.LANCZOS)

        with _pipe_lock:
            pipe = _pipe

        ip_scale = style["ip_scale"]
        strength = style["strength"]

        print(f"[Generator] Style={style_key} Mode={_identity_mode} "
              f"Steps={style['steps']} Strength={strength} "
              f"Guidance={style['guidance']} IP-Scale={ip_scale}")

        start = time.time()
        generator = torch.Generator(device="cuda").manual_seed(
            int(time.time()) % 2**32)

        gen_kwargs = {
            "prompt": style["positive"],
            "negative_prompt": style["negative"],
            "image": source_pil,
            "strength": strength,
            "num_inference_steps": style["steps"],
            "guidance_scale": style["guidance"],
            "generator": generator,
        }

        # ── FaceID mode: use face recognition embeddings ──────────────────
        if _identity_mode == "faceid":
            face_emb, face_info = _get_face_embedding(frame)
            if face_emb is not None:
                pipe.set_ip_adapter_scale(ip_scale)

                # Project embedding through the FaceID projection layer
                proj_layers = pipe.unet.encoder_hid_proj.image_projection_layers
                if proj_layers:
                    projected = proj_layers[0](face_emb)
                    # Duplicate for classifier-free guidance (unconditional + conditional)
                    neg_embed = torch.zeros_like(projected)
                    gen_kwargs["ip_adapter_image_embeds"] = [
                        torch.cat([neg_embed, projected])
                    ]
                    print(f"[Generator] FaceID embedding projected "
                          f"(shape={projected.shape})")
                else:
                    print("[Generator] WARN: No projection layers found")
            else:
                print("[Generator] WARN: No face found — generating without identity")

        # ── Regular IP-Adapter mode: use face image ───────────────────────
        elif _identity_mode == "ip_adapter":
            face_pil = _get_face_image_for_ip(frame)
            pipe.set_ip_adapter_scale(ip_scale)
            gen_kwargs["ip_adapter_image"] = face_pil
            print(f"[Generator] Using regular IP-Adapter with face crop")

        # ── No identity mode ──────────────────────────────────────────────
        else:
            print("[Generator] WARN: No identity adapter — avatar won't look like you")

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
            print(f"[Generator] AI engine ready (identity={_identity_mode})")
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
