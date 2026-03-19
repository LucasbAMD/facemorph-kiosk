"""
generator.py — AMD-Adapt AI Avatar Style Generator
Uses SDXL + IP-Adapter on ROCm for identity-preserving stylized avatars.

Supports both SDXL Turbo (fast, 4 steps) and SDXL Base (quality, 20 steps).
Detects which model is loaded and adjusts parameters automatically.

Identity preservation:
  Uses IP-Adapter SDXL with a face crop — the CLIP image encoder captures
  the person's appearance (face shape, skin tone, hair) and conditions
  the generation so the output resembles them.
"""

import cv2
import numpy as np
import threading
import time
import warnings
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore", message=".*upcast_vae.*")

# ── Model paths ───────────────────────────────────────────────────────────────
SDXL_PATH = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
             "sd_xl_turbo_1.0_fp16.safetensors")

IP_ADAPTER_DIR = Path.home() / "kiosk_models" / "ip_adapter"
IP_ADAPTER_BIN = IP_ADAPTER_DIR / "sdxl_models" / "ip-adapter_sdxl.bin"
IMAGE_ENCODER = IP_ADAPTER_DIR / "models" / "image_encoder"

# ── Detect model type from filename ───────────────────────────────────────────
_is_turbo = "turbo" in SDXL_PATH.name.lower()

# ── Style definitions ─────────────────────────────────────────────────────────
STYLES = {
    "avatar": {
        "label": "Avatar",
        "positive": (
            "blue skin alien Na'vi from Avatar movie, "
            "deep saturated blue skin color on entire face and body, "
            "glowing cyan bioluminescent dots on cheeks, "
            "yellow cat eyes, pointed elf ears, wide flat nose, "
            "Pandora jungle, cinematic, masterpiece"
        ),
        "negative": (
            "human skin, white skin, pink skin, pale skin, normal skin color, "
            "brown skin, realistic skin tone, normal human, "
            "cartoon, anime, blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.97, "guidance": 0.0, "steps": 8},
        "base":   {"strength": 0.85, "guidance": 7.5, "steps": 20},
        "ip_scale": 0.6,
    },
    "claymation": {
        "label": "Claymation",
        "positive": (
            "claymation character figurine, plasticine clay texture, "
            "round soft features, chunky proportions, fingerprint marks, "
            "Wallace and Gromit style, studio lighting, miniature set, "
            "masterpiece, stop-motion still"
        ),
        "negative": (
            "photorealistic, real human, CGI, 3D render, smooth plastic, "
            "blurry, low quality, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.90, "guidance": 0.0, "steps": 6},
        "base":   {"strength": 0.82, "guidance": 8.0, "steps": 20},
        "ip_scale": 0.6,
    },
    "anime": {
        "label": "Anime",
        "positive": (
            "2D anime drawing, thick black ink outlines, "
            "flat cel-shaded coloring, huge shiny anime eyes, "
            "colorful spiky stylized hair, smooth simple skin, "
            "manga illustration, Japanese animation style, "
            "masterpiece, vibrant colors"
        ),
        "negative": (
            "photorealistic, real person, photograph, 3d render, "
            "realistic face, realistic skin, natural lighting, "
            "blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.93, "guidance": 0.0, "steps": 7},
        "base":   {"strength": 0.85, "guidance": 8.0, "steps": 20},
        "ip_scale": 0.55,
    },
    "ghost": {
        "label": "Cyberpunk",
        "positive": (
            "cyberpunk portrait, neon pink and cyan lighting on face, "
            "chrome cybernetic implants on temple and jawline, "
            "glowing circuit tattoos, LED eyes, mirrored visor, "
            "rain-soaked futuristic city background with holograms, "
            "masterpiece, cinematic neon lighting"
        ),
        "negative": (
            "natural lighting, daytime, medieval, fantasy, "
            "cartoon, anime, blurry, deformed, text, watermark"
        ),
        "turbo":  {"strength": 0.90, "guidance": 0.0, "steps": 7},
        "base":   {"strength": 0.80, "guidance": 7.5, "steps": 20},
        "ip_scale": 0.6,
    },
}


# ── Pipeline state ────────────────────────────────────────────────────────────
_pipe = None
_pipe_lock = threading.Lock()
_pipe_ready = False
_face_app = None
_face_app_lock = threading.Lock()
_has_ip_adapter = False


def _init_insightface():
    """Initialize insightface for better face detection (optional)."""
    global _face_app
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="antelopev2",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        print("[Generator] insightface ready — better face detection")
        return True
    except Exception as e:
        print(f"[Generator] insightface not available, using OpenCV: {e}")
        return False


def _load_pipeline():
    """Load SDXL + IP-Adapter. Runs in background thread."""
    global _pipe, _pipe_ready, _has_ip_adapter

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

        # ── Load IP-Adapter for identity preservation ─────────────────────
        if IP_ADAPTER_BIN.exists() and IMAGE_ENCODER.exists():
            try:
                print("[Generator] Loading IP-Adapter SDXL...")
                pipe.load_ip_adapter(
                    str(IP_ADAPTER_DIR),
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.bin",
                    image_encoder_folder=str(IMAGE_ENCODER),
                )

                # Validate IP-Adapter by checking projection dimensions
                # The IP-Adapter image encoder must match the UNet's expected dims
                print("[Generator] Validating IP-Adapter compatibility...")
                try:
                    from transformers import CLIPVisionModelWithProjection
                    encoder = pipe.image_encoder
                    if encoder is not None:
                        # Get image encoder output dim
                        enc_dim = encoder.config.hidden_size
                        # SDXL expects 1280-dim from ViT-bigG, ViT-H gives 1024
                        print(f"[Generator] Image encoder dim={enc_dim}")
                        if enc_dim < 1280:
                            raise ValueError(
                                f"IP-Adapter image encoder dim={enc_dim} but "
                                f"SDXL needs 1280. Wrong ip-adapter weights file."
                            )
                except (AttributeError, ImportError):
                    # If we can't check, do a real test inference
                    test_img = Image.new("RGB", (512, 512), (128, 128, 128))
                    test_src = Image.new("RGB", (1024, 1024), (128, 128, 128))
                    pipe.set_ip_adapter_scale(0.5)
                    with torch.inference_mode():
                        pipe(
                            prompt="test",
                            image=test_src,
                            ip_adapter_image=test_img,
                            strength=0.99,
                            num_inference_steps=2,
                            guidance_scale=0.0,
                        )
                _has_ip_adapter = True
                print("[Generator] IP-Adapter loaded — identity preservation ON")
            except Exception as e:
                print(f"[Generator] IP-Adapter incompatible, disabling: {e}")
                try:
                    pipe.unload_ip_adapter()
                except Exception:
                    pass
                _has_ip_adapter = False
        else:
            print("[Generator] IP-Adapter not found — run: python setup_models.py")
            if not IP_ADAPTER_BIN.exists():
                print(f"  Missing: {IP_ADAPTER_BIN}")
            if not IMAGE_ENCODER.exists():
                print(f"  Missing: {IMAGE_ENCODER}")

        # Initialize insightface for better face detection (optional)
        _init_insightface()

        with _pipe_lock:
            _pipe = pipe
        _pipe_ready = True
        print(f"[Generator] Ready! model={model_type} ip_adapter={_has_ip_adapter}")

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
    # Try insightface first (more reliable)
    if _face_app is not None:
        try:
            with _face_app_lock:
                faces = _face_app.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
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
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
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
    y1, x1 = (h - size) // 2, (w - size) // 2
    return Image.fromarray(cv2.cvtColor(frame[y1:y1+size, x1:x1+size], cv2.COLOR_BGR2RGB))


def _detect_gender(frame):
    """Detect gender from the largest face using insightface. Returns 'male', 'female', or 'unknown'."""
    if _face_app is not None:
        try:
            with _face_app_lock:
                faces = _face_app.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                # insightface gender: 0=female, 1=male
                if hasattr(face, 'gender'):
                    return "male" if face.gender == 1 else "female"
                # some versions use 'sex' attribute
                if hasattr(face, 'sex'):
                    return "male" if face.sex == "M" else "female"
        except Exception:
            pass
    return "unknown"


def _get_face_image(frame):
    """Extract a face crop resized to 512x512 for IP-Adapter conditioning."""
    # Try insightface
    if _face_app is not None:
        try:
            with _face_app_lock:
                faces = _face_app.get(frame)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                pad = int(max(x2-x1, y2-y1) * 0.3)
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(fw, x2+pad), min(fh, y2+pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(
                        (512, 512), Image.LANCZOS)
        except Exception:
            pass

    # Fallback to Haar
    detector = _get_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(max(w, h) * 0.3)
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(fw, x+w+pad), min(fh, y+h+pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(
                (512, 512), Image.LANCZOS)

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
        (512, 512), Image.LANCZOS)


def _extract_person_crop(frame, box, padding_ratio=0.65):
    """Extract a face/body crop from frame using a bounding box dict."""
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    fh, fw = frame.shape[:2]

    # The box is a body box — find the face within it
    bx1, by1 = max(0, x), max(0, y)
    bx2, by2 = min(fw, x + w), min(fh, y + h)
    body_roi = frame[by1:by2, bx1:bx2]

    # Try to detect face within body box
    if body_roi.size > 0:
        detector = _get_face_cascade()
        gray = cv2.cvtColor(body_roi, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) > 0:
            fx, fy, fw2, fh2 = max(faces, key=lambda r: r[2] * r[3])
            # Map back to full frame coordinates
            abs_fx = bx1 + fx
            abs_fy = by1 + fy
            pad_x = int(fw2 * padding_ratio)
            pad_y = int(fh2 * padding_ratio)
            cx1 = max(0, abs_fx - pad_x)
            cy1 = max(0, abs_fy - pad_y)
            cx2 = min(fw, abs_fx + fw2 + pad_x)
            cy2 = min(fh, abs_fy + fh2 + pad_y)
            # Make roughly square
            crop_w, crop_h = cx2 - cx1, cy2 - cy1
            if crop_w > crop_h:
                diff = crop_w - crop_h
                cy1 = max(0, cy1 - diff // 2)
                cy2 = min(fh, cy2 + diff // 2)
            elif crop_h > crop_w:
                diff = crop_h - crop_w
                cx1 = max(0, cx1 - diff // 2)
                cx2 = min(fw, cx2 + diff // 2)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    # Fallback: use top portion of body box as face area
    face_h = min(by1 + int(h * 0.5), fh)
    crop = frame[by1:face_h, bx1:bx2]
    if crop.size > 0:
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    return extract_face_crop(frame, padding_ratio)


# ── Generation ────────────────────────────────────────────────────────────────
def _build_prompts(style, gender):
    """Build gender-aware positive and negative prompts."""
    prompt = style["positive"]
    neg = style["negative"]
    if gender == "male":
        prompt = f"male, man, masculine features, {prompt}"
        neg = f"female, woman, feminine, breasts, {neg}"
    elif gender == "female":
        prompt = f"female, woman, feminine features, {prompt}"
        neg = f"male, man, masculine, beard, {neg}"
    return prompt, neg


def _generate_single(pipe, source_pil, style, style_key, gender="unknown"):
    """Generate one avatar from a single face crop PIL image."""
    global _has_ip_adapter
    import torch

    params = style["turbo"] if _is_turbo else style["base"]
    prompt, neg_prompt = _build_prompts(style, gender)

    model_type = "Turbo" if _is_turbo else "Base"
    print(f"[Generator] Style={style_key} Model={model_type} "
          f"Steps={params['steps']} Strength={params['strength']} "
          f"Gender={gender}")

    generator = torch.Generator(device="cuda").manual_seed(
        int(time.time()) % 2**32)

    gen_kwargs = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "image": source_pil,
        "strength": params["strength"],
        "num_inference_steps": params["steps"],
        "guidance_scale": params["guidance"],
        "generator": generator,
    }

    if _has_ip_adapter:
        try:
            pipe.set_ip_adapter_scale(style["ip_scale"])
            gen_kwargs["ip_adapter_image"] = source_pil.resize(
                (512, 512), Image.LANCZOS)
        except Exception as e:
            print(f"[Generator] IP-Adapter conditioning failed: {e}")

    with torch.inference_mode():
        result = pipe(**gen_kwargs).images[0]

    return result


def generate_avatar(frame, style_key, gender="unknown", face_boxes=None):
    """
    Generate stylized avatar(s) from a webcam frame.
    If face_boxes is provided, generates one avatar per person and combines.
    gender: 'male', 'female', or 'unknown' — used to guide prompt.
    """
    global _has_ip_adapter

    if not is_ready():
        return None, "AI pipeline loading — please wait a moment"

    style = STYLES.get(style_key)
    if not style:
        return None, f"Unknown style: {style_key}"

    try:
        import torch

        with _pipe_lock:
            pipe = _pipe

        # Auto-detect gender from insightface if not provided
        if gender == "unknown":
            gender = _detect_gender(frame)

        # Determine face crops to generate
        if face_boxes and len(face_boxes) > 1:
            # Multi-person: extract each person's face
            crops = []
            for box in face_boxes:
                crop = _extract_person_crop(frame, box)
                crops.append(crop.resize((1024, 1024), Image.LANCZOS))
            print(f"[Generator] Multi-person: {len(crops)} faces")
        else:
            # Single person (or no boxes): use largest face
            if face_boxes and len(face_boxes) == 1:
                crop = _extract_person_crop(frame, face_boxes[0])
            else:
                crop = extract_face_crop(frame)
            crops = [crop.resize((1024, 1024), Image.LANCZOS)]

        start = time.time()
        results = []
        for i, source_pil in enumerate(crops):
            if len(crops) > 1:
                print(f"[Generator] Generating person {i+1}/{len(crops)}...")
            result = _generate_single(pipe, source_pil, style, style_key, gender)
            results.append(cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR))

        elapsed = time.time() - start
        print(f"[Generator] Done in {elapsed:.1f}s ({len(results)} avatar(s))")

        if len(results) == 1:
            return results[0], None

        # Combine multiple results into a side-by-side grid
        return _combine_grid(results), None

    except Exception as e:
        import traceback
        traceback.print_exc()

        # ── Fallback: unload IP-Adapter and retry plain img2img ───────────
        try:
            print("[Generator] Retrying without IP-Adapter...")
            import torch

            with _pipe_lock:
                pipe = _pipe
                try:
                    pipe.unload_ip_adapter()
                except Exception:
                    pass

            _has_ip_adapter = False

            source_pil = extract_face_crop(frame).resize((1024, 1024), Image.LANCZOS)
            prompt, neg_prompt = _build_prompts(style, gender)
            params = style["turbo"] if _is_turbo else style["base"]
            generator = torch.Generator(device="cuda").manual_seed(42)

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

            print("[Generator] Fallback succeeded (IP-Adapter disabled for future runs)")
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR), None

        except Exception as e2:
            import traceback
            traceback.print_exc()
            return None, f"Generation failed: {e2}"


def _combine_grid(images):
    """Combine multiple avatar images into a side-by-side grid."""
    n = len(images)
    if n == 0:
        return None
    if n == 1:
        return images[0]

    # Resize all to same height
    target_h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, target_h))
        resized.append(img)

    # For 2-3 people: horizontal row
    if n <= 3:
        gap = 4
        total_w = sum(img.shape[1] for img in resized) + gap * (n - 1)
        canvas = np.zeros((target_h, total_w, 3), dtype=np.uint8)
        x = 0
        for img in resized:
            canvas[:, x:x+img.shape[1]] = img
            x += img.shape[1] + gap
        return canvas

    # 4+ people: 2-column grid
    cols = 2
    rows = (n + 1) // 2
    target_w = min(img.shape[1] for img in resized)
    cell_resized = []
    for img in resized:
        if img.shape[1] != target_w:
            scale = target_w / img.shape[1]
            new_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (target_w, new_h))
        cell_resized.append(img)

    cell_h = min(img.shape[0] for img in cell_resized)
    gap = 4
    canvas = np.zeros((cell_h * rows + gap * (rows - 1),
                        target_w * cols + gap, 3), dtype=np.uint8)
    for i, img in enumerate(cell_resized):
        r, c = i // cols, i % cols
        y = r * (cell_h + gap)
        x = c * (target_w + gap)
        canvas[y:y+cell_h, x:x+target_w] = img[:cell_h, :target_w]

    return canvas


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
            print(f"[Generator] AI engine ready (ip_adapter={_has_ip_adapter})")
        else:
            print("[Generator] AI engine failed to load")

    def check_available(self):
        self.available = is_ready()
        return self.available

    def generate(self, frame, style_key, selection_mask=None, face_boxes=None,
                 gender="unknown"):
        if self._status == "generating":
            return False
        self._status = "generating"
        self._result = None
        self._message = "Generating avatar..."
        self._thread = threading.Thread(
            target=self._run,
            args=(frame.copy(), style_key, gender, face_boxes),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, frame, style_key, gender="unknown", face_boxes=None):
        img, err = generate_avatar(frame, style_key, gender=gender,
                                   face_boxes=face_boxes)
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
