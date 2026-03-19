"""
generator.py — AMD Adapt Kiosk AI Generation
Replaces comfy_bridge.py. Uses diffusers + IP-Adapter directly on ROCm.
No ComfyUI required.

How it works:
  1. Loads SDXL base 1.0 from local safetensors file
  2. Loads IP-Adapter SDXL — conditions generation on person's actual face
  3. Takes webcam frame + face crop + character prompt
  4. Runs img2img at denoise 0.65 — preserves body structure
  5. IP-Adapter weight 0.6 — person's face features carry through to output

Result: person genuinely looks like themselves as the character,
not a generic version of the character.
"""

import cv2
import numpy as np
import threading
import time
import gc
from pathlib import Path
from PIL import Image

# ── Model paths ───────────────────────────────────────────────────────────────
SDXL_PATH      = (Path.home() / "ComfyUI" / "models" / "checkpoints" /
                   "sd_xl_turbo_1.0_fp16.safetensors")
IP_ADAPTER_DIR = Path.home() / "kiosk_models" / "ip_adapter"

# ── Character prompts — male + female variants ────────────────────────────────
CHARACTER_PROMPTS = {
    "navi": {
        "positive_m": (
            "portrait of a male Na'vi from Avatar, blue alien skin, "
            "cyan bioluminescent face markings, large amber eyes, pointed ears, "
            "same facial structure and features as the person, "
            "Pandora jungle background, cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "portrait of a female Na'vi from Avatar, blue alien skin, "
            "cyan bioluminescent face markings, large amber eyes, pointed ears, "
            "same facial structure and features as the person, "
            "Pandora jungle background, cinematic, photorealistic, 8k"
        ),
        "negative": (
            "human skin, pink skin, multiple people, ugly, blurry, "
            "cartoon, low quality, deformed"
        ),
        "denoise": 0.65,
        "ip_weight": 0.6,
    },
    "hulk": {
        "positive_m": (
            "portrait of a male Incredible Hulk, massive green muscles, "
            "green skin, same facial structure as the person, "
            "stormy sky background, cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "portrait of a female She-Hulk, strong muscular green-skinned woman, "
            "green skin, same facial structure as the person, "
            "stormy sky background, cinematic, photorealistic, 8k"
        ),
        "negative": (
            "normal skin, multiple people, ugly, blurry, cartoon, low quality"
        ),
        "denoise": 0.65,
        "ip_weight": 0.6,
    },
    "thanos": {
        "positive_m": (
            "portrait of Thanos the Marvel villain, purple Titan skin, "
            "same facial structure as the person, infinity gauntlet, "
            "cosmic nebula background, cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "portrait of a female purple Titan, Gamora-style, purple skin, "
            "same facial structure as the person, cosmic armor, "
            "nebula background, cinematic, photorealistic, 8k"
        ),
        "negative": (
            "human skin, multiple people, ugly, blurry, cartoon, low quality"
        ),
        "denoise": 0.65,
        "ip_weight": 0.6,
    },
    "predator": {
        "positive_m": (
            "portrait of a Predator alien warrior, mandibles, dreadlocks, "
            "biomask helmet, mesh armor, same facial structure as the person, "
            "dark jungle background, cinematic, highly detailed, 8k"
        ),
        "positive_f": (
            "portrait of a female Predator alien warrior, mandibles, dreadlocks, "
            "biomask helmet, same facial structure as the person, "
            "dark jungle background, cinematic, highly detailed, 8k"
        ),
        "negative": (
            "human face, multiple people, ugly, blurry, cartoon, low quality"
        ),
        "denoise": 0.65,
        "ip_weight": 0.55,
    },
    "ghost": {
        "positive_m": (
            "portrait of a male spectral ghost, translucent glowing body, "
            "pale ethereal skin, white aura, same facial structure as the person, "
            "dark atmospheric background, cinematic, 8k"
        ),
        "positive_f": (
            "portrait of a female spectral ghost, translucent glowing body, "
            "pale ethereal skin, white flowing aura, same facial structure as the person, "
            "dark atmospheric background, cinematic, 8k"
        ),
        "negative": (
            "solid opaque body, multiple people, ugly, blurry, cartoon, low quality"
        ),
        "denoise": 0.62,
        "ip_weight": 0.55,
    },
    "groot": {
        "positive_m": (
            "portrait of Groot from Guardians of the Galaxy, brown bark skin, "
            "twig hair with green leaves, same facial structure as the person, "
            "forest background with light shafts, cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "portrait of a female Groot, living tree humanoid woman, "
            "brown bark skin, flower and leaf hair, same facial structure as the person, "
            "forest background with light shafts, cinematic, photorealistic, 8k"
        ),
        "negative": (
            "human skin, multiple people, ugly, blurry, cartoon, low quality"
        ),
        "denoise": 0.65,
        "ip_weight": 0.6,
    },
    "cyberpunk": {
        "positive_m": (
            "portrait of a male cyberpunk character, neon hair streaks, "
            "glowing circuit clothing, cybernetic eye implants, "
            "same facial structure as the person, "
            "neon rain-soaked city background, cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "portrait of a female cyberpunk character, neon hair streaks, "
            "glowing circuit clothing, cybernetic eye implants, "
            "same facial structure as the person, "
            "neon rain-soaked city background, cinematic, photorealistic, 8k"
        ),
        "negative": (
            "fantasy, medieval, multiple people, ugly, blurry, cartoon, low quality"
        ),
        "denoise": 0.65,
        "ip_weight": 0.6,
    },
    "claymation": {
        "positive_m": (
            "claymation figurine of a male person, smooth matte clay texture, "
            "same facial features as the person, Aardman style, "
            "stop-motion background, plasticine, handcrafted"
        ),
        "positive_f": (
            "claymation figurine of a female person, smooth matte clay texture, "
            "same facial features as the person, Aardman style, "
            "stop-motion background, plasticine, handcrafted"
        ),
        "negative": (
            "photorealistic skin, CGI, multiple people, ugly, blurry, low quality"
        ),
        "denoise": 0.70,
        "ip_weight": 0.65,
    },
    "anime": {
        "positive_m": (
            "male anime character, same face as the person, "
            "same hair color and style, sharp anime eyes, "
            "cel-shaded skin, clean line art, vibrant colors, "
            "Studio Ghibli style, soft bokeh background"
        ),
        "positive_f": (
            "female anime character, same face as the person, "
            "same hair color and style, large expressive eyes, "
            "cel-shaded skin, clean line art, vibrant colors, "
            "Studio Ghibli style, soft bokeh background"
        ),
        "negative": (
            "photorealistic, photograph, 3d render, multiple people, "
            "ugly, blurry, bad anatomy, deformed eyes"
        ),
        "denoise": 0.68,
        "ip_weight": 0.65,
    },
}


def get_prompts(char_key, gender):
    cfg = CHARACTER_PROMPTS.get(char_key, CHARACTER_PROMPTS["navi"])
    suffix   = "f" if gender == "female" else "m"
    positive = cfg.get(f"positive_{suffix}", cfg.get("positive_m", ""))
    negative = cfg.get("negative", "ugly, blurry, low quality")
    return positive, negative


# ── Pipeline loader ───────────────────────────────────────────────────────────
_pipe      = None
_pipe_lock = threading.Lock()
_pipe_ready = False


def _load_pipeline():
    """
    Load SDXL + IP-Adapter pipeline once at startup.
    Runs in a background thread so the kiosk UI is responsive immediately.
    """
    global _pipe, _pipe_ready
    try:
        import torch

        print("[Generator] Loading SDXL base 1.0...")
        print(f"[Generator] From: {SDXL_PATH}")

        if not SDXL_PATH.exists():
            print(f"[Generator] ERR: Model not found at {SDXL_PATH}")
            return

        # Try from_single_file first, fall back to from_pretrained
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                str(SDXL_PATH),
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        except Exception as e1:
            print(f"[Generator] from_single_file failed ({e1}), trying alternate loader...")
            from diffusers import StableDiffusionXLImg2ImgPipeline
            from diffusers.loaders import FromSingleFileMixin
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                str(SDXL_PATH),
                torch_dtype=torch.float16,
            )

        # Move to GPU
        pipe = pipe.to("cuda")

        # IP-Adapter disabled for now — dimension mismatch with current diffusers
        # version. Basic img2img works reliably without it.
        # Will be re-enabled once diffusers version is pinned.
        print("[Generator] Using img2img without IP-Adapter")

        with _pipe_lock:
            _pipe = pipe
        _pipe_ready = True
        print("[Generator] Ready!")

    except Exception as e:
        print(f"[Generator] ERR loading pipeline: {e}")
        import traceback
        traceback.print_exc()


def is_ready():
    return _pipe_ready and _pipe is not None


# ── Face extraction ───────────────────────────────────────────────────────────
def _extract_face(frame):
    """
    Extract the largest face from the frame for IP-Adapter conditioning.
    Returns a PIL Image of the face crop, or the full frame if no face found.
    """
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(max(w, h) * 0.4)
        fh, fw = frame.shape[:2]
        x1 = max(0, x - pad);     y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad); y2 = min(fh, y + h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    # Fallback: full frame
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ── Gender detection ──────────────────────────────────────────────────────────
_GENDER_MODEL_DIR = Path.home() / ".cache" / "kiosk_models"
_GENDER_PROTO     = _GENDER_MODEL_DIR / "gender_deploy.prototxt"
_GENDER_MODEL     = _GENDER_MODEL_DIR / "gender_net.caffemodel"
_FACE_PROTO       = _GENDER_MODEL_DIR / "opencv_face_detector.pbtxt"
_FACE_MODEL       = _GENDER_MODEL_DIR / "opencv_face_detector_uint8.pb"
_gender_net       = None
_face_net         = None
_gender_lock      = threading.Lock()


def _ensure_gender_models():
    import urllib.request as ur
    _GENDER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        _GENDER_PROTO: (
            "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/"
            "master/gender_deploy.prototxt"
        ),
        _GENDER_MODEL: (
            "https://github.com/smahesh29/Gender-and-Age-Detection/raw/"
            "master/gender_net.caffemodel"
        ),
        _FACE_PROTO: (
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
            "face_detector/opencv_face_detector.pbtxt"
        ),
        _FACE_MODEL: (
            "https://github.com/opencv/opencv_3rdparty/raw/"
            "dnn_samples_face_detector_20180220_uint8/"
            "opencv_face_detector_uint8.pb"
        ),
    }
    for path, url in files.items():
        if not path.exists():
            try:
                ur.urlretrieve(url, path)
            except Exception:
                pass


def detect_gender_for_box(frame, box):
    """Detect gender from a face bounding box. Returns 'male', 'female', or 'unknown'."""
    global _gender_net, _face_net
    try:
        with _gender_lock:
            if _gender_net is None:
                _ensure_gender_models()
                if _FACE_MODEL.exists() and _GENDER_MODEL.exists():
                    _face_net   = cv2.dnn.readNet(str(_FACE_MODEL), str(_FACE_PROTO))
                    _gender_net = cv2.dnn.readNet(str(_GENDER_MODEL), str(_GENDER_PROTO))

            if _gender_net is None:
                return "unknown"

        fh, fw = frame.shape[:2]
        x1 = max(0, box["x"] - 20)
        y1 = max(0, box["y"] - 20)
        x2 = min(fw, box["x"] + box["w"] + 20)
        y2 = min(fh, box["y"] + box["h"] + 20)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"

        blob = cv2.dnn.blobFromImage(
            roi, 1.0, (227, 227),
            [78.4263377603, 87.7689143744, 114.895847746],
            swapRB=False
        )
        with _gender_lock:
            _gender_net.setInput(blob)
            preds = _gender_net.forward()[0]

        gender = "female" if preds[0] > preds[1] else "male"
        print(f"[Gender] {gender} (conf={float(max(preds)):.2f})")
        return gender
    except Exception as e:
        print(f"[Gender] Detection failed: {e}")
        return "unknown"


# ── Main generation ───────────────────────────────────────────────────────────
def generate_character(frame, char_key, selection_mask=None,
                       face_boxes=None, timeout=120):
    """
    Generate a character transformation of the person in the frame.

    Args:
        frame:          Full BGR camera frame
        char_key:       Character key e.g. 'navi', 'hulk', 'anime'
        selection_mask: Float32 mask of selected person (unused, kept for compat)
        face_boxes:     List of selected face dicts from face_processor
        timeout:        Max seconds to wait (unused, kept for compat)

    Returns:
        (image_bgr, error_string) — one will be None
    """
    if not is_ready():
        return None, "AI pipeline not ready yet — please wait a moment and try again"

    try:
        import torch

        # ── Detect gender ──────────────────────────────────────────────────
        gender = "unknown"
        if face_boxes:
            for box in face_boxes:
                registered = box.get("registered_gender", "unknown")
                if registered in ("male", "female"):
                    gender = registered
                    print(f"[Gender] Person {box['index']}: {gender} (from profile)")
                    break
            if gender == "unknown":
                gender = detect_gender_for_box(frame, face_boxes[0])
        print(f"[Generator] character={char_key} gender={gender}")

        # ── Build prompts ──────────────────────────────────────────────────
        cfg      = CHARACTER_PROMPTS.get(char_key, CHARACTER_PROMPTS["navi"])
        positive, negative = get_prompts(char_key, gender)
        denoise  = cfg.get("denoise", 0.65)

        # ── Prepare source image ───────────────────────────────────────────
        h, w = frame.shape[:2]
        scale = 1024 / max(h, w)
        nh = int((h * scale) // 64) * 64
        nw = int((w * scale) // 64) * 64
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        source_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        # ── Run pipeline ───────────────────────────────────────────────────
        with _pipe_lock:
            pipe = _pipe

        print(f"[Generator] Running img2img — character={char_key} gender={gender}")
        start = time.time()

        generator = torch.Generator(device="cuda").manual_seed(
            int(time.time()) % 2**32)

        with torch.inference_mode():
            result = pipe(
                prompt=positive,
                negative_prompt=negative,
                image=source_pil,
                strength=denoise,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]

        elapsed = time.time() - start
        print(f"[Generator] Done in {elapsed:.1f}s — output {result.size}")

        # Convert PIL → BGR numpy for the rest of the pipeline
        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return result_bgr, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)


# ── ComfyBridge-compatible class (same interface as original) ─────────────────
class ComfyBridge:
    """
    Drop-in replacement for the original ComfyBridge class.
    Same public interface — main.py needs zero changes.
    """
    def __init__(self):
        self._result     = None
        self._status     = "idle"
        self._message    = ""
        self._thread     = None
        self.available   = False  # becomes True when pipeline loads
        self.instantid   = False

        # Load pipeline in background thread so kiosk starts instantly
        threading.Thread(target=self._init_pipeline, daemon=True).start()

    def _init_pipeline(self):
        _load_pipeline()
        self.available = is_ready()
        if self.available:
            print("[Generator] ComfyBridge ready — AI generation available")
        else:
            print("[Generator] ERR: Pipeline failed to load")

    def check_available(self):
        self.available = is_ready()
        return self.available

    def generate(self, frame, char_key, selection_mask=None, face_boxes=None):
        if self._status == "generating":
            return False
        self._status  = "generating"
        self._result  = None
        self._message = "Transforming..."
        self._thread  = threading.Thread(
            target=self._run,
            args=(frame.copy(), char_key, selection_mask, face_boxes),
            daemon=True
        )
        self._thread.start()
        return True

    def _run(self, frame, char_key, mask, face_boxes):
        img, err = generate_character(frame, char_key, mask, face_boxes)
        if img is not None:
            self._result  = img
            self._status  = "done"
            self._message = "Done!"
        else:
            self._status  = "error"
            self._message = err or "Generation failed"

    def get_status(self):
        return {"status": self._status, "message": self._message}

    def get_result(self):
        if self._result is not None:
            img          = self._result.copy()
            self._status = "idle"
            self._result = None
            return img
        return None

    def reset(self):
        self._status  = "idle"
        self._result  = None
        self._message = ""
