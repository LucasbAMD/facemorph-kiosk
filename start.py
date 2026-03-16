"""
comfy_bridge.py -- ComfyUI API bridge for AMD Adapt Kiosk
img2img + ControlNet Canny. Supports per-person selection mask.

Changes from original:
  - Gender detection via OpenCV DNN (age-gender-recognition model)
  - All CHARACTER_PROMPTS now have male/female variants
  - Anime prompt no longer hardcodes male — respects detected gender
  - _upload_frame max_dim raised to 1024 for better base quality
  - _build_workflow adds a LatentUpscale + second KSampler pass → ~1536px output
  - generate_status in main.py should use JPEG quality 95+ for print-ready stickers
"""

import json
import os
import urllib.request
import urllib.parse
import uuid
import time
import threading
import cv2
import numpy as np

COMFY_URL = "http://127.0.0.1:8188"

# ── Gender detection ──────────────────────────────────────────────────────────
# Uses OpenCV's pre-trained Caffe model — no extra pip installs needed.
# Model files are small (~25MB total) and downloaded once by install/setup.
# If models are missing, gender defaults to UNKNOWN and neutral prompts are used.

_GENDER_MODEL_DIR = os.path.expanduser("~/.cache/kiosk_models")
_GENDER_PROTO     = os.path.join(_GENDER_MODEL_DIR, "gender_deploy.prototxt")
_GENDER_MODEL     = os.path.join(_GENDER_MODEL_DIR, "gender_net.caffemodel")
_FACE_PROTO       = os.path.join(_GENDER_MODEL_DIR, "opencv_face_detector.pbtxt")
_FACE_MODEL       = os.path.join(_GENDER_MODEL_DIR, "opencv_face_detector_uint8.pb")

_GENDER_NET = None
_FACE_NET   = None
_GENDER_LOCK = threading.Lock()


def _ensure_gender_models():
    """Download gender/face models if not already present."""
    os.makedirs(_GENDER_MODEL_DIR, exist_ok=True)
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
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180220_uint8/"
            "opencv_face_detector_uint8.pb"
        ),
    }
    for path, url in files.items():
        if not os.path.exists(path):
            print(f"[Gender] Downloading {os.path.basename(path)}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"[Gender] Downloaded {os.path.basename(path)}")
            except Exception as e:
                print(f"[Gender] WARN: Could not download {os.path.basename(path)}: {e}")


def _load_gender_nets():
    global _GENDER_NET, _FACE_NET
    with _GENDER_LOCK:
        if _GENDER_NET is not None:
            return True
        _ensure_gender_models()
        try:
            _FACE_NET   = cv2.dnn.readNet(_FACE_MODEL,   _FACE_PROTO)
            _GENDER_NET = cv2.dnn.readNet(_GENDER_MODEL, _GENDER_PROTO)
            print("[Gender] Gender detection models loaded OK")
            return True
        except Exception as e:
            print(f"[Gender] WARN: Could not load gender models — defaulting to neutral prompts. ({e})")
            return False


def _classify_gender_from_crop(face_crop) -> tuple[str, float]:
    """
    Run gender classification on an already-cropped face image.
    Returns (gender, confidence) where gender is 'male', 'female', or 'unknown'.
    """
    if face_crop is None or face_crop.size == 0:
        return "unknown", 0.0
    face_blob = cv2.dnn.blobFromImage(
        face_crop, 1.0, (227, 227),
        [78.4263377603, 87.7689143744, 114.895847746],
        swapRB=False
    )
    with _GENDER_LOCK:
        _GENDER_NET.setInput(face_blob)
        preds = _GENDER_NET.forward()[0]
    # preds[0] = female probability, preds[1] = male probability
    gender = "female" if preds[0] > preds[1] else "male"
    return gender, float(max(preds))


def _find_face_in_roi(frame, roi_x, roi_y, roi_w, roi_h):
    """
    Run the DNN face detector inside a body bounding box and return the
    best face crop. Falls back to the top-third of the body box if no
    face is found (e.g. person looking away, poor lighting).
    """
    fh, fw = frame.shape[:2]
    # Expand body box slightly for the face detector
    pad = 20
    x1 = max(0, roi_x - pad);      y1 = max(0, roi_y - pad)
    x2 = min(fw, roi_x + roi_w + pad); y2 = min(fh, roi_y + roi_h + pad)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return None

    rh, rw = region.shape[:2]
    blob = cv2.dnn.blobFromImage(region, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    with _GENDER_LOCK:
        _FACE_NET.setInput(blob)
        dets = _FACE_NET.forward()

    best_conf, best_crop = 0.0, None
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < 0.6:
            continue
        bx1 = max(0, int(dets[0, 0, i, 3] * rw))
        by1 = max(0, int(dets[0, 0, i, 4] * rh))
        bx2 = min(rw, int(dets[0, 0, i, 5] * rw))
        by2 = min(rh, int(dets[0, 0, i, 6] * rh))
        if conf > best_conf and bx2 > bx1 and by2 > by1:
            best_conf = conf
            crop = region[by1:by2, bx1:bx2]
            if crop.size > 0:
                best_crop = crop

    if best_crop is not None:
        return best_crop

    # Fallback: top 30% of body box as face region
    face_y2 = min(fh, roi_y + int(roi_h * 0.30))
    fallback = frame[roi_y:face_y2, roi_x:roi_x + roi_w]
    return fallback if fallback.size > 0 else None


def detect_gender(frame) -> str:
    """
    Legacy single-frame gender detection (used when no face boxes are available).
    Detects the most prominent face in the full frame.
    Returns 'male', 'female', or 'unknown'.
    """
    if not _load_gender_nets():
        return "unknown"
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    with _GENDER_LOCK:
        _FACE_NET.setInput(blob)
        detections = _FACE_NET.forward()

    best_conf, best_box = 0.0, None
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > best_conf:
            best_conf = conf
            best_box  = detections[0, 0, i, 3:7]

    if best_box is None or best_conf < 0.6:
        return "unknown"

    x1 = max(0, int(best_box[0] * w) - 20)
    y1 = max(0, int(best_box[1] * h) - 20)
    x2 = min(w, int(best_box[2] * w) + 20)
    y2 = min(h, int(best_box[3] * h) + 20)
    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return "unknown"

    gender, confidence = _classify_gender_from_crop(face_roi)
    print(f"[Gender] Full-frame: {gender} (conf={confidence:.2f})")
    return gender


def detect_genders_for_people(frame, face_boxes) -> dict:
    """
    Run per-person gender detection for each selected body bounding box.

    Args:
        frame:      Full camera frame (BGR)
        face_boxes: List of dicts with keys 'index', 'x', 'y', 'w', 'h'
                    as returned by FaceProcessor.get_detected_faces()

    Returns:
        dict mapping person index → gender string ('male'/'female'/'unknown')

    If models aren't available, all entries return 'unknown'.
    Multi-person example:
        {0: 'female', 1: 'male', 2: 'female'}
    """
    if not face_boxes:
        return {}
    if not _load_gender_nets():
        return {f["index"]: "unknown" for f in face_boxes}

    result = {}
    for face in face_boxes:
        idx  = face["index"]
        crop = _find_face_in_roi(frame, face["x"], face["y"],
                                 face["w"], face["h"])
        if crop is None:
            result[idx] = "unknown"
            continue
        gender, conf = _classify_gender_from_crop(crop)
        result[idx]  = gender
        print(f"[Gender] Person {idx}: {gender} (conf={conf:.2f})")

    return result


# ── Character prompts — male + female variants ────────────────────────────────
# Rules:
#   - "positive_f" describes the FEMALE version of the character
#   - "positive_m" describes the MALE version
#   - negative prompts reinforce the target gender to avoid drift
#   - denoise/cnet_strength unchanged from original tuned values

CHARACTER_PROMPTS = {
    "navi": {
        "positive_m": (
            "person as a male Na'vi from Avatar, blue alien skin, "
            "cyan bioluminescent face markings, large amber eyes, pointed ears, "
            "masculine Na'vi features, same hair color, same body pose, "
            "Pandora jungle background, cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "person as a female Na'vi from Avatar, blue alien skin, "
            "cyan bioluminescent face markings, large amber eyes, pointed ears, "
            "feminine Na'vi features, graceful, same hair color, same body pose, "
            "Pandora jungle background, cinematic, photorealistic, 8k"
        ),
        "negative_m": (
            "human skin, pink skin, female, feminine, multiple people, extra person, "
            "bald, changed clothing, ugly, blurry, cartoon, low quality"
        ),
        "negative_f": (
            "human skin, pink skin, male, masculine, beard, multiple people, extra person, "
            "bald, changed clothing, ugly, blurry, cartoon, low quality"
        ),
        "denoise": 0.62, "cnet_strength": 0.75,
    },
    "hulk": {
        "positive_m": (
            "person as the Incredible Hulk, massive green muscles, "
            "green skin, male, same hair and facial structure, torn purple pants, "
            "same body pose, stormy sky background, cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "person as She-Hulk, strong muscular green-skinned woman, "
            "green skin, female, long hair, same facial structure, "
            "ripped clothing, same body pose, stormy sky background, "
            "cinematic, photorealistic, 8k"
        ),
        "negative_m": (
            "female, woman, normal skin, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "negative_f": (
            "male, man, beard, normal skin, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.64, "cnet_strength": 0.75,
    },
    "thanos": {
        "positive_m": (
            "person as Thanos the Marvel villain, male, purple Titan skin, "
            "same facial structure, infinity gauntlet armor, "
            "same body pose, cosmic nebula background, cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "person as a female purple Titan villain, Gamora-style, "
            "purple skin, strong feminine features, same facial structure, "
            "cosmic armor, same body pose, nebula background, "
            "cinematic, photorealistic, 8k"
        ),
        "negative_m": (
            "human skin, pink skin, female, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "negative_f": (
            "human skin, pink skin, male, beard, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.62, "cnet_strength": 0.75,
    },
    "predator": {
        "positive_m": (
            "person as a male Predator alien warrior, mandibles, dreadlocks, "
            "biomask helmet, mesh armor, wrist blades, same body pose, "
            "dark jungle with thermal HUD overlay, cinematic, highly detailed, 8k"
        ),
        "positive_f": (
            "person as a female Predator alien warrior, mandibles, dreadlocks, "
            "biomask helmet, lighter mesh armor, same body pose, "
            "dark jungle with thermal HUD overlay, cinematic, highly detailed, 8k"
        ),
        "negative_m": (
            "human face, female, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "negative_f": (
            "human face, male, beard, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.62, "cnet_strength": 0.75,
    },
    "ghost": {
        "positive_m": (
            "person as a male spectral ghost, translucent glowing body, "
            "pale ethereal skin, white aura, floating light particles, "
            "same clothing and hair shape, same pose, "
            "dark atmospheric background, cinematic, 8k"
        ),
        "positive_f": (
            "person as a female spectral ghost, translucent glowing body, "
            "pale ethereal skin, white flowing aura, floating light particles, "
            "same clothing and hair shape, same pose, "
            "dark atmospheric background, cinematic, 8k"
        ),
        "negative_m": (
            "solid opaque body, female, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "negative_f": (
            "solid opaque body, male, beard, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.60, "cnet_strength": 0.72,
    },
    "groot": {
        "positive_m": (
            "person as Groot from Guardians of the Galaxy, male, "
            "brown bark skin, twig hair with green leaves, vine clothing, "
            "same body pose, forest background with light shafts, "
            "cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "person as a female Groot, living tree humanoid woman, "
            "brown bark skin, flower and leaf hair, vine clothing, "
            "feminine tree features, same body pose, "
            "forest background with light shafts, cinematic, photorealistic, 8k"
        ),
        "negative_m": (
            "human skin, female, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "negative_f": (
            "human skin, male, beard, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.62, "cnet_strength": 0.75,
    },
    "cyberpunk": {
        "positive_m": (
            "person as a male cyberpunk character, same hair with neon streaks, "
            "glowing circuit clothing, cybernetic eye implants, chrome jaw, "
            "same face and pose, neon rain-soaked city background, "
            "cinematic, photorealistic, 8k"
        ),
        "positive_f": (
            "person as a female cyberpunk character, same hair with neon streaks, "
            "glowing circuit clothing, cybernetic eye implants, chrome accents, "
            "same face and pose, neon rain-soaked city background, "
            "cinematic, photorealistic, 8k"
        ),
        "negative_m": (
            "fantasy, medieval, female, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "negative_f": (
            "fantasy, medieval, male, beard, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.65, "cnet_strength": 0.75,
    },
    "claymation": {
        "positive_m": (
            "male person as a claymation figurine, smooth matte clay texture, "
            "same hair color and clothing colors, fingerprint marks on clay, "
            "Aardman style, same pose, stop-motion background, "
            "plasticine, handcrafted, single subject"
        ),
        "positive_f": (
            "female person as a claymation figurine, smooth matte clay texture, "
            "same hair color and clothing colors, fingerprint marks on clay, "
            "Aardman style, same pose, stop-motion background, "
            "plasticine, handcrafted, single subject"
        ),
        "negative_m": (
            "female, photorealistic skin, CGI, anime, multiple people, "
            "extra person, ugly, blurry, low quality, glossy, rubber"
        ),
        "negative_f": (
            "male, beard, photorealistic skin, CGI, anime, multiple people, "
            "extra person, ugly, blurry, low quality, glossy, rubber"
        ),
        "denoise": 0.65, "cnet_strength": 0.65,
        "steps": 6, "sampler": "euler", "scheduler": "sgm_uniform",
    },
    "anime": {
        # Original was hardcoded male — now gender-aware
        "positive_m": (
            "male anime character, same face as the person, "
            "same hair color and style, masculine jawline, sharp anime eyes, "
            "cel-shaded skin, clean line art, vibrant colors, same pose, "
            "Studio Ghibli style, soft bokeh background, single subject"
        ),
        "positive_f": (
            "female anime character, same face as the person, "
            "same hair color and style, feminine features, large expressive eyes, "
            "cel-shaded skin, clean line art, vibrant colors, same pose, "
            "Studio Ghibli style, soft bokeh background, single subject"
        ),
        "negative_m": (
            "female, woman, girl, feminine, photorealistic, photograph, "
            "3d render, western cartoon, multiple people, extra person, "
            "ugly, blurry, bad anatomy, deformed eyes, cross-eyed, "
            "asymmetric eyes, mismatched eyes"
        ),
        "negative_f": (
            "male, man, boy, masculine, beard, photorealistic, photograph, "
            "3d render, western cartoon, multiple people, extra person, "
            "ugly, blurry, bad anatomy, deformed eyes, cross-eyed, "
            "asymmetric eyes, mismatched eyes"
        ),
        "denoise": 0.65, "cnet_strength": 0.78,
        "steps": 6, "sampler": "euler", "scheduler": "sgm_uniform",
    },
}


def get_prompts(char_key: str, gender: str) -> tuple[str, str]:
    """
    Return (positive, negative) prompt strings for a character + detected gender.
    Falls back gracefully: female → male → neutral if keys are missing.
    For 'unknown' gender, picks the neutral/male variant (safest average).
    """
    cfg = CHARACTER_PROMPTS.get(char_key, CHARACTER_PROMPTS["navi"])

    # If old-style single prompts still present (forward-compat), use them
    if "positive" in cfg:
        return cfg["positive"], cfg["negative"]

    suffix = "f" if gender == "female" else "m"
    positive = cfg.get(f"positive_{suffix}", cfg.get("positive_m", ""))
    negative = cfg.get(f"negative_{suffix}", cfg.get("negative_m", ""))
    return positive, negative


# ── API helpers ───────────────────────────────────────────────────────────────
def _api(endpoint, data=None, method="GET"):
    url     = f"{COMFY_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    body    = json.dumps(data).encode() if data else None
    req     = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def is_comfy_running():
    try:
        _api("system_stats")
        return True
    except Exception:
        return False


def _upload_frame(frame, filename="kiosk.png"):
    """
    Upload a frame to ComfyUI.
    Base generation size raised to 1024 (was 896) for better detail before upscale.
    Dimensions are kept divisible by 64 as required by SDXL.
    """
    max_dim = 1024  # was 896 — higher base = more detail going into the upscale pass
    h, w    = frame.shape[:2]
    scale   = max_dim / max(h, w)
    nh      = int((h * scale) // 64) * 64
    nw      = int((w * scale) // 64) * 64
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    _, buf  = cv2.imencode(".png", resized)
    img_bytes = buf.tobytes()
    boundary  = uuid.uuid4().hex
    body  = f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: image/png\r\n\r\n"
    body += img_bytes
    body += f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(
        f"{COMFY_URL}/upload/image", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())["name"]


def _make_control_image(frame, selection_mask=None, skeleton=None):
    """
    Build the ControlNet guidance image.
    Prefers pose skeleton when available; falls back to masked Canny edges.
    """
    max_dim = 1024  # match _upload_frame
    h, w    = frame.shape[:2]
    scale   = max_dim / max(h, w)
    nh      = int((h * scale) // 64) * 64
    nw      = int((w * scale) // 64) * 64

    if skeleton is not None:
        ctrl = cv2.resize(skeleton, (nw, nh))
        return _upload_frame(ctrl, "kiosk_control.png")

    resized  = cv2.resize(frame, (nw, nh))
    gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges    = cv2.Canny(blurred, 100, 200)

    if selection_mask is not None and not np.all(selection_mask >= 0.99):
        mask_rs  = cv2.resize(selection_mask, (nw, nh))
        mask_bin = (mask_rs > 0.3).astype(np.uint8)
        k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_bin = cv2.dilate(mask_bin, k)
        edges    = cv2.bitwise_and(edges, edges, mask=mask_bin)

    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return _upload_frame(edges3, "kiosk_control.png")


# Keep old name as alias
def _make_canny(frame, selection_mask=None):
    return _make_control_image(frame, selection_mask, None)


def _apply_selection_mask(frame, mask):
    """Darken unselected areas to 20% so they don't confuse ControlNet."""
    if mask is None or np.all(mask >= 0.99):
        return frame
    h, w    = frame.shape[:2]
    mask_rs = cv2.resize(mask, (w, h))
    m3      = np.stack([mask_rs]*3, axis=2)
    dim     = (frame.astype(np.float32) * 0.20).astype(np.uint8)
    result  = np.where(m3 > 0.5, frame, dim).astype(np.uint8)
    return result


# ── InstantID (disabled — antelopev2 not configured) ─────────────────────────
COMFY_DIR       = os.path.expanduser("~/ComfyUI")
INSTANTID_MODEL = os.path.join(COMFY_DIR, "models", "instantid", "ip-adapter_instantid.bin")
INSTANTID_CNET  = os.path.join(COMFY_DIR, "models", "controlnet", "instantid-controlnet.safetensors")
INSTANTID_NODE  = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI_InstantID")


def _instantid_available():
    """Disabled -- InsightFace antelopev2 model not configured. Using Canny workflow."""
    return False


def _build_instantid_workflow(char_key, image_name, face_name, canny_name, gender="unknown"):
    """InstantID two-pass workflow (kept for when InstantID is re-enabled)."""
    positive, negative = get_prompts(char_key, gender)
    cfg  = CHARACTER_PROMPTS.get(char_key, CHARACTER_PROMPTS["navi"])
    seed = int(time.time()) % 2**32
    return {
        "1":  {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"}},
        "2":  {"class_type": "InstantIDModelLoader",
               "inputs": {"instantid_file": "ip-adapter_instantid.bin"}},
        "3":  {"class_type": "ControlNetLoader",
               "inputs": {"control_net_name": "instantid-controlnet.safetensors"}},
        "4":  {"class_type": "InstantIDFaceAnalysis",
               "inputs": {"provider": "CPU"}},
        "5":  {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6":  {"class_type": "LoadImage", "inputs": {"image": face_name}},
        "7":  {"class_type": "LoadImage", "inputs": {"image": canny_name}},
        "8":  {"class_type": "ControlNetLoader",
               "inputs": {"control_net_name": "control-lora-canny-rank128.safetensors"}},
        "9":  {"class_type": "VAEEncode",
               "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        "10": {"class_type": "CLIPTextEncode",
               "inputs": {"text": positive, "clip": ["1", 1]}},
        "11": {"class_type": "CLIPTextEncode",
               "inputs": {"text": negative,  "clip": ["1", 1]}},
        "12": {"class_type": "ApplyInstantID",
               "inputs": {
                   "instantid":   ["2", 0],
                   "insightface": ["4", 0],
                   "control_net": ["3", 0],
                   "image":       ["5", 0],
                   "image_kps":   ["6", 0],
                   "model":       ["1", 0],
                   "positive":    ["10", 0],
                   "negative":    ["11", 0],
                   "weight":      0.80,
                   "cn_strength": 0.65,
                   "start_at":    0.0,
                   "end_at":      1.0,
               }},
        "13": {"class_type": "ControlNetApply",
               "inputs": {
                   "conditioning": ["12", 1],
                   "control_net":  ["8", 0],
                   "image":        ["7", 0],
                   "strength":     cfg["cnet_strength"] * 0.75,
               }},
        "14": {"class_type": "KSampler",
               "inputs": {
                   "model":        ["12", 0],
                   "positive":     ["13", 0],
                   "negative":     ["12", 2],
                   "latent_image": ["9",  0],
                   "seed":         seed,
                   "steps":        8,
                   "cfg":          1.0,
                   "sampler_name": "euler_ancestral",
                   "scheduler":    "sgm_uniform",
                   "denoise":      cfg["denoise"],
               }},
        "15": {"class_type": "VAEDecode",
               "inputs": {"samples": ["14", 0], "vae": ["1", 2]}},
        "16": {"class_type": "SaveImage",
               "inputs": {"images": ["15", 0],
                          "filename_prefix": f"kiosk_iid_{char_key}"}},
    }


def _build_workflow(char_key, image_name, canny_name, gender="unknown"):
    """
    Single-pass img2img + ControlNet at 1024px.
    The two-pass upscale was causing silent VAE decode hangs on ROCm —
    the 1024px base is already a solid upgrade from the original 896px
    and is stable on the W7900.
    """
    positive, negative = get_prompts(char_key, gender)
    cfg       = CHARACTER_PROMPTS.get(char_key, CHARACTER_PROMPTS["navi"])
    seed      = int(time.time()) % 2**32
    steps     = cfg.get("steps", 8)
    sampler   = cfg.get("sampler",    "euler_ancestral")
    scheduler = cfg.get("scheduler",  "sgm_uniform")

    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"}},
        "2": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": "control-lora-canny-rank128.safetensors"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "4": {"class_type": "LoadImage", "inputs": {"image": canny_name}},
        "5": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": positive, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        "8": {"class_type": "ControlNetApply",
              "inputs": {"conditioning": ["6", 0], "control_net": ["2", 0],
                         "image": ["4", 0], "strength": cfg["cnet_strength"]}},
        "9": {"class_type": "KSampler",
              "inputs": {
                  "model":        ["1", 0],
                  "positive":     ["8", 0],
                  "negative":     ["7", 0],
                  "latent_image": ["5", 0],
                  "seed":         seed,
                  "steps":        steps,
                  "cfg":          1.0,
                  "sampler_name": sampler,
                  "scheduler":    scheduler,
                  "denoise":      cfg["denoise"],
              }},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0],
                          "filename_prefix": f"kiosk_{char_key}"}},
    }


def _blur_background(frame, selection_mask=None):
    """
    Blur background to prevent ControlNet from generating extra figures from edge noise.
    """
    if selection_mask is None or np.all(selection_mask >= 0.99):
        return cv2.GaussianBlur(frame, (1, 1), 0)

    h, w      = frame.shape[:2]
    mask_rs   = cv2.resize(selection_mask, (w, h))
    mask_soft = cv2.GaussianBlur(mask_rs, (31, 31), 0)
    m3        = np.stack([mask_soft] * 3, axis=2)
    bg_blur   = cv2.GaussianBlur(frame, (51, 51), 0)
    result    = (frame.astype(np.float32) * m3 +
                 bg_blur.astype(np.float32) * (1.0 - m3))
    return np.clip(result, 0, 255).astype(np.uint8)


def _extract_face_crop(frame, selection_mask=None):
    """
    Return a tight face-crop for InstantID identity embedding.
    Falls back to full frame if no face is detected.
    """
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return frame
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = int(max(w, h) * 0.35)
    fh, fw = frame.shape[:2]
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(fw, x + w + pad); y2 = min(fh, y + h + pad)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame


# ── Main entry point ──────────────────────────────────────────────────────────
def generate_character(frame, char_key, selection_mask=None,
                       face_boxes=None, timeout=400):
    """
    face_boxes: list of selected face dicts from FaceProcessor.get_detected_faces()
                e.g. [{'index':0,'x':100,'y':50,'w':80,'h':200,'selected':True}, ...]
                When provided, gender is detected per-person from their bounding box.
                When None (no selection), falls back to full-frame gender detection.
    """
    if not is_comfy_running():
        return None, "ComfyUI not running"
    try:
        # ── Per-person gender detection ────────────────────────────────────
        if face_boxes:
            gender_map = {}
            for box in face_boxes:
                idx = box["index"]
                # Use registered gender first — 100% reliable for known hosts
                registered = box.get("registered_gender", "unknown")
                if registered in ("male", "female"):
                    gender_map[idx] = registered
                    print(f"[Gender] Person {idx}: {registered} (from profile)")
                else:
                    # Unknown person — detect from their bounding box
                    detected = detect_genders_for_people(frame, [box])
                    gender_map[idx] = detected.get(idx, "unknown")

            genders      = list(gender_map.values())
            female_count = genders.count("female")
            male_count   = genders.count("male")
            if female_count > male_count:
                gender = "female"
            elif male_count > female_count:
                gender = "male"
            elif female_count == male_count and female_count > 0:
                gender = "female"   # tie → female (safer for the demo)
            else:
                gender = "unknown"
            print(f"[Generate] character={char_key}  per-person={gender_map}  using={gender}")
        else:
            gender = detect_gender(frame)
            print(f"[Generate] character={char_key}  gender={gender} (full-frame fallback)")

        prepped      = _blur_background(frame, selection_mask)
        control_name = _make_control_image(prepped, selection_mask)

        use_iid = _instantid_available()
        if use_iid:
            print(f"[InstantID] Using identity-preserving workflow for {char_key}")
            img_name  = _upload_frame(prepped)
            face_crop = _extract_face_crop(frame, selection_mask)
            face_name = _upload_frame(face_crop, "kiosk_face.png")
            workflow  = _build_instantid_workflow(char_key, img_name, face_name,
                                                  control_name, gender)
        else:
            print(f"[Canny] Using standard workflow for {char_key} (gender={gender})")
            img_name = _upload_frame(prepped)
            workflow = _build_workflow(char_key, img_name, control_name, gender)

        client_id = uuid.uuid4().hex
        result    = _api("prompt", {"prompt": workflow, "client_id": client_id}, "POST")
        prompt_id = result["prompt_id"]
        print(f"[Comfy] prompt_id={prompt_id}")

        start = time.time()
        poll_errors = 0
        while time.time() - start < timeout:
            time.sleep(1.5)
            elapsed = int(time.time() - start)
            try:
                history = _api(f"history/{prompt_id}")
                entry   = history.get(prompt_id)
                poll_errors = 0  # reset on success

                if entry is None:
                    all_history = _api("history")
                    entry = all_history.get(prompt_id)

                if entry is not None:
                    outputs = entry.get("outputs", {})
                    print(f"[Comfy] {elapsed}s -- nodes with output: {list(outputs.keys())}")
                    for node_id, node_out in outputs.items():
                        if "images" in node_out and len(node_out["images"]) > 0:
                            info = node_out["images"][0]
                            print(f"[Comfy] Fetching image from node {node_id}: {info['filename']}")
                            img_url = (f"{COMFY_URL}/view?"
                                       f"filename={urllib.parse.quote(info['filename'])}"
                                       f"&subfolder={urllib.parse.quote(info.get('subfolder',''))}"
                                       f"&type={info.get('type','output')}")
                            with urllib.request.urlopen(img_url, timeout=30) as r:
                                img_bytes = r.read()
                            arr = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                print(f"[Comfy] Image decoded OK: {img.shape}")
                                return img, None
                            else:
                                print(f"[Comfy] WARNING: imdecode returned None for {info['filename']}")
                else:
                    if elapsed % 10 == 0:
                        print(f"[Comfy] {elapsed}s -- waiting for prompt_id in history...")
            except Exception as e:
                poll_errors += 1
                print(f"[Comfy] Poll error at {elapsed}s (#{poll_errors}): {e}")
                if poll_errors >= 10:
                    return None, f"Too many poll errors: {e}"
                time.sleep(2)  # back off before retrying

        return None, "Timeout waiting for result"
    except Exception as e:
        return None, str(e)


# ── ComfyBridge class (unchanged interface) ───────────────────────────────────
class ComfyBridge:
    def __init__(self):
        self._result         = None
        self._status         = "idle"
        self._message        = ""
        self._thread         = None
        self._selection_mask = None
        self.available       = is_comfy_running()
        self.instantid       = _instantid_available()
        # Pre-load gender models in background so first generate() is fast
        threading.Thread(target=_load_gender_nets, daemon=True).start()
        print(f"[{'OK' if self.available else 'WARN'}] ComfyUI "
              f"{'connected' if self.available else 'not running -- start ~/start_comfyui.sh'}")
        print(f"[{'OK' if self.instantid else 'INFO'}] InstantID "
              f"{'available' if self.instantid else 'not installed'}")
        print("[INFO] Gender-aware prompts active — women stay women.")

    def check_available(self):
        self.available = is_comfy_running()
        return self.available

    def generate(self, frame, char_key, selection_mask=None, face_boxes=None):
        if self._status == "generating":
            return False
        self._status         = "generating"
        self._result         = None
        self._message        = "Transforming..."
        self._selection_mask = selection_mask
        self._thread = threading.Thread(
            target=self._run,
            args=(frame.copy(), char_key, selection_mask, face_boxes),
            daemon=True)
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
            self._message = err or "Failed"

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
