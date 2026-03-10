"""
comfy_bridge.py -- ComfyUI API bridge for AMD Adapt Kiosk
img2img + ControlNet Canny. Supports per-person selection mask.
"""

import json
import os
import uuid
import time
import urllib.request
import urllib.parse
import threading
import cv2
import numpy as np

COMFY_URL = "http://127.0.0.1:8188"

# Prompts instruct transformation of the subject, not generation of new characters.
# denoise 0.72-0.78 is needed to actually transform the person's appearance.
# cnet_strength 0.85 preserves pose without over-constraining the transformation.
# Prompts use img2img transformation approach:
# - Keep denoise 0.68-0.72 so original hair, face structure and clothing silhouette are preserved
# - Explicitly instruct the model to TRANSFORM the subject, not replace them
# - Negative prompts prevent hallucinated extra people from background noise
CHARACTER_PROMPTS = {
    "navi": {
        "positive": (
            "person as a Na'vi from Avatar, blue alien skin, "
            "cyan face markings, large amber eyes, pointed ears, "
            "same hair color and clothing silhouette, same body pose, "
            "Pandora jungle background, cinematic, photorealistic, 8k"
        ),
        "negative": (
            "human skin, pink skin, multiple people, extra person, "
            "bald, changed clothing, ugly, blurry, cartoon, low quality"
        ),
        "denoise": 0.62, "cnet_strength": 0.75,
    },
    "hulk": {
        "positive": (
            "person as the Incredible Hulk, massive green muscles, "
            "green skin, same hair and facial structure, torn purple pants, "
            "same body pose, stormy sky background, "
            "cinematic, photorealistic, 8k"
        ),
        "negative": (
            "normal skin, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.64, "cnet_strength": 0.75,
    },
    "thanos": {
        "positive": (
            "person as Thanos the Marvel villain, purple Titan skin, "
            "same facial structure, infinity gauntlet armor, "
            "same body pose, cosmic nebula background, "
            "cinematic, photorealistic, 8k"
        ),
        "negative": (
            "human skin, pink skin, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.62, "cnet_strength": 0.75,
    },
    "predator": {
        "positive": (
            "person as a Predator alien warrior, mandibles, dreadlocks, "
            "biomask helmet, mesh armor, wrist blades, same body pose, "
            "dark jungle with thermal HUD overlay, "
            "cinematic, highly detailed, 8k"
        ),
        "negative": (
            "human face, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.62, "cnet_strength": 0.75,
    },
    "ghost": {
        "positive": (
            "person as a spectral ghost, translucent glowing body, "
            "pale ethereal skin, white aura, floating light particles, "
            "same clothing and hair shape, same pose, "
            "dark atmospheric background, cinematic, 8k"
        ),
        "negative": (
            "solid opaque body, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.60, "cnet_strength": 0.72,
    },
    "groot": {
        "positive": (
            "person as Groot from Guardians of the Galaxy, "
            "brown bark skin, twig hair with green leaves, vine clothing, "
            "same body pose, forest background with light shafts, "
            "cinematic, photorealistic, 8k"
        ),
        "negative": (
            "human skin, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.62, "cnet_strength": 0.75,
    },
    "cyberpunk": {
        "positive": (
            "person as a cyberpunk character, same hair with neon streaks, "
            "glowing circuit clothing, cybernetic eye implants, chrome jaw, "
            "same face and pose, neon rain-soaked city background, "
            "cinematic, photorealistic, 8k"
        ),
        "negative": (
            "fantasy, medieval, multiple people, extra person, "
            "ugly, blurry, cartoon, low quality, duplicate"
        ),
        "denoise": 0.65, "cnet_strength": 0.75,
    },
    "claymation": {
        "positive": (
            "person as a claymation figurine, smooth matte clay texture, "
            "same hair color and clothing colors, fingerprint marks on clay, "
            "Aardman style, same pose, stop-motion background, "
            "plasticine, handcrafted, single subject"
        ),
        "negative": (
            "photorealistic skin, CGI, anime, multiple people, "
            "extra person, ugly, blurry, low quality, glossy, rubber"
        ),
        "denoise": 0.65, "cnet_strength": 0.65, "steps": 6, "sampler": "euler", "scheduler": "sgm_uniform",
    },
    "anime": {
        "positive": (
            "male anime character, man, same face as the person, "
            "same hair color and style, masculine jawline, "
            "sharp anime eyes, cel-shaded skin, clean line art, "
            "vibrant colors, same pose, Studio Ghibli style, "
            "soft bokeh background, single subject"
        ),
        "negative": (
            "female, woman, girl, feminine, photorealistic, photograph, "
            "3d render, western cartoon, multiple people, extra person, "
            "ugly, blurry, bad anatomy, deformed eyes, cross-eyed, "
            "asymmetric eyes, mismatched eyes"
        ),
        "denoise": 0.65, "cnet_strength": 0.78, "steps": 6, "sampler": "euler", "scheduler": "sgm_uniform",
    },
}

def _api(endpoint, data=None, method="GET"):
    url     = f"{COMFY_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    body    = json.dumps(data).encode() if data else None
    req     = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def is_comfy_running():
    try:
        _api("system_stats")
        return True
    except Exception:
        return False

def _upload_frame(frame, filename="kiosk.png"):
    max_dim = 896
    h, w    = frame.shape[:2]
    scale   = max_dim / max(h, w)
    nh      = int((h * scale) // 64) * 64
    nw      = int((w * scale) // 64) * 64
    resized = cv2.resize(frame, (nw, nh))
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
    Build the ControlNet guidance image:
    - If skeleton (MediaPipe pose) is available: use it -- best pose fidelity
    - Otherwise: fall back to Canny edges masked to person silhouette
    """
    max_dim = 896
    h, w    = frame.shape[:2]
    scale   = max_dim / max(h, w)
    nh      = int((h * scale) // 64) * 64
    nw      = int((w * scale) // 64) * 64

    if skeleton is not None:
        # Use pose skeleton -- white lines on black, perfectly captures body pose
        ctrl = cv2.resize(skeleton, (nw, nh))
        return _upload_frame(ctrl, "kiosk_control.png")

    # Fallback: Canny edges
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
    """
    Crop the frame so only selected people are prominent.
    Unselected areas are darkened to 20% brightness.
    If mask is all-ones (no selection), return frame unchanged.
    """
    if mask is None or np.all(mask >= 0.99):
        return frame
    h, w    = frame.shape[:2]
    mask_rs = cv2.resize(mask, (w, h))
    m3      = np.stack([mask_rs]*3, axis=2)
    dim     = (frame.astype(np.float32) * 0.20).astype(np.uint8)
    result  = np.where(m3 > 0.5, frame, dim).astype(np.uint8)
    return result

# ── InstantID model paths ──────────────────────────────────────────────────────
COMFY_DIR       = os.path.expanduser("~/ComfyUI")
INSTANTID_MODEL = os.path.join(COMFY_DIR, "models", "instantid", "ip-adapter_instantid.bin")
INSTANTID_CNET  = os.path.join(COMFY_DIR, "models", "controlnet", "instantid-controlnet.safetensors")
INSTANTID_NODE  = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI_InstantID")

def _instantid_available():
    """True only when all three InstantID components are present on disk."""
    return (
        os.path.exists(INSTANTID_MODEL) and
        os.path.exists(INSTANTID_CNET)  and
        os.path.exists(INSTANTID_NODE)
    )


def _build_instantid_workflow(char_key, image_name, face_name, canny_name):
    """
    InstantID two-pass workflow.

    Layers the face identity embedding (via InsightFace + ip-adapter_instantid)
    on top of the existing Canny structural guidance. The result is that the
    generated character genuinely looks like the specific person standing at
    the kiosk, not just a generic member of that species/style.

    Node map:
      1  CheckpointLoaderSimple  SDXL-Turbo fp16
      2  InstantIDModelLoader    ip-adapter_instantid.bin
      3  ControlNetLoader        instantid-controlnet.safetensors  (face keypoints)
      4  InstantIDFaceAnalysis   InsightFace buffalo_l (runs on CPU -- stable on ROCm)
      5  LoadImage               full source frame  (structure reference)
      6  LoadImage               face-crop reference (identity reference)
      7  LoadImage               canny edge map
      8  ControlNetLoader        control-lora-canny-rank128 (pose/structure layer)
      9  VAEEncode               encode source frame
      10 CLIPTextEncode positive
      11 CLIPTextEncode negative
      12 ApplyInstantID          inject face identity → patched model + conditionings
      13 ControlNetApply         add Canny pose on top of InstantID conditionings
      14 KSampler Pass 1         768px, 10 steps
      15 LatentUpscale           bislerp 768→1024
      16 KSampler Pass 2         1024px, 6 steps, denoise 0.35
      17 VAEDecode
      18 SaveImage
    """
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
               "inputs": {"text": cfg["positive"], "clip": ["1", 1]}},
        "11": {"class_type": "CLIPTextEncode",
               "inputs": {"text": cfg["negative"],  "clip": ["1", 1]}},
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
                   "weight":      0.80,        # fixed: was ip_weight
                   "cn_strength": 0.65,
                   "start_at":    0.0,
                   "end_at":      1.0,
               }},
        # ApplyInstantID outputs: [0]=MODEL, [1]=positive, [2]=negative
        "13": {"class_type": "ControlNetApply",
               "inputs": {
                   "conditioning": ["12", 1],          # positive conditioning
                   "control_net":  ["8", 0],
                   "image":        ["7", 0],
                   "strength":     cfg["cnet_strength"] * 0.75,
               }},
        "14": {"class_type": "KSampler",
               "inputs": {
                   "model":        ["12", 0],           # patched MODEL
                   "positive":     ["13", 0],
                   "negative":     ["12", 2],           # negative conditioning
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


def _build_workflow(char_key, image_name, canny_name):
    """Single-pass img2img + ControlNet at 896px using SDXL base 1.0."""
    cfg       = CHARACTER_PROMPTS.get(char_key, CHARACTER_PROMPTS["navi"])
    seed      = int(time.time()) % 2**32
    steps     = cfg.get("steps", 8)
    # SDXL-Turbo uses euler_ancestral + sgm_uniform
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
              "inputs": {"text": cfg["positive"], "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": cfg["negative"], "clip": ["1", 1]}},
        "8": {"class_type": "ControlNetApply",
              "inputs": {"conditioning": ["6", 0], "control_net": ["2", 0],
                         "image": ["4", 0], "strength": cfg["cnet_strength"]}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["7", 0],
                         "latent_image": ["5", 0], "seed": seed,
                         "steps": steps, "cfg": 1.0,
                         "sampler_name": sampler, "scheduler": scheduler,
                         "denoise": cfg["denoise"]}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": f"kiosk_{char_key}"}},
    }

def _blur_background(frame, selection_mask=None):
    """
    Heavily blur the background, keep person sharp.
    Prevents background edges from confusing ControlNet into generating extra figures.
    """
    if selection_mask is None or np.all(selection_mask >= 0.99):
        # No specific selection -- use rembg-style simple background blur
        # Just blur the whole image lightly to reduce edge noise
        return cv2.GaussianBlur(frame, (1, 1), 0)

    h, w     = frame.shape[:2]
    mask_rs  = cv2.resize(selection_mask, (w, h))

    # Soften mask edges
    mask_soft = cv2.GaussianBlur(mask_rs, (31, 31), 0)
    m3        = np.stack([mask_soft] * 3, axis=2)

    # Heavily blur background
    bg_blur = cv2.GaussianBlur(frame, (51, 51), 0)

    # Composite: sharp person + blurred background
    result = (frame.astype(np.float32) * m3 +
              bg_blur.astype(np.float32) * (1.0 - m3))
    return np.clip(result, 0, 255).astype(np.uint8)


def _extract_face_crop(frame, selection_mask=None):
    """
    Return a tight face-crop from the frame for InstantID identity embedding.
    Falls back to the full frame if no face is detected.
    Uses OpenCV Haar Cascade -- no extra dependencies.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return frame  # no face detected -- use full frame
    # Pick the largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = int(max(w, h) * 0.35)
    fh, fw = frame.shape[:2]
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(fw, x + w + pad); y2 = min(fh, y + h + pad)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame


def generate_character(frame, char_key, selection_mask=None, timeout=400):
    if not is_comfy_running():
        return None, "ComfyUI not running"
    try:
        prepped      = _blur_background(frame, selection_mask)
        control_name = _make_control_image(prepped, selection_mask)

        use_iid = _instantid_available()
        if use_iid:
            print(f"[InstantID] Using identity-preserving workflow for {char_key}")
            img_name  = _upload_frame(prepped)
            face_crop = _extract_face_crop(frame, selection_mask)  # use original (sharp) for identity
            face_name = _upload_frame(face_crop, "kiosk_face.png")
            workflow  = _build_instantid_workflow(char_key, img_name, face_name, control_name)
        else:
            print(f"[Canny] Using standard workflow for {char_key} (InstantID not installed)")
            img_name = _upload_frame(prepped)
            workflow = _build_workflow(char_key, img_name, control_name)

        client_id = uuid.uuid4().hex
        result    = _api("prompt", {"prompt": workflow, "client_id": client_id}, "POST")
        prompt_id = result["prompt_id"]
        print(f"[Comfy] prompt_id={prompt_id}")

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(1.0)
            elapsed = int(time.time() - start)
            try:
                # Primary: fetch this specific prompt's history
                history = _api(f"history/{prompt_id}")
                entry = history.get(prompt_id)

                # Fallback: search full history if specific entry missing
                if entry is None:
                    all_history = _api("history")
                    entry = all_history.get(prompt_id)

                if entry is not None:
                    outputs = entry.get("outputs", {})
                    print(f"[Comfy] {elapsed}s -- found history, nodes with output: {list(outputs.keys())}")
                    for node_id, node_out in outputs.items():
                        if "images" in node_out and len(node_out["images"]) > 0:
                            info = node_out["images"][0]
                            print(f"[Comfy] Fetching image from node {node_id}: {info['filename']}")
                            img_url = (f"{COMFY_URL}/view?"
                                      f"filename={urllib.parse.quote(info['filename'])}"
                                      f"&subfolder={urllib.parse.quote(info.get('subfolder',''))}"
                                      f"&type={info.get('type','output')}")
                            with urllib.request.urlopen(img_url, timeout=15) as r:
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
                print(f"[Comfy] Poll error at {elapsed}s: {e}")

        return None, "Timeout waiting for result"
    except Exception as e:
        return None, str(e)


class ComfyBridge:
    def __init__(self):
        self._result        = None
        self._status        = "idle"
        self._message       = ""
        self._thread        = None
        self._selection_mask = None
        self.available      = is_comfy_running()
        self.instantid      = _instantid_available()
        print(f"[{'OK' if self.available else 'WARN'}] ComfyUI "
              f"{'connected' if self.available else 'not running -- start ~/start_comfyui.sh'}")
        print(f"[{'OK' if self.instantid else 'INFO'}] InstantID "
              f"{'available -- identity-preserving mode active' if self.instantid else 'not installed -- run install_instantid.sh to enable'}")

    def check_available(self):
        self.available = is_comfy_running()
        return self.available

    def generate(self, frame, char_key, selection_mask=None):
        if self._status == "generating":
            return False
        self._status         = "generating"
        self._result         = None
        self._message        = "Transforming selected people..."
        self._selection_mask = selection_mask
        self._thread = threading.Thread(
            target=self._run, args=(frame.copy(), char_key, selection_mask), daemon=True)
        self._thread.start()
        return True

    def _run(self, frame, char_key, mask):
        img, err = generate_character(frame, char_key, mask)
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
            img           = self._result.copy()
            self._status  = "idle"
            self._result  = None
            return img
        return None

    def reset(self):
        self._status  = "idle"
        self._result  = None
        self._message = ""
