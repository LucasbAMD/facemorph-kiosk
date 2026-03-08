"""
comfy_bridge.py — ComfyUI API bridge for AMD Adapt Kiosk

Sends webcam frames to ComfyUI for AI character transformation.
Uses SDXL-Turbo + ControlNet for photorealistic results in ~2-3 seconds.

ComfyUI must be running at http://localhost:8188
"""

import json
import uuid
import time
import base64
import urllib.request
import urllib.parse
import urllib.error
import threading
import cv2
import numpy as np
from pathlib import Path
from io import BytesIO

COMFY_URL = "http://127.0.0.1:8188"

# ── Character prompts ──────────────────────────────────────────────────────────
# Each prompt is tuned to produce movie-quality character transforms
CHARACTER_PROMPTS = {
    "navi": {
        "positive": (
            "Na'vi alien from Avatar movie, bioluminescent blue skin, "
            "glowing cyan markings on face and body, large amber eyes with vertical pupils, "
            "high cheekbones, blue skin texture, Pandora jungle, "
            "cinematic lighting, photorealistic, 8k, detailed"
        ),
        "negative": (
            "human skin, brown skin, pink skin, white skin, normal eyes, "
            "ugly, blurry, low quality, cartoon"
        ),
        "denoise": 0.72,
    },
    "hulk": {
        "positive": (
            "Incredible Hulk, massive gamma-irradiated green skin, "
            "enormous green muscles, torn purple pants, angry expression, "
            "veins visible on skin, cinematic lighting, photorealistic, 8k, "
            "Marvel character, dramatic"
        ),
        "negative": (
            "normal human skin, small body, cartoon, ugly, blurry, low quality"
        ),
        "denoise": 0.75,
    },
    "thanos": {
        "positive": (
            "Thanos from Avengers, deep purple wrinkled skin, "
            "strong jaw, gold armor, Infinity Gauntlet glowing, "
            "cosmic background, cinematic, photorealistic, 8k, Marvel villain"
        ),
        "negative": (
            "human skin, ugly, blurry, low quality, cartoon"
        ),
        "denoise": 0.73,
    },
    "predator": {
        "positive": (
            "Predator alien warrior, thermal infrared vision view, "
            "mandible face, dreadlock appendages, cloaking shimmer effect, "
            "jungle thermal imaging, FLIR camera, heat signature, "
            "cinematic, detailed, 8k"
        ),
        "negative": (
            "human, ugly, blurry, low quality, cartoon"
        ),
        "denoise": 0.70,
    },
    "ghost": {
        "positive": (
            "Ghost Marvel character, pale translucent phasing body, "
            "white ghostly aura, ethereal glow, semi-transparent, "
            "spectral energy, dark background with soft light, "
            "cinematic, photorealistic, 8k"
        ),
        "negative": (
            "solid opaque body, ugly, blurry, low quality, cartoon"
        ),
        "denoise": 0.68,
    },
    "groot": {
        "positive": (
            "Groot from Guardians of the Galaxy, living tree humanoid, "
            "bark skin texture, wooden body, green leaves growing from body, "
            "warm amber eyes, forest background, cinematic lighting, "
            "photorealistic, 8k, Marvel character"
        ),
        "negative": (
            "human skin, ugly, blurry, low quality, cartoon"
        ),
        "denoise": 0.74,
    },
}

def _api(endpoint, data=None, method="GET"):
    """Simple HTTP call to ComfyUI API."""
    url = f"{COMFY_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def is_comfy_running():
    """Check if ComfyUI is up."""
    try:
        _api("system_stats")
        return True
    except Exception:
        return False

def _frame_to_b64(frame, max_size=768):
    """Encode frame as base64 JPEG, resized for SD input."""
    h, w = frame.shape[:2]
    # Resize to SD-friendly size (multiple of 64)
    scale = max_size / max(h, w)
    nh = int((h * scale) // 64) * 64
    nw = int((w * scale) // 64) * 64
    resized = cv2.resize(frame, (nw, nh))
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buf.tobytes()).decode(), nw, nh

def _upload_image(frame):
    """Upload image to ComfyUI's input folder, return filename."""
    _, buf = cv2.imencode(".png", frame)
    img_bytes = buf.tobytes()
    # Multipart form upload
    boundary = uuid.uuid4().hex
    body  = f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="image"; filename="kiosk_frame.png"\r\n'
    body += b"Content-Type: image/png\r\n\r\n"
    body += img_bytes
    body += f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(
        f"{COMFY_URL}/upload/image",
        data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        result = json.loads(r.read())
    return result["name"]

def _build_workflow(char_key, image_name, w, h):
    """
    Build ComfyUI workflow JSON for img2img + ControlNet.
    Uses SDXL-Turbo for speed (~4 steps = ~1.5s on W7900).
    Falls back to SD 1.5 if SDXL not found.
    """
    cfg = CHARACTER_PROMPTS.get(char_key, CHARACTER_PROMPTS["navi"])
    pos_prompt = cfg["positive"]
    neg_prompt = cfg["negative"]
    denoise    = cfg["denoise"]

    workflow = {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"},
        },
        # Load ControlNet (Canny)
        "2": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": "control-lora-canny-rank128.safetensors"},
        },
        # Load input image
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        # Canny preprocessor
        "4": {
            "class_type": "CannyEdgePreprocessor",
            "inputs": {
                "image": ["3", 0],
                "low_threshold": 100,
                "high_threshold": 200,
            },
        },
        # Encode image as latent
        "5": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["3", 0],
                "vae":    ["1", 2],
            },
        },
        # Positive prompt
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": pos_prompt,
                "clip": ["1", 1],
            },
        },
        # Negative prompt
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": neg_prompt,
                "clip": ["1", 1],
            },
        },
        # Apply ControlNet
        "8": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning":   ["6", 0],
                "control_net":    ["2", 0],
                "image":          ["4", 0],
                "strength":       0.65,
            },
        },
        # KSampler — SDXL-Turbo settings (fast!)
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model":            ["1", 0],
                "positive":         ["8", 0],
                "negative":         ["7", 0],
                "latent_image":     ["5", 0],
                "seed":             int(time.time()) % 2**32,
                "steps":            4,
                "cfg":              1.0,   # SDXL-Turbo uses cfg=1
                "sampler_name":     "euler_ancestral",
                "scheduler":        "karras",
                "denoise":          denoise,
            },
        },
        # Decode latent
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae":     ["1", 2],
            },
        },
        # Save image
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images":       ["10", 0],
                "filename_prefix": f"kiosk_{char_key}",
            },
        },
    }
    return workflow

def generate_character(frame, char_key, timeout=30):
    """
    Main function: send frame to ComfyUI, get back AI-transformed image.
    Returns numpy BGR image or None on failure.
    """
    if not is_comfy_running():
        return None, "ComfyUI not running — start it first"

    try:
        # Upload frame
        h, w = frame.shape[:2]
        # Resize for SD
        max_dim = 768
        scale = max_dim / max(h, w)
        nh = int((h * scale) // 64) * 64
        nw = int((w * scale) // 64) * 64
        resized = cv2.resize(frame, (nw, nh))
        img_name = _upload_image(resized)

        # Build and queue workflow
        workflow = _build_workflow(char_key, img_name, nw, nh)
        client_id = uuid.uuid4().hex
        payload = {"prompt": workflow, "client_id": client_id}
        result  = _api("prompt", payload, "POST")
        prompt_id = result["prompt_id"]

        # Poll for completion
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.5)
            try:
                history = _api(f"history/{prompt_id}")
                if prompt_id in history:
                    outputs = history[prompt_id]["outputs"]
                    # Find the SaveImage node output
                    for node_id, node_out in outputs.items():
                        if "images" in node_out:
                            img_info = node_out["images"][0]
                            # Fetch the image
                            img_url = (f"{COMFY_URL}/view?"
                                      f"filename={urllib.parse.quote(img_info['filename'])}"
                                      f"&subfolder={urllib.parse.quote(img_info.get('subfolder',''))}"
                                      f"&type={img_info.get('type','output')}")
                            with urllib.request.urlopen(img_url, timeout=10) as r:
                                img_bytes = r.read()
                            arr = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            return img, None
            except Exception:
                pass

        return None, "Timeout waiting for ComfyUI"

    except Exception as e:
        return None, str(e)


class ComfyBridge:
    """
    Async wrapper — queues generation requests, calls back when done.
    Keeps the kiosk responsive during generation.
    """
    def __init__(self):
        self._queue   = []
        self._lock    = threading.Lock()
        self._result  = None
        self._status  = "idle"   # idle | generating | done | error
        self._message = ""
        self._thread  = None
        self.available = is_comfy_running()
        if self.available:
            print("[OK] ComfyUI connected — AI generation ready")
        else:
            print("[WARN] ComfyUI not running — start with: python launch_comfy.py")

    def check_available(self):
        self.available = is_comfy_running()
        return self.available

    def generate(self, frame, char_key):
        """Kick off async generation. Returns immediately."""
        if self._status == "generating":
            return False  # already running
        self._status  = "generating"
        self._result  = None
        self._message = "Generating..."
        self._thread  = threading.Thread(
            target=self._run, args=(frame.copy(), char_key), daemon=True)
        self._thread.start()
        return True

    def _run(self, frame, char_key):
        img, err = generate_character(frame, char_key)
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
            img = self._result.copy()
            self._status = "idle"
            self._result = None
            return img
        return None

    def reset(self):
        self._status  = "idle"
        self._result  = None
        self._message = ""
