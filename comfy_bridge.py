"""
comfy_bridge.py — ComfyUI API bridge for AMD Adapt Kiosk
img2img + ControlNet Canny. Supports per-person selection mask.
"""

import json
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
CHARACTER_PROMPTS = {
    "navi":     {
        "positive": "full body portrait of a single Na'vi from Avatar movie, blue skin, bioluminescent cyan stripes on face and body, large amber eyes, Pandora jungle background with glowing plants, cinematic lighting, photorealistic, 8k, one person only",
        "negative": "human skin, multiple people, two people, three people, ugly, blurry, cartoon, duplicate, extra person, background figures, human face",
        "denoise": 0.75, "cnet_strength": 0.85,
    },
    "hulk":     {
        "positive": "full body portrait of the Incredible Hulk, massive green muscles, green skin all over body, torn purple pants, single subject, angry expression, dramatic sky background, cinematic, photorealistic, 8k",
        "negative": "normal skin, multiple people, two people, cartoon, ugly, blurry, duplicate, extra person",
        "denoise": 0.76, "cnet_strength": 0.85,
    },
    "thanos":   {
        "positive": "full body portrait of Thanos Marvel villain, purple textured skin, gold and black armor, strong jaw and chin, single subject, cosmic space background, cinematic, photorealistic, 8k",
        "negative": "human skin, multiple people, two people, ugly, blurry, cartoon, duplicate, extra person",
        "denoise": 0.75, "cnet_strength": 0.85,
    },
    "predator": {
        "positive": "full body portrait of the Predator alien warrior, mandibles, dreadlocks, biomask, single subject, dense jungle background, thermal vision overlay, cinematic, detailed, 8k",
        "negative": "human face, multiple people, two people, ugly, blurry, cartoon, duplicate, extra person",
        "denoise": 0.74, "cnet_strength": 0.85,
    },
    "ghost":    {
        "positive": "full body portrait of a ghost, pale translucent spectral body, white ethereal aura, glowing edges, single subject, dark moody background, cinematic, 8k",
        "negative": "solid opaque body, multiple people, two people, ugly, blurry, cartoon, duplicate, extra person",
        "denoise": 0.72, "cnet_strength": 0.82,
    },
    "groot":    {
        "positive": "full body portrait of Groot from Guardians of the Galaxy, living tree humanoid, bark textured skin, wooden arms and body, green leaves growing from body, single subject, forest background, cinematic, photorealistic, 8k",
        "negative": "human skin, multiple people, two people, ugly, blurry, cartoon, duplicate, extra person",
        "denoise": 0.75, "cnet_strength": 0.85,
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
    max_dim = 768
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
    - If skeleton (MediaPipe pose) is available: use it — best pose fidelity
    - Otherwise: fall back to Canny edges masked to person silhouette
    """
    max_dim = 768
    h, w    = frame.shape[:2]
    scale   = max_dim / max(h, w)
    nh      = int((h * scale) // 64) * 64
    nw      = int((w * scale) // 64) * 64

    if skeleton is not None:
        # Use pose skeleton — white lines on black, perfectly captures body pose
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

def _build_workflow(char_key, image_name, canny_name):
    cfg = CHARACTER_PROMPTS.get(char_key, CHARACTER_PROMPTS["navi"])
    return {
        "1":  {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"sd_xl_turbo_1.0_fp16.safetensors"}},
        "2":  {"class_type":"ControlNetLoader","inputs":{"control_net_name":"control-lora-canny-rank128.safetensors"}},
        "3":  {"class_type":"LoadImage","inputs":{"image":image_name}},
        "4":  {"class_type":"LoadImage","inputs":{"image":canny_name}},
        "5":  {"class_type":"VAEEncode","inputs":{"pixels":["3",0],"vae":["1",2]}},
        "6":  {"class_type":"CLIPTextEncode","inputs":{"text":cfg["positive"],"clip":["1",1]}},
        "7":  {"class_type":"CLIPTextEncode","inputs":{"text":cfg["negative"],"clip":["1",1]}},
        "8":  {"class_type":"ControlNetApply","inputs":{"conditioning":["6",0],"control_net":["2",0],"image":["4",0],"strength":cfg["cnet_strength"]}},
        "9":  {"class_type":"KSampler","inputs":{"model":["1",0],"positive":["8",0],"negative":["7",0],"latent_image":["5",0],"seed":int(time.time())%2**32,"steps":6,"cfg":1.0,"sampler_name":"euler_ancestral","scheduler":"karras","denoise":cfg["denoise"]}},
        "10": {"class_type":"VAEDecode","inputs":{"samples":["9",0],"vae":["1",2]}},
        "11": {"class_type":"SaveImage","inputs":{"images":["10",0],"filename_prefix":f"kiosk_{char_key}"}},
    }

def _blur_background(frame, selection_mask=None):
    """
    Heavily blur the background, keep person sharp.
    Prevents background edges from confusing ControlNet into generating extra figures.
    """
    if selection_mask is None or np.all(selection_mask >= 0.99):
        # No specific selection — use rembg-style simple background blur
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


def generate_character(frame, char_key, selection_mask=None, timeout=180):
    if not is_comfy_running():
        return None, "ComfyUI not running"
    try:
        # Blur background so model focuses on the person
        prepped      = _blur_background(frame, selection_mask)
        img_name     = _upload_frame(prepped)
        control_name = _make_control_image(prepped, selection_mask)
        workflow     = _build_workflow(char_key, img_name, control_name)
        client_id  = uuid.uuid4().hex
        result     = _api("prompt", {"prompt": workflow, "client_id": client_id}, "POST")
        prompt_id  = result["prompt_id"]

        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.5)
            try:
                history = _api(f"history/{prompt_id}")
                if prompt_id in history:
                    for _, node_out in history[prompt_id]["outputs"].items():
                        if "images" in node_out:
                            info    = node_out["images"][0]
                            img_url = (f"{COMFY_URL}/view?"
                                      f"filename={urllib.parse.quote(info['filename'])}"
                                      f"&subfolder={urllib.parse.quote(info.get('subfolder',''))}"
                                      f"&type={info.get('type','output')}")
                            with urllib.request.urlopen(img_url, timeout=10) as r:
                                img_bytes = r.read()
                            arr = np.frombuffer(img_bytes, dtype=np.uint8)
                            return cv2.imdecode(arr, cv2.IMREAD_COLOR), None
            except Exception:
                pass
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
        print(f"[{'OK' if self.available else 'WARN'}] ComfyUI "
              f"{'connected' if self.available else 'not running — start ~/start_comfyui.sh'}")

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
