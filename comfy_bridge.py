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

CHARACTER_PROMPTS = {
    "navi":     {"positive":"Na'vi Avatar alien, bioluminescent blue skin, glowing cyan face markings, large amber eyes, Pandora jungle, cinematic, photorealistic, 8k","negative":"human skin, ugly, blurry, cartoon","denoise":0.75,"cnet_strength":0.80},
    "hulk":     {"positive":"Incredible Hulk Marvel, massive green muscles, green skin, torn purple pants, angry, cinematic, photorealistic, 8k","negative":"normal skin, cartoon, ugly, blurry","denoise":0.78,"cnet_strength":0.75},
    "thanos":   {"positive":"Thanos Marvel villain, purple wrinkled skin, gold armor, strong jaw, cosmic background, cinematic, photorealistic, 8k","negative":"human skin, ugly, blurry, cartoon","denoise":0.76,"cnet_strength":0.75},
    "predator": {"positive":"Predator alien warrior, mandibles, dreadlocks, thermal imaging view, jungle, cinematic, detailed, 8k","negative":"human, ugly, blurry, cartoon","denoise":0.72,"cnet_strength":0.70},
    "ghost":    {"positive":"Ghost Marvel character, pale translucent body, white spectral aura, ethereal glow, dark background, cinematic, 8k","negative":"solid body, ugly, blurry, cartoon","denoise":0.70,"cnet_strength":0.70},
    "groot":    {"positive":"Groot Guardians of the Galaxy, living tree humanoid, bark skin, wooden body, green leaves, forest, cinematic, photorealistic, 8k","negative":"human skin, ugly, blurry, cartoon","denoise":0.76,"cnet_strength":0.75},
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

def _make_canny(frame):
    """Compute Canny edges locally and upload — no custom ComfyUI node needed."""
    max_dim = 768
    h, w    = frame.shape[:2]
    scale   = max_dim / max(h, w)
    nh      = int((h * scale) // 64) * 64
    nw      = int((w * scale) // 64) * 64
    resized = cv2.resize(frame, (nw, nh))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(gray, 100, 200)
    edges3  = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return _upload_frame(edges3, "kiosk_canny.png")

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
        "9":  {"class_type":"KSampler","inputs":{"model":["1",0],"positive":["8",0],"negative":["7",0],"latent_image":["5",0],"seed":int(time.time())%2**32,"steps":4,"cfg":1.0,"sampler_name":"euler_ancestral","scheduler":"karras","denoise":cfg["denoise"]}},
        "10": {"class_type":"VAEDecode","inputs":{"samples":["9",0],"vae":["1",2]}},
        "11": {"class_type":"SaveImage","inputs":{"images":["10",0],"filename_prefix":f"kiosk_{char_key}"}},
    }

def generate_character(frame, char_key, selection_mask=None, timeout=180):
    if not is_comfy_running():
        return None, "ComfyUI not running"
    try:
        # Apply selection mask so only chosen people are transformed
        masked_frame = _apply_selection_mask(frame, selection_mask)

        img_name   = _upload_frame(masked_frame)
        canny_name = _make_canny(masked_frame)
        workflow   = _build_workflow(char_key, img_name, canny_name)
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
