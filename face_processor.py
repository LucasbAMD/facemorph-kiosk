"""
face_processor.py — AMD Adapt Kiosk
Legit character transforms:
  - Minecraft: Steve skin texture warped onto face via facial landmarks (no model needed)
  - All others: Real InsightFace face swap using character reference photos
  - rembg background removal + character backgrounds
  - Falls back gracefully if model/photos not available
"""

import cv2
import numpy as np
import threading
import time
import base64
from pathlib import Path

# ── rembg ─────────────────────────────────────────────────────────────────────
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# ── InsightFace ───────────────────────────────────────────────────────────────
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# ── ONNX providers ────────────────────────────────────────────────────────────
def _detect_providers():
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        for p in ["DmlExecutionProvider","ROCMExecutionProvider",
                  "MIGraphXExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]:
            if p in available:
                print(f"[OK] ONNX provider: {p}")
                return [p, "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]

ONNX_PROVIDERS = _detect_providers()

# ══════════════════════════════════════════════════════════════════════════════
# Minecraft Steve face texture (64x64 skin, face region 8x8 at offset 8,8)
# We encode a minimal Steve face as raw pixel data
# ══════════════════════════════════════════════════════════════════════════════

# Steve's face colors (BGR)
STEVE_SKIN   = (110, 150, 190)   # tan/brown skin
STEVE_HAIR   = ( 60,  80, 105)   # dark brown hair
STEVE_EYE_W  = (220, 220, 220)   # eye white
STEVE_PUPIL  = ( 30,  30, 150)   # blue iris/pupil
STEVE_MOUTH  = ( 40,  60, 100)   # mouth dark
STEVE_BEARD  = ( 60,  50,  40)   # beard dark

# 8x8 Steve face grid (each cell = 1 "block pixel")
# S=skin, H=hair, W=eye-white, P=pupil, M=mouth, B=beard/shadow, X=dark
STEVE_FACE_8x8 = [
    "HHHHHHHH",  # row 0 - hair
    "HHHHHHHH",  # row 1 - hair
    "SSWWSWWS",  # row 2 - eyes row (W=white around eye)
    "SSPPSPPSS"[:8],  # row 3 - pupils (truncated)
    "SSSSSSSS",  # row 4 - nose area
    "SBBBBSSS"[:8],  # row 5 - mouth top
    "SBMMMBSS"[:8],  # row 6 - mouth
    "SSSSSSSS",  # row 7 - chin
]

def build_steve_texture(size=256):
    """Build a clean Steve face texture at given size."""
    cell = size // 8
    tex = np.zeros((size, size, 3), dtype=np.uint8)

    color_map = {
        'H': STEVE_HAIR,
        'S': STEVE_SKIN,
        'W': STEVE_EYE_W,
        'P': STEVE_PUPIL,
        'M': STEVE_MOUTH,
        'B': STEVE_BEARD,
        'X': (20, 20, 20),
    }

    for row_i, row_str in enumerate(STEVE_FACE_8x8):
        for col_i, ch in enumerate(row_str[:8]):
            color = color_map.get(ch, STEVE_SKIN)
            y1, y2 = row_i * cell, (row_i + 1) * cell
            x1, x2 = col_i * cell, (col_i + 1) * cell
            tex[y1:y2, x1:x2] = color

    # Hard pixelated look — no antialiasing
    return tex

STEVE_TEXTURE = build_steve_texture(512)

# ══════════════════════════════════════════════════════════════════════════════
# Character definitions — each has a default reference photo filename to look for
# ══════════════════════════════════════════════════════════════════════════════
CHARACTERS = {
    "navi":     {"label":"Na'vi",   "subtitle":"Avatar · Pandora",     "emoji":"🔵","color":"#00d4ff","ref":"Avatar_Navi.jpg"},
    "hulk":     {"label":"Hulk",    "subtitle":"Gamma · Avengers",     "emoji":"💚","color":"#22dd44","ref":"Hulk.jpg"},
    "thanos":   {"label":"Thanos",  "subtitle":"Infinity · Mad Titan", "emoji":"💜","color":"#9b30ff","ref":"Thanos.jpg"},
    "predator": {"label":"Predator","subtitle":"Thermal · Cloaked",    "emoji":"👁️","color":"#ff6600","ref":"Predator.jpg"},
    "ghost":    {"label":"Ghost",   "subtitle":"Spectral · Ethereal",  "emoji":"👻","color":"#aaddff","ref":"Ghost.jpg"},
    "groot":    {"label":"Groot",   "subtitle":"Guardians · Flora",    "emoji":"🌿","color":"#8B5E3C","ref":"Groot.jpg"},
}
CHARACTER_ORDER = ["navi","hulk","thanos","predator","ghost","groot"]
MINECRAFT_KEY   = "minecraft"

# ══════════════════════════════════════════════════════════════════════════════
# Background generators
# ══════════════════════════════════════════════════════════════════════════════
def make_pandora_bg(h, w):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = (15, 20, 10)
    rng = np.random.default_rng(42)
    for _ in range(300):
        x = int(rng.integers(0, w));  y = int(rng.integers(0, h))
        r = int(rng.integers(2, 8));  brightness = int(rng.integers(120, 255))
        hue = int(rng.integers(85, 115))
        col = cv2.cvtColor(np.array([[[hue, 200, brightness]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0].tolist()
        cv2.circle(bg, (x,y), r, col, -1)
        cv2.circle(bg, (x,y), r*3, [c//4 for c in col], -1)
    for x in range(0, w, w//6):
        shaft = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.line(shaft, (x+rng.integers(-30,30), 0), (x, h), (20,40,15), 18)
        bg = cv2.add(bg, shaft)
    return cv2.GaussianBlur(bg, (25, 25), 0)

def make_minecraft_bg(h, w):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    sky_h = int(h * 0.55)
    for y in range(sky_h):
        t = y / sky_h
        bg[y, :] = (int(135+t*40), int(180+t*20), int(235+t*10))
    bg[sky_h:sky_h+int(h*0.07), :] = (34, 139, 34)
    bg[sky_h+int(h*0.07):, :] = (67, 96, 134)
    bs = 20
    small = cv2.resize(bg, (w//bs, h//bs), interpolation=cv2.INTER_LINEAR)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    for cx in [w//5, w//2, 4*w//5]:
        cv2.ellipse(bg,(cx,int(sky_h*0.3)),(60,25),0,0,360,(240,240,240),-1)
        cv2.ellipse(bg,(cx-30,int(sky_h*0.33)),(40,18),0,0,360,(240,240,240),-1)
        cv2.ellipse(bg,(cx+30,int(sky_h*0.33)),(40,18),0,0,360,(240,240,240),-1)
    return bg

def make_space_bg(h, w, tint=(80,20,80)):
    bg = np.zeros((h,w,3), dtype=np.uint8)
    bg[:] = [t//6 for t in tint]
    rng = np.random.default_rng(7)
    for _ in range(400):
        x,y = int(rng.integers(0,w)), int(rng.integers(0,h))
        b = int(rng.integers(80,255))
        cv2.circle(bg,(x,y),1,(b,b,b),-1)
    return bg

def make_forest_bg(h, w):
    bg = np.zeros((h,w,3), dtype=np.uint8)
    bg[:int(h*0.4),:] = (30,80,20)
    bg[int(h*0.4):,:] = (10,40,10)
    rng = np.random.default_rng(3)
    for _ in range(20):
        x = int(rng.integers(0,w)); h2 = int(rng.uniform(h*0.3,h*0.9))
        r = int(rng.integers(12,35))
        cv2.line(bg,(x,h),(x,h2),(30,60,80),r//3)
        cv2.circle(bg,(x,h2),r,(20,100,30),-1)
    return cv2.GaussianBlur(bg,(31,31),0)

def make_thermal_bg(h, w):
    bg = np.zeros((h,w,3), dtype=np.uint8)
    bg[:] = (0,0,20)
    return bg

# ══════════════════════════════════════════════════════════════════════════════
# Background segmenter (rembg threaded)
# ══════════════════════════════════════════════════════════════════════════════
class BackgroundSegmenter:
    def __init__(self):
        self._mask = None; self._frame = None
        self._lock = threading.Lock(); self._running = True
        self.ready = False; self.session = None
        if REMBG_AVAILABLE:
            threading.Thread(target=self._init_and_run, daemon=True).start()

    def _init_and_run(self):
        try:
            print("[..] Loading rembg (first run downloads ~170 MB)...")
            self.session = new_session("u2net_human_seg")
            self.ready = True
            print("[OK] rembg ready — clean segmentation active")
            self._loop()
        except Exception as e:
            print(f"[ERROR] rembg init: {e}")

    def _loop(self):
        while self._running:
            with self._lock:
                frame = self._frame
            if frame is not None and self.session:
                try:
                    mask = remove(frame, session=self.session,
                                  only_mask=True, post_process_mask=True)
                    mask = cv2.GaussianBlur(mask, (21,21), 0)
                    with self._lock:
                        self._mask = mask
                except Exception:
                    pass
            time.sleep(0.07)

    def update(self, frame):
        with self._lock:
            self._frame = frame

    def get_mask(self, shape):
        with self._lock:
            mask = self._mask
        if mask is None:
            return None
        h, w = shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        return mask.astype(np.float32) / 255.0

    def stop(self):
        self._running = False

# ══════════════════════════════════════════════════════════════════════════════
# Main processor
# ══════════════════════════════════════════════════════════════════════════════
class FaceProcessor:
    def __init__(self, faces_dir="faces"):
        self.faces_dir = Path(faces_dir)
        self.current_mode      = "passthrough"
        self.current_character = "hulk"
        self.current_face_key  = None
        self._lock = threading.Lock()

        self.face_catalog  = {}
        self.char_refs     = {}   # char_key → face img array (loaded lazily)
        self._bg_cache     = {}

        self._load_face_catalog()
        self._load_char_refs()

        self.face_app = None
        self.swapper  = None
        if INSIGHTFACE_AVAILABLE:
            self._init_insightface()

        self.seg = BackgroundSegmenter()

        # OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_insightface(self):
        try:
            self.face_app = FaceAnalysis(name="buffalo_l", providers=ONNX_PROVIDERS)
            self.face_app.prepare(ctx_id=0, det_size=(640,640))
            mp_ = Path("models/inswapper_128.onnx")
            if mp_.exists():
                self.swapper = insightface.model_zoo.get_model(str(mp_), providers=ONNX_PROVIDERS)
                print("[OK] InsightFace swapper ready")
            else:
                print("[WARN] models/inswapper_128.onnx not found — face swap unavailable")
        except Exception as e:
            print(f"[ERROR] InsightFace: {e}")

    def _load_face_catalog(self):
        supported = {".jpg",".jpeg",".png",".webp"}
        if not self.faces_dir.exists():
            self.faces_dir.mkdir(parents=True); return
        for cat_dir in self.faces_dir.iterdir():
            if not cat_dir.is_dir(): continue
            for p in cat_dir.iterdir():
                if p.suffix.lower() not in supported: continue
                img = cv2.imread(str(p))
                if img is None: continue
                key = f"{cat_dir.name}/{p.stem}"
                self.face_catalog[key] = {
                    "label": p.stem.replace("_"," ").title(),
                    "category": cat_dir.name.title(),
                    "img": img, "path": str(p),
                }
        print(f"[OK] Loaded {len(self.face_catalog)} face(s)")

    def _load_char_refs(self):
        """Load character reference photos from faces/fantasy/ by filename."""
        for char_key, cfg in CHARACTERS.items():
            ref_name = cfg["ref"]
            # Check fantasy/ and celebrity/ folders
            for cat in ["fantasy","celebrity","custom"]:
                p = self.faces_dir / cat / ref_name
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        self.char_refs[char_key] = img
                        print(f"[OK] Character ref loaded: {char_key} → {p}")
                        break

    def reload_catalog(self):
        with self._lock:
            self.face_catalog.clear()
            self.char_refs.clear()
            self._load_face_catalog()
            self._load_char_refs()

    def get_catalog(self):
        return [{"key":k,"label":v["label"],"category":v["category"]}
                for k,v in self.face_catalog.items()]

    def get_characters(self):
        result = []
        for k in CHARACTER_ORDER:
            cfg = CHARACTERS[k]
            has_ref = k in self.char_refs
            swap_ready = has_ref and self.swapper is not None
            result.append({
                "key": k,
                "label": cfg["label"],
                "subtitle": cfg["subtitle"],
                "emoji": cfg["emoji"],
                "color": cfg["color"],
                "swap_ready": swap_ready,
                "ref_photo": cfg["ref"],
            })
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode, face_key=None, character=None):
        with self._lock:
            self.current_mode = mode
            if face_key:  self.current_face_key  = face_key
            if character and (character in CHARACTERS or character == MINECRAFT_KEY):
                self.current_character = character

    def process_frame(self, frame):
        if frame is None: return frame
        self.seg.update(frame)
        with self._lock:
            mode     = self.current_mode
            face_key = self.current_face_key
            char_key = self.current_character
        try:
            if   mode == "character":  return self._apply_character(frame, char_key)
            elif mode == "minecraft":  return self._apply_minecraft(frame)
            elif mode == "cyberpunk":  return self._apply_cyberpunk(frame)
            elif mode == "faceswap" and face_key:
                return self._do_face_swap(frame, self.face_catalog.get(face_key,{}).get("img"))
            return frame
        except Exception as e:
            print(f"[ERROR] process_frame: {e}")
            return frame

    # ── Character dispatch ────────────────────────────────────────────────────

    def _apply_character(self, frame, char_key):
        """
        If we have a reference photo + swap model: do a real face swap.
        Otherwise: fall back to color transform (still decent).
        """
        ref_img = self.char_refs.get(char_key)

        if ref_img is not None and self.swapper is not None and self.face_app is not None:
            # Real face swap — user literally gets the character's face
            result = self._do_face_swap(frame, ref_img)
        else:
            # Fallback: color transform (shows what's needed)
            result = self._color_fallback(frame, char_key)
            if ref_img is None and self.swapper is None:
                self._overlay_missing_msg(result,
                    f"Add faces/fantasy/{CHARACTERS[char_key]['ref']} + inswapper_128.onnx")
            elif ref_img is None:
                self._overlay_missing_msg(result,
                    f"Add faces/fantasy/{CHARACTERS[char_key]['ref']}")
            elif self.swapper is None:
                self._overlay_missing_msg(result,
                    "Download models/inswapper_128.onnx to enable")

        # Always add the character background behind the person
        mask = self.seg.get_mask(frame.shape)
        if mask is not None:
            bg = self._get_bg(char_key, frame.shape[0], frame.shape[1])
            result = self._composite(result, frame, mask, bg)
            result = self._rim_glow(result, mask, self._char_glow_color(char_key))

        return result

    def _color_fallback(self, frame, char_key):
        """Dramatic color transform when no face swap available."""
        params = {
            "navi":     (103, 160, 0.85),
            "hulk":     (65,  180, 0.90),
            "thanos":   (145, 130, 0.72),
            "predator": None,
            "ghost":    (105,   0, 1.10),
            "groot":    (18,  100, 0.78),
        }
        if char_key == "predator":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal = cv2.applyColorMap(cv2.bitwise_not(gray), cv2.COLORMAP_JET)
            return thermal
        p = params.get(char_key, (103, 160, 0.85))
        return self._shift_hue(frame, p[0], p[1], p[2])

    def _char_glow_color(self, char_key):
        colors = {
            "navi": (255,200,30), "hulk": (30,220,50),
            "thanos": (200,50,255), "predator": (0,80,255),
            "ghost": (255,245,220), "groot": (40,180,40),
        }
        return colors.get(char_key, (255,255,255))

    def _overlay_missing_msg(self, frame, msg):
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h-50), (frame.shape[1], h), (0,0,0), -1)
        cv2.putText(frame, msg, (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,180), 1, cv2.LINE_AA)

    # ── Real face swap ────────────────────────────────────────────────────────

    def _do_face_swap(self, frame, ref_img):
        """Swap all detected faces in frame with the reference face."""
        if ref_img is None or not self.face_app or not self.swapper:
            return frame
        try:
            ref_faces = self.face_app.get(ref_img)
            if not ref_faces:
                return frame
            live_faces = self.face_app.get(frame)
            if not live_faces:
                return frame
            result = frame.copy()
            for lf in live_faces:
                result = self.swapper.get(result, lf, ref_faces[0], paste_back=True)
            return result
        except Exception as e:
            print(f"[WARN] Face swap: {e}")
            return frame

    # ── Minecraft Steve ───────────────────────────────────────────────────────

    def _apply_minecraft(self, frame):
        """
        Minecraft transform:
        1. Detect face(s) with OpenCV
        2. Warp Steve texture onto each face region
        3. Pixelate entire frame with MC palette
        4. MC background behind person
        """
        h, w = frame.shape[:2]
        result = frame.copy()

        # Pixelate whole frame first (large blocks)
        bs = 20
        small = cv2.resize(frame, (w//bs, h//bs), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        # MC palette quantisation (fast approximate)
        pixelated = self._mc_colorize(pixelated)
        # Grid lines
        for x in range(0, w, bs): cv2.line(pixelated,(x,0),(x,h),(0,0,0),1)
        for y in range(0, h, bs): cv2.line(pixelated,(0,y),(w,y),(0,0,0),1)
        result = pixelated

        # Warp Steve face texture over detected faces
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
        for (fx, fy, fw, fh) in faces:
            # Add some margin and go square
            pad = int(fw * 0.1)
            x1 = max(0, fx - pad);  y1 = max(0, fy - pad)
            x2 = min(w, fx+fw+pad); y2 = min(h, fy+fh+pad)
            face_w = x2 - x1;       face_h = y2 - y1
            # Resize Steve texture to face size, keep pixelated
            block = max(4, face_w // 8)
            steve_small = cv2.resize(STEVE_TEXTURE, (8,8), interpolation=cv2.INTER_NEAREST)
            steve_face  = cv2.resize(steve_small, (face_w, face_h),
                                     interpolation=cv2.INTER_NEAREST)
            result[y1:y2, x1:x2] = steve_face

        # Background swap
        mask = self.seg.get_mask(frame.shape)
        if mask is not None:
            # Blocky mask
            mk = (mask*255).astype(np.uint8)
            mk_small = cv2.resize(mk,(w//bs,h//bs),interpolation=cv2.INTER_NEAREST)
            mk_block  = cv2.resize(mk_small,(w,h),interpolation=cv2.INTER_NEAREST).astype(np.float32)/255.0
            bg = self._get_bg("minecraft", h, w)
            result = self._composite(result, frame, mk_block, bg)

        return result

    def _mc_colorize(self, img):
        """Fast MC palette — just desaturate slightly and boost contrast."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1].astype(np.int16) - 20, 0, 255).astype(np.uint8)
        hsv[:,:,2] = np.clip(((hsv[:,:,2].astype(np.float32)-128)*1.3+128), 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ── Cyberpunk ─────────────────────────────────────────────────────────────

    def _apply_cyberpunk(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        neon  = np.zeros_like(frame)
        neon[:,:,0] = edges; neon[:,:,2] = edges
        neon[:,:,1] = (edges*0.5).astype(np.uint8)
        dark   = cv2.convertScaleAbs(frame, alpha=0.35, beta=0)
        result = cv2.add(dark, neon)
        result[:,:,1] = np.clip(result[:,:,1].astype(np.int32)+20, 0, 255).astype(np.uint8)
        return result

    # ── Background / composite helpers ───────────────────────────────────────

    def _get_bg(self, char_key, h, w):
        key = f"{char_key}_{h}_{w}"
        if key not in self._bg_cache:
            if   char_key == "navi":               bg = make_pandora_bg(h, w)
            elif char_key == "minecraft":          bg = make_minecraft_bg(h, w)
            elif char_key in ("thanos","ghost"):   bg = make_space_bg(h, w)
            elif char_key == "groot":              bg = make_forest_bg(h, w)
            elif char_key == "predator":           bg = make_thermal_bg(h, w)
            else:                                  bg = np.zeros((h,w,3), dtype=np.uint8)
            self._bg_cache[key] = bg
        return self._bg_cache[key]

    def _composite(self, transformed, original, mask_f, bg=None):
        m = np.stack([mask_f]*3, axis=2)
        if bg is not None:
            return np.clip(transformed.astype(np.float32)*m +
                           bg.astype(np.float32)*(1-m), 0, 255).astype(np.uint8)
        return np.clip(transformed.astype(np.float32)*m +
                       original.astype(np.float32)*(1-m), 0, 255).astype(np.uint8)

    def _rim_glow(self, frame, mask_f, color_bgr, width=3):
        m8 = (mask_f*255).astype(np.uint8)
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        edge = cv2.subtract(cv2.dilate(m8,k,iterations=width), m8)
        edge_b = cv2.GaussianBlur(edge,(11,11),0).astype(np.float32)/255.0
        rim = np.zeros_like(frame, dtype=np.float32)
        for c, val in enumerate(color_bgr):
            rim[:,:,c] = edge_b * val
        return np.clip(frame.astype(np.float32)+rim, 0, 255).astype(np.uint8)

    def _shift_hue(self, frame, hue, sat_floor=120, val_scale=0.9):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:,:,0] = hue
        hsv[:,:,1] = np.clip(hsv[:,:,1]+(255-hsv[:,:,1])*0.6, sat_floor, 255)
        hsv[:,:,2] = np.clip((hsv[:,:,2]*val_scale), 20, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
