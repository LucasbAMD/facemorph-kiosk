"""
face_processor.py — AMD Adapt Kiosk
Full character transformation pipeline:
  1. rembg segments person from background (threaded, ~16fps)
  2. Apply full-body color shift to match character (skin/clothes → character color)
  3. Face swap character's face onto user's face (InsightFace inswapper)
  4. Replace background with character's world
  5. Rim glow around person silhouette

Result: User looks like they ARE the character — right color, right face, right world.
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("[WARN] rembg not installed — run: venv\\Scripts\\python.exe -m pip install rembg")

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
        for p in ["DmlExecutionProvider","ROCMExecutionProvider",
                  "CUDAExecutionProvider","CPUExecutionProvider"]:
            if p in ort.get_available_providers():
                print(f"[OK] ONNX: {p}")
                return [p, "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]

ONNX_PROVIDERS = _detect_providers()

# ══════════════════════════════════════════════════════════════════════════════
# Character definitions
# Each character has:
#   hue        - target hue (HSV 0-179) for body color shift
#   sat_boost  - how much to boost saturation
#   val_scale  - value multiplier (brightness)
#   glow       - BGR rim glow color
#   ref        - reference photo filename (place in faces/fantasy/)
#   bg_fn      - which background generator to use
# ══════════════════════════════════════════════════════════════════════════════
CHAR_CONFIG = {
    "navi": {
        "label":"Na'vi", "subtitle":"Avatar · Pandora", "emoji":"🔵", "color":"#00d4ff",
        "hue":103, "sat_boost":120, "val_scale":0.82,
        "glow":(255,210,40), "bg":"pandora",
        "ref":"Avatar_Navi.jpg",
    },
    "hulk": {
        "label":"Hulk", "subtitle":"Gamma · Avengers", "emoji":"💚", "color":"#22dd44",
        "hue":62, "sat_boost":140, "val_scale":0.92,
        "glow":(30,230,50), "bg":"gamma",
        "ref":"Hulk.jpg",
    },
    "thanos": {
        "label":"Thanos", "subtitle":"Infinity · Mad Titan", "emoji":"💜", "color":"#9b30ff",
        "hue":148, "sat_boost":100, "val_scale":0.68,
        "glow":(200,50,255), "bg":"space",
        "ref":"Thanos.jpg",
    },
    "predator": {
        "label":"Predator", "subtitle":"Thermal · Cloaked", "emoji":"👁️", "color":"#ff6600",
        "hue":None, "sat_boost":0, "val_scale":1.0,   # thermal — special handling
        "glow":(0,80,255), "bg":"thermal",
        "ref":"Predator.jpg",
    },
    "ghost": {
        "label":"Ghost", "subtitle":"Spectral · Ethereal", "emoji":"👻", "color":"#aaddff",
        "hue":105, "sat_boost":-80, "val_scale":1.25,  # negative = desaturate
        "glow":(255,245,220), "bg":"space_blue",
        "ref":"Ghost.jpg",
    },
    "groot": {
        "label":"Groot", "subtitle":"Guardians · Flora", "emoji":"🌿", "color":"#8B5E3C",
        "hue":18, "sat_boost":60, "val_scale":0.75,
        "glow":(40,180,40), "bg":"forest",
        "ref":"Groot.jpg",
    },
}
CHARACTER_ORDER = ["navi","hulk","thanos","predator","ghost","groot"]

# ══════════════════════════════════════════════════════════════════════════════
# Background generators  (cached after first call)
# ══════════════════════════════════════════════════════════════════════════════
def _pandora_bg(h, w):
    bg = np.full((h,w,3), (12,18,8), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(400):
        x,y = int(rng.integers(0,w)), int(rng.integers(0,h))
        r   = int(rng.integers(2,9))
        hv  = int(rng.integers(85,115))
        bv  = int(rng.integers(120,255))
        col = cv2.cvtColor(np.array([[[hv,200,bv]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0].tolist()
        cv2.circle(bg,(x,y),r,col,-1)
        cv2.circle(bg,(x,y),r*4,[c//5 for c in col],-1)
    # light shafts
    shaft = np.zeros_like(bg)
    for x in range(0,w,w//7):
        cv2.line(shaft,(x,0),(x+rng.integers(-40,40),h),(15,35,10),20)
    bg = cv2.add(cv2.GaussianBlur(bg,(31,31),0), cv2.GaussianBlur(shaft,(21,21),0))
    # blue mist at base
    mist = np.zeros_like(bg); mist[int(h*0.7):,:] = (30,15,5)
    return cv2.add(bg, mist)

def _gamma_bg(h, w):
    """Gamma-ray lab / destroyed city — dark with green energy."""
    bg = np.zeros((h,w,3),dtype=np.uint8)
    bg[:,:] = (5,12,5)
    # Rubble silhouettes
    rng = np.random.default_rng(11)
    for _ in range(15):
        x1 = int(rng.integers(0,w)); y1 = h
        x2 = x1+int(rng.integers(30,120))
        y2 = h-int(rng.integers(40,200))
        pts = np.array([[x1,y1],[x2,y2],[x2+rng.integers(-20,20),h]],np.int32)
        cv2.fillPoly(bg,[pts],(8,18,8))
    # Green energy streaks
    for _ in range(30):
        x1,y1 = int(rng.integers(0,w)), int(rng.integers(0,h))
        x2,y2 = x1+rng.integers(-60,60), y1+rng.integers(-60,60)
        cv2.line(bg,(x1,y1),(x2,y2),(0,int(rng.integers(40,120)),0),1)
    return cv2.GaussianBlur(bg,(15,15),0)

def _space_bg(h, w, tint=(60,15,60)):
    bg = np.zeros((h,w,3),dtype=np.uint8)
    bg[:] = [t//5 for t in tint]
    rng = np.random.default_rng(7)
    for _ in range(600):
        x,y = int(rng.integers(0,w)), int(rng.integers(0,h))
        b   = int(rng.integers(80,255))
        s   = 1 if rng.random() > 0.97 else 0
        cv2.circle(bg,(x,y),s,(b,b,b),-1) if s else None
        bg[y,x] = (b,b,b)
    # nebula blob
    nb = np.zeros_like(bg)
    cv2.ellipse(nb,(w//3,h//3),(w//4,h//5),30,0,360,[t//2 for t in tint],-1)
    bg = cv2.add(bg, cv2.GaussianBlur(nb,(101,101),0))
    return bg

def _thermal_bg(h, w):
    """Near-black thermal background — makes the person pop."""
    bg = np.zeros((h,w,3),dtype=np.uint8)
    bg[:] = (0,0,15)
    return bg

def _forest_bg(h, w):
    bg = np.zeros((h,w,3),dtype=np.uint8)
    bg[:int(h*0.45),:] = (20,55,10)
    bg[int(h*0.45):,:] = (8,28,5)
    rng = np.random.default_rng(3)
    for _ in range(25):
        x  = int(rng.integers(0,w))
        ty = int(rng.uniform(h*0.15,h*0.7))
        r  = int(rng.integers(20,55))
        cv2.line(bg,(x,h),(x,ty),(20,50,15),max(3,r//4))
        for dr in range(3):
            cv2.circle(bg,(x+rng.integers(-r//2,r//2),ty-dr*r//3),
                       r-dr*10,(10+dr*5,80+dr*20,10+dr*5),-1)
    return cv2.GaussianBlur(bg,(31,31),0)

BG_FN = {
    "pandora":    _pandora_bg,
    "gamma":      _gamma_bg,
    "space":      lambda h,w: _space_bg(h,w,(60,15,60)),
    "space_blue": lambda h,w: _space_bg(h,w,(30,30,80)),
    "thermal":    _thermal_bg,
    "forest":     _forest_bg,
    "black":      lambda h,w: np.zeros((h,w,3),dtype=np.uint8),
}

# ══════════════════════════════════════════════════════════════════════════════
# Threaded background segmenter (rembg)
# ══════════════════════════════════════════════════════════════════════════════
class Segmenter:
    def __init__(self):
        self._mask = None; self._frame = None
        self._lock = threading.Lock(); self._running = True
        self.ready = False
        if REMBG_AVAILABLE:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            print("[..] Loading rembg (downloads ~170 MB on first run)...")
            sess = new_session("u2net_human_seg")
            self.ready = True
            print("[OK] rembg ready — person segmentation active")
            while self._running:
                with self._lock: frame = self._frame
                if frame is not None:
                    try:
                        m = remove(frame, session=sess, only_mask=True, post_process_mask=True)
                        m = cv2.GaussianBlur(m,(19,19),0)
                        with self._lock: self._mask = m
                    except Exception: pass
                time.sleep(0.06)
        except Exception as e:
            print(f"[ERROR] rembg: {e}")

    def update(self, frame):
        with self._lock: self._frame = frame

    def get(self, h, w):
        with self._lock: m = self._mask
        if m is None: return None
        if m.shape[:2] != (h,w): m = cv2.resize(m,(w,h))
        return m.astype(np.float32)/255.0

    def stop(self): self._running = False

# ══════════════════════════════════════════════════════════════════════════════
# FaceProcessor
# ══════════════════════════════════════════════════════════════════════════════
class FaceProcessor:
    def __init__(self, faces_dir="faces"):
        self.faces_dir = Path(faces_dir)
        self.mode      = "passthrough"
        self.character = "navi"
        self.face_key  = None
        self._lock     = threading.Lock()
        self._bg_cache = {}

        self.catalog   = {}
        self.char_refs = {}
        self._load_catalog()
        self._load_char_refs()

        self.face_app = None
        self.swapper  = None
        if INSIGHTFACE_AVAILABLE:
            self._init_insightface()

        self.seg = Segmenter()

        self.face_det = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _init_insightface(self):
        try:
            self.face_app = FaceAnalysis(name="buffalo_l", providers=ONNX_PROVIDERS)
            self.face_app.prepare(ctx_id=0, det_size=(640,640))
            mp = Path("models/inswapper_128.onnx")
            if mp.exists() and mp.stat().st_size > 400_000_000:
                self.swapper = insightface.model_zoo.get_model(str(mp), providers=ONNX_PROVIDERS)
                print("[OK] Face swap model ready")
            else:
                print("[WARN] inswapper_128.onnx missing/corrupted — run python download_models.py")
        except Exception as e:
            print(f"[ERROR] InsightFace: {e}")

    def _load_catalog(self):
        if not self.faces_dir.exists(): self.faces_dir.mkdir(parents=True); return
        ext = {".jpg",".jpeg",".png",".webp"}
        for cat in self.faces_dir.iterdir():
            if not cat.is_dir(): continue
            for p in cat.iterdir():
                if p.suffix.lower() not in ext: continue
                img = cv2.imread(str(p))
                if img is None: continue
                key = f"{cat.name}/{p.stem}"
                self.catalog[key] = {"label":p.stem.replace("_"," ").title(),
                                     "category":cat.name.title(),"img":img,"path":str(p)}
        print(f"[OK] {len(self.catalog)} face(s) loaded")

    def _load_char_refs(self):
        for key, cfg in CHAR_CONFIG.items():
            for cat in ["fantasy","celebrity","custom"]:
                p = self.faces_dir / cat / cfg["ref"]
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        self.char_refs[key] = img
                        print(f"[OK] Ref loaded: {key} ← {p.name}")
                        break

    def reload_catalog(self):
        with self._lock:
            self.catalog.clear(); self.char_refs.clear()
            self._load_catalog(); self._load_char_refs()

    def get_catalog(self):
        return [{"key":k,"label":v["label"],"category":v["category"]} for k,v in self.catalog.items()]

    def get_characters(self):
        out = []
        for k in CHARACTER_ORDER:
            c = CHAR_CONFIG[k]
            out.append({
                "key":k, "label":c["label"], "subtitle":c["subtitle"],
                "emoji":c["emoji"], "color":c["color"],
                "swap_ready": k in self.char_refs and self.swapper is not None,
                "ref_photo":  c["ref"],
            })
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode, face_key=None, character=None):
        with self._lock:
            self.mode = mode
            if face_key:  self.face_key  = face_key
            if character and character in CHAR_CONFIG:
                self.character = character

    def process_frame(self, frame):
        if frame is None: return frame
        self.seg.update(frame)
        with self._lock:
            mode = self.mode; char = self.character; fkey = self.face_key
        try:
            if   mode == "character":             return self._character(frame, char)
            elif mode == "minecraft":             return self._minecraft(frame)
            elif mode == "cyberpunk":             return self._cyberpunk(frame)
            elif mode == "faceswap" and fkey:     return self._swap_faces(frame, self.catalog.get(fkey,{}).get("img"))
            return frame
        except Exception as e:
            print(f"[ERR] process_frame: {e}"); import traceback; traceback.print_exc()
            return frame

    # ══════════════════════════════════════════════════════════════════════════
    # CHARACTER transform — the main pipeline
    # ══════════════════════════════════════════════════════════════════════════
    def _character(self, frame, char):
        cfg  = CHAR_CONFIG[char]
        h, w = frame.shape[:2]
        mask = self.seg.get(h, w)

        # ── Step 1: Body color transform ──────────────────────────────────────
        if char == "predator":
            # Thermal — invert + heatmap
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            colored = cv2.applyColorMap(cv2.bitwise_not(gray), cv2.COLORMAP_JET)
            body    = colored
        elif char == "ghost":
            # Pale, desaturated, slightly luminous
            body = self._hue_shift(frame, cfg["hue"], cfg["sat_boost"], cfg["val_scale"])
            body = cv2.addWeighted(body, 0.5, frame, 0.5, 0)
        else:
            body = self._hue_shift(frame, cfg["hue"], cfg["sat_boost"], cfg["val_scale"])

        # ── Step 2: Face swap (if model + ref photo available) ─────────────────
        result = body.copy()
        ref    = self.char_refs.get(char)
        if ref is not None and self.swapper is not None and self.face_app is not None:
            swapped = self._swap_faces(body, ref)
            if swapped is not None:
                result = swapped

        # ── Step 3: Composite person over character background ─────────────────
        if mask is not None:
            bg = self._bg(cfg["bg"], h, w)
            # Blend edges softly
            m3 = np.stack([mask]*3, axis=2)
            result = np.clip(result.astype(np.float32)*m3 +
                             bg.astype(np.float32)*(1.0-m3), 0,255).astype(np.uint8)
            # Rim glow
            result = self._rim(result, mask, cfg["glow"])
        else:
            # rembg still loading — show progress msg
            cv2.putText(result, "Loading segmentation...", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,200), 2)

        # ── Step 4: Show status if face swap not available ─────────────────────
        if ref is None or self.swapper is None:
            missing = []
            if ref is None:        missing.append(f"faces/fantasy/{cfg['ref']}")
            if self.swapper is None: missing.append("models/inswapper_128.onnx")
            msg = "For full effect: " + " + ".join(missing)
            overlay = result.copy()
            cv2.rectangle(overlay, (0,h-44),(w,h),(0,0,0),-1)
            result = cv2.addWeighted(overlay,0.6,result,0.4,0)
            cv2.putText(result, msg, (8,h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,230,180), 1, cv2.LINE_AA)

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # MINECRAFT
    # ══════════════════════════════════════════════════════════════════════════
    def _minecraft(self, frame):
        h, w  = frame.shape[:2]
        bs    = 22  # block size

        # Pixelate entire frame
        small = cv2.resize(frame, (w//bs, h//bs), interpolation=cv2.INTER_NEAREST)
        pix   = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)

        # Block grid
        grid = pix.copy()
        for x in range(0,w,bs): cv2.line(grid,(x,0),(x,h),(0,0,0),1)
        for y in range(0,h,bs): cv2.line(grid,(0,y),(w,y),(0,0,0),1)

        # Warp Steve texture onto detected faces
        result = grid.copy()
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = self.face_det.detectMultiScale(gray, 1.1, 5, minSize=(50,50))
        for (fx,fy,fw,fh) in faces:
            pad = int(fw*0.15)
            x1,y1 = max(0,fx-pad),    max(0,fy-pad)
            x2,y2 = min(w,fx+fw+pad), min(h,fy+fh+pad)
            sw,sh  = x2-x1, y2-y1
            # Build Steve face at exact block resolution
            bw,bh = max(1,sw//8), max(1,sh//8)
            steve = self._build_steve(bw*8, bh*8)
            result[y1:y2, x1:x2] = cv2.resize(steve,(sw,sh),interpolation=cv2.INTER_NEAREST)

        # Background swap
        mask = self.seg.get(h,w)
        if mask is not None:
            # Blocky mask edges — snap mask to block grid
            mk = (mask*255).astype(np.uint8)
            ms = cv2.resize(mk,(w//bs,h//bs),interpolation=cv2.INTER_NEAREST)
            mb = cv2.resize(ms,(w,h),interpolation=cv2.INTER_NEAREST).astype(np.float32)/255.0
            bg = self._bg("mc", h, w)
            m3 = np.stack([mb]*3, axis=2)
            result = np.clip(result.astype(np.float32)*m3 +
                             bg.astype(np.float32)*(1-m3), 0,255).astype(np.uint8)
        return result

    def _build_steve(self, tw, th):
        """Build Steve skin texture at given size (pixel art)."""
        # 8x8 pixel layout: H=hair, S=skin, E=eye, P=pupil, M=mouth, D=dark
        STEVE = [
            "HHHHHHHH",
            "HHHHHHHH",
            "SSESSSESS"[:8],
            "SSPSSPS"[:8]+"S",
            "SSSSSSSS",
            "SDDDDSS"[:8],
            "SDMMMS"[:8]+"SS"[:1],
            "SSSSSSSS",
        ]
        COLORS = {
            "H":(65,  90, 115),   # dark brown hair
            "S":(115,155,195),    # Steve skin (warm tan, BGR)
            "E":(230,230,230),    # eye white
            "P":( 40, 40,180),    # blue iris
            "M":( 30, 40, 80),    # mouth/beard dark
            "D":( 50, 55, 80),    # darker skin shadow
        }
        tex = np.zeros((th, tw, 3), dtype=np.uint8)
        ch  = th // 8
        cw  = tw // 8
        for ri, row in enumerate(STEVE):
            for ci, ch_char in enumerate(row[:8]):
                col = COLORS.get(ch_char, COLORS["S"])
                y1,y2 = ri*ch, (ri+1)*ch
                x1,x2 = ci*cw, (ci+1)*cw
                tex[y1:y2, x1:x2] = col
        return tex

    # ══════════════════════════════════════════════════════════════════════════
    # CYBERPUNK
    # ══════════════════════════════════════════════════════════════════════════
    def _cyberpunk(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        neon  = np.zeros_like(frame)
        neon[:,:,0] = edges           # blue channel
        neon[:,:,2] = edges           # red channel → magenta
        dark  = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)
        out   = cv2.add(dark, neon)
        # Scanlines
        for y in range(0,frame.shape[0],4):
            out[y,:] = (out[y,:].astype(np.int32)*0.7).astype(np.uint8)
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # FACE SWAP (InsightFace)
    # ══════════════════════════════════════════════════════════════════════════
    def _swap_faces(self, frame, ref_img):
        if ref_img is None or not self.face_app or not self.swapper:
            return frame
        try:
            ref_faces  = self.face_app.get(ref_img)
            live_faces = self.face_app.get(frame)
            if not ref_faces or not live_faces:
                return frame
            result = frame.copy()
            for lf in live_faces:
                result = self.swapper.get(result, lf, ref_faces[0], paste_back=True)
            return result
        except Exception as e:
            print(f"[WARN] swap: {e}")
            return frame

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _hue_shift(self, img, hue, sat_boost, val_scale):
        """Shift ENTIRE IMAGE to target hue + boost saturation + scale value."""
        if hue is None: return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        # Force hue to character hue everywhere
        hsv[:,:,0] = hue
        # Boost or reduce saturation
        if sat_boost > 0:
            # Pull saturation UP — flat areas become vivid
            hsv[:,:,1] = np.clip(
                hsv[:,:,1].astype(np.float32) + (255-hsv[:,:,1])*0.7 + sat_boost,
                0, 255)
        else:
            # Pull saturation DOWN (ghost)
            hsv[:,:,1] = np.clip(hsv[:,:,1]+sat_boost, 0, 255)
        # Scale value
        hsv[:,:,2] = np.clip(hsv[:,:,2].astype(np.float32)*val_scale, 10, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _rim(self, frame, mask_f, color_bgr, width=3):
        m8  = (mask_f*255).astype(np.uint8)
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        dil = cv2.dilate(m8, k, iterations=width)
        edge= cv2.subtract(dil, m8)
        ef  = cv2.GaussianBlur(edge,(13,13),0).astype(np.float32)/255.0
        rim = np.zeros_like(frame, dtype=np.float32)
        for c,v in enumerate(color_bgr): rim[:,:,c] = ef*v
        return np.clip(frame.astype(np.float32)+rim, 0,255).astype(np.uint8)

    def _bg(self, name, h, w):
        key = f"{name}_{h}_{w}"
        if key not in self._bg_cache:
            if name == "mc":
                self._bg_cache[key] = self._mc_bg(h,w)
            else:
                fn = BG_FN.get(name, BG_FN["black"])
                self._bg_cache[key] = fn(h,w)
        return self._bg_cache[key]

    def _mc_bg(self, h, w):
        bg = np.zeros((h,w,3),dtype=np.uint8)
        sh = int(h*0.55)
        # Sky
        for y in range(sh):
            t = y/sh
            bg[y,:] = (int(200+t*35), int(210+t*25), int(240+t*15))
        # Clouds (pixelated)
        for cx in [w//5,w//2,4*w//5]:
            for dx,dy in [(-40,0),(0,-15),(40,0),(0,0),(-20,-12),(20,-12)]:
                cv2.rectangle(bg,(cx+dx-25,sh//3+dy),(cx+dx+25,sh//3+dy+20),(240,240,240),-1)
        # Grass
        bg[sh:sh+int(h*0.06),:] = (34,130,34)
        # Dirt
        bg[sh+int(h*0.06):,:] = (67,96,134)
        # Pixelate
        bs=20
        s=cv2.resize(bg,(w//bs,h//bs),interpolation=cv2.INTER_NEAREST)
        bg=cv2.resize(s,(w,h),interpolation=cv2.INTER_NEAREST)
        # Grid
        for x in range(0,w,bs): cv2.line(bg,(x,0),(x,h),(0,0,0),1)
        for y in range(0,h,bs): cv2.line(bg,(0,y),(w,y),(0,0,0),1)
        return bg
