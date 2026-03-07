"""
face_processor.py — FaceSwap Kiosk
Modes:
  passthrough  — raw webcam
  character    — 6 full-body preset characters (segmentation + pose + style)
  faceswap     — InsightFace celebrity swap
  minecraft    — pixelated blocks
  cyberpunk    — neon edge glow

Character presets (full body, works at 10+ feet):
  navi      — Avatar Na'vi: blue skin, bioluminescent markings
  hulk      — Hulk: gamma-green rage, muscle glow, vein lines
  thanos    — Thanos: deep purple, Infinity Stone glow dots
  predator  — Predator: thermal/infrared cloaking inversion
  ghost     — Ghost: ethereal white-blue spectre, desaturated + aura
  groot     — Groot: earthy bark-brown, nature glow markings
"""

import cv2
import numpy as np
import threading
from pathlib import Path

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ── ONNX provider auto-detection (ROCm → CUDA → CPU) ─────────────────────────
def _detect_providers():
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        for p in ["DmlExecutionProvider",
                  "ROCMExecutionProvider", "MIGraphXExecutionProvider",
                  "CUDAExecutionProvider", "CPUExecutionProvider"]:
            if p in available:
                print(f"[OK] ONNX provider: {p}")
                return [p, "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]

ONNX_PROVIDERS = _detect_providers()

# ══════════════════════════════════════════════════════════════════════════════
# Character Definitions
# ══════════════════════════════════════════════════════════════════════════════
CHARACTERS = {
    "navi": {
        "label":    "Na'vi",
        "subtitle": "Avatar · Pandora",
        "emoji":    "🔵",
        "color":    "#00d4ff",
        "hue":      103,   # blue-teal
        "sat_boost": 90,
        "val_scale": 0.88,
        "glow_primary":   (255, 240,  80),   # cyan  (BGR)
        "glow_secondary": (180, 255, 255),
        "rim_color":      (220, 200,  10),   # cyan rim
    },
    "hulk": {
        "label":    "Hulk",
        "subtitle": "Gamma · Avengers",
        "emoji":    "💚",
        "color":    "#22dd44",
        "hue":      62,    # gamma green
        "sat_boost": 110,
        "val_scale": 0.92,
        "glow_primary":   ( 50, 255,  80),   # bright green
        "glow_secondary": (150, 255, 150),
        "rim_color":      ( 30, 200,  30),
    },
    "thanos": {
        "label":    "Thanos",
        "subtitle": "Infinity · Mad Titan",
        "emoji":    "💜",
        "color":    "#9b30ff",
        "hue":      145,   # deep purple (HSV 0-179)
        "sat_boost": 80,
        "val_scale": 0.75,
        "glow_primary":   (200,  60, 255),   # gold → orange (Infinity stones)
        "glow_secondary": (100,  80, 255),
        "rim_color":      (180,  40, 200),
    },
    "predator": {
        "label":    "Predator",
        "subtitle": "Thermal · Cloaked",
        "emoji":    "👁️",
        "color":    "#ff6600",
        "hue":      None,  # special: thermal inversion, no hue shift
        "sat_boost": 0,
        "val_scale": 1.0,
        "glow_primary":   (  0, 120, 255),   # orange-red
        "glow_secondary": (  0,  60, 180),
        "rim_color":      (  0,  80, 200),
    },
    "ghost": {
        "label":    "Ghost",
        "subtitle": "Spectral · Ethereal",
        "emoji":    "👻",
        "color":    "#aaddff",
        "hue":      105,   # pale blue-white
        "sat_boost": -40,  # desaturate
        "val_scale": 1.15, # brighten
        "glow_primary":   (255, 240, 200),   # white-blue
        "glow_secondary": (255, 255, 255),
        "rim_color":      (255, 220, 180),
    },
    "groot": {
        "label":    "Groot",
        "subtitle": "Guardians · Flora",
        "emoji":    "🌿",
        "color":    "#8B5E3C",
        "hue":      18,    # warm bark brown
        "sat_boost": 50,
        "val_scale": 0.80,
        "glow_primary":   ( 60, 200,  80),   # forest green
        "glow_secondary": (100, 180, 120),
        "rim_color":      ( 40, 160,  40),
    },
}

# Ordered list for UI
CHARACTER_ORDER = ["navi", "hulk", "thanos", "predator", "ghost", "groot"]


class FaceProcessor:
    def __init__(self, faces_dir: str = "faces"):
        self.faces_dir = Path(faces_dir)
        self.current_mode      = "passthrough"
        self.current_character = "navi"
        self.current_face_key  = None
        self._lock = threading.Lock()

        self.face_catalog: dict = {}
        self._load_face_catalog()

        self.face_app = None
        self.swapper  = None
        if INSIGHTFACE_AVAILABLE:
            self._init_insightface()

        self.segmenter = None
        self.pose      = None
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_insightface(self):
        try:
            self.face_app = FaceAnalysis(name="buffalo_l", providers=ONNX_PROVIDERS)
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            mp_ = Path("models/inswapper_128.onnx")
            if mp_.exists():
                self.swapper = insightface.model_zoo.get_model(str(mp_), providers=ONNX_PROVIDERS)
                print("[OK] InsightFace swapper loaded")
            else:
                print("[WARN] models/inswapper_128.onnx not found")
        except Exception as e:
            print(f"[ERROR] InsightFace: {e}")

    def _init_mediapipe(self):
        try:
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            print("[OK] Body segmentation loaded")
        except Exception as e:
            print(f"[ERROR] Segmenter: {e}")
        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                smooth_landmarks=True, enable_segmentation=False,
                min_detection_confidence=0.4, min_tracking_confidence=0.4,
            )
            print("[OK] Pose loaded")
        except Exception as e:
            print(f"[ERROR] Pose: {e}")
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=4, refine_landmarks=True,
                min_detection_confidence=0.4, min_tracking_confidence=0.4,
            )
            print("[OK] Face mesh loaded")
        except Exception as e:
            print(f"[ERROR] FaceMesh: {e}")

    # ── Catalog ───────────────────────────────────────────────────────────────

    def _load_face_catalog(self):
        supported = {".jpg", ".jpeg", ".png", ".webp"}
        if not self.faces_dir.exists():
            self.faces_dir.mkdir(parents=True)
            return
        for cat_dir in self.faces_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            for p in cat_dir.iterdir():
                if p.suffix.lower() not in supported:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                key = f"{cat_dir.name}/{p.stem}"
                self.face_catalog[key] = {
                    "label":    p.stem.replace("_", " ").replace("-", " ").title(),
                    "category": cat_dir.name.title(),
                    "img":      img,
                    "path":     str(p),
                }
        print(f"[OK] Loaded {len(self.face_catalog)} face(s)")

    def reload_catalog(self):
        with self._lock:
            self.face_catalog.clear()
            self._load_face_catalog()

    def get_catalog(self):
        return [{"key": k, "label": v["label"], "category": v["category"]}
                for k, v in self.face_catalog.items()]

    def get_characters(self):
        return [
            {"key": k,
             "label":    CHARACTERS[k]["label"],
             "subtitle": CHARACTERS[k]["subtitle"],
             "emoji":    CHARACTERS[k]["emoji"],
             "color":    CHARACTERS[k]["color"]}
            for k in CHARACTER_ORDER
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: str, face_key: str = None, character: str = None):
        with self._lock:
            self.current_mode = mode
            if face_key:
                self.current_face_key = face_key
            if character and character in CHARACTERS:
                self.current_character = character

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame
        with self._lock:
            mode      = self.current_mode
            face_key  = self.current_face_key
            char_key  = self.current_character
        try:
            if   mode == "character":              return self._apply_character(frame, char_key)
            elif mode == "minecraft":              return self._apply_minecraft(frame)
            elif mode == "cyberpunk":              return self._apply_cyberpunk(frame)
            elif mode == "faceswap" and face_key:  return self._apply_faceswap(frame, face_key)
            else:                                  return frame
        except Exception as e:
            print(f"[ERROR] process_frame ({mode}): {e}")
            return frame

    # ══════════════════════════════════════════════════════════════════════════
    # Full-Body Character Transform
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_character(self, frame: np.ndarray, char_key: str) -> np.ndarray:
        cfg = CHARACTERS.get(char_key, CHARACTERS["navi"])
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── 1. Body segmentation mask ─────────────────────────────────────
        body_mask = self._get_body_mask(rgb, h, w)
        if body_mask is None:
            return self._color_shift_fallback(frame, cfg)

        # ── 2. Special-case: Predator uses thermal inversion ──────────────
        if char_key == "predator":
            return self._apply_predator(frame, body_mask, rgb)

        # ── 3. Special-case: Ghost uses ethereal transparency ─────────────
        if char_key == "ghost":
            return self._apply_ghost(frame, body_mask, rgb, cfg)

        # ── 4. Standard: hue-shift the body ──────────────────────────────
        shifted = self._color_shift(frame, cfg)
        mask3   = np.stack([(body_mask / 255.0).astype(np.float32)] * 3, axis=2)
        result  = (shifted.astype(np.float32) * mask3 +
                   frame.astype(np.float32)   * (1.0 - mask3)).astype(np.uint8)

        # ── 5. Pose markings ──────────────────────────────────────────────
        if self.pose:
            pose_res = self.pose.process(rgb)
            if pose_res.pose_landmarks:
                self._draw_markings(result, pose_res.pose_landmarks, w, h, cfg, char_key)

        # ── 6. Face mesh markings ─────────────────────────────────────────
        if self.face_mesh:
            fm = self.face_mesh.process(rgb)
            if fm.multi_face_landmarks:
                for fl in fm.multi_face_landmarks:
                    self._draw_face_markings(result, fl, w, h, cfg, char_key)

        # ── 7. Rim light ──────────────────────────────────────────────────
        result = self._add_rim(result, body_mask, cfg["rim_color"])

        return result

    # ── Segmentation helper ────────────────────────────────────────────────

    def _get_body_mask(self, rgb, h, w):
        if not self.segmenter:
            return None
        seg = self.segmenter.process(rgb)
        if seg.segmentation_mask is None:
            return None
        _, binary = cv2.threshold(seg.segmentation_mask, 0.55, 255, cv2.THRESH_BINARY)
        binary = cv2.GaussianBlur(binary.astype(np.uint8), (15, 15), 0)
        _, mask = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        return mask

    # ── Color shift ────────────────────────────────────────────────────────

    def _color_shift(self, frame: np.ndarray, cfg: dict) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = cfg["hue"]
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + cfg["sat_boost"], 0, 255)
        hsv[:, :, 2] = np.clip(
            (hsv[:, :, 2].astype(np.float32) * cfg["val_scale"]).astype(np.int16),
            20, 255
        )
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _color_shift_fallback(self, frame, cfg):
        """Whole-frame shift when no segmentation."""
        if cfg["hue"] is None:
            return self._thermal_invert(frame)
        return self._color_shift(frame, cfg)

    # ── Special character effects ──────────────────────────────────────────

    def _apply_predator(self, frame, body_mask, rgb):
        """Thermal imaging + scan-line grid + heat glow."""
        # Thermal colour map: invert luminance, map to COLORMAP_JET
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        thermal = cv2.applyColorMap(inverted, cv2.COLORMAP_JET)

        # Apply thermal only to person
        mask3  = np.stack([(body_mask / 255.0).astype(np.float32)] * 3, axis=2)
        result = (thermal.astype(np.float32) * mask3 +
                  frame.astype(np.float32)   * (1.0 - mask3)).astype(np.uint8)

        # Overlay scan grid on body region
        h, w = frame.shape[:2]
        grid = result.copy()
        for y in range(0, h, 8):
            cv2.line(grid, (0, y), (w, y), (0, 40, 0), 1)
        result = cv2.addWeighted(result, 0.85, grid, 0.15, 0)

        # Orange rim glow
        result = self._add_rim(result, body_mask, (0, 80, 220))
        return result

    def _apply_ghost(self, frame, body_mask, rgb, cfg):
        """Pale, semi-transparent spectral appearance."""
        # Desaturate + brighten person body
        shifted = self._color_shift(frame, cfg)

        # Make the ghost body translucent by blending 60/40 with frame
        mask3   = np.stack([(body_mask / 255.0).astype(np.float32)] * 3, axis=2)
        body_t  = (shifted.astype(np.float32) * 0.6 +
                   frame.astype(np.float32)   * 0.4).astype(np.uint8)
        result  = (body_t.astype(np.float32) * mask3 +
                   frame.astype(np.float32)  * (1.0 - mask3)).astype(np.uint8)

        # Soft white aura around entire body
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        aura_mask = cv2.dilate(body_mask, k, iterations=2)
        aura_blur = cv2.GaussianBlur(aura_mask, (31, 31), 0)
        aura = np.zeros_like(frame)
        for c, frac in zip([0, 1, 2], [0.9, 0.85, 0.7]):  # white-blue
            aura[:, :, c] = (aura_blur.astype(np.float32) * frac).astype(np.uint8)
        result = cv2.add(result, aura // 3)

        # Pose markings in white
        if self.pose:
            pr = self.pose.process(rgb)
            if pr.pose_landmarks:
                self._draw_markings(result, pr.pose_landmarks,
                                    frame.shape[1], frame.shape[0], cfg, "ghost")
        return result

    # ── Markings ────────────────────────────────────────────────────────────

    def _draw_markings(self, frame, pose_lm, w, h, cfg, char_key):
        """Draw bioluminescent / character-specific body markings."""
        lms = pose_lm.landmark
        gp  = cfg["glow_primary"]
        gs  = cfg["glow_secondary"]

        def pt(idx):
            lm = lms[idx]
            return (int(lm.x * w), int(lm.y * h)), lm.visibility

        # Core skeleton connections
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),  # torso box
            (11, 13), (12, 14), (13, 15), (14, 16),  # arms
            (23, 25), (24, 26),                       # upper legs
        ]

        line_width = 2 if char_key in ("ghost", "groot") else 3

        for (a, b) in connections:
            pa, va = pt(a)
            pb, vb = pt(b)
            if va < 0.35 or vb < 0.35:
                continue
            cv2.line(frame, pa, pb, gs,            line_width + 2, cv2.LINE_AA)
            cv2.line(frame, pa, pb, gp,            line_width,     cv2.LINE_AA)

        # Joint dots
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]:
            p, vis = pt(idx)
            if vis < 0.35:
                continue
            cv2.circle(frame, p, 5,  gs, -1, cv2.LINE_AA)
            cv2.circle(frame, p, 9,  gp,  1, cv2.LINE_AA)

        # Character-specific extras
        if char_key == "thanos":
            # Infinity stones glow — 6 dots across knuckles / joints
            stone_colors_bgr = [
                (  0,   0, 200),  # Space (blue)
                ( 30,  80, 255),  # Reality (red-orange)
                (  0, 180, 255),  # Power (purple→orange)
                (  0, 220, 255),  # Mind (yellow)
                ( 40, 255, 100),  # Time (green)
                (200, 100, 255),  # Soul (orange)
            ]
            stone_indices = [15, 16, 13, 14, 11, 12]
            for idx, color in zip(stone_indices, stone_colors_bgr):
                p, vis = pt(idx)
                if vis < 0.3: continue
                cv2.circle(frame, p, 8,  color,  -1, cv2.LINE_AA)
                cv2.circle(frame, p, 12, color,   1, cv2.LINE_AA)

        if char_key == "hulk":
            # Vein lines along forearms (elbow→wrist)
            for (elbow, wrist) in [(13, 15), (14, 16)]:
                pe, ve = pt(elbow)
                pw, vw = pt(wrist)
                if ve < 0.35 or vw < 0.35: continue
                # Two parallel offset veins
                dx = pw[0] - pe[0]; dy = pw[1] - pe[1]
                norm = max(1, int((dx**2 + dy**2)**0.5))
                ox, oy = int(-dy/norm * 6), int(dx/norm * 6)
                cv2.line(frame,
                    (pe[0]+ox, pe[1]+oy), (pw[0]+ox, pw[1]+oy),
                    (80, 255, 100), 1, cv2.LINE_AA)
                cv2.line(frame,
                    (pe[0]-ox, pe[1]-oy), (pw[0]-ox, pw[1]-oy),
                    (80, 255, 100), 1, cv2.LINE_AA)

    def _draw_face_markings(self, frame, face_lm, w, h, cfg, char_key):
        """Character-specific face markings."""
        lms = face_lm.landmark
        gp  = cfg["glow_primary"]

        def pt(idx):
            lm = lms[idx]
            return (int(lm.x * w), int(lm.y * h))

        if char_key in ("navi", "groot"):
            # Diagonal cheek stripes
            for (a, b) in [(234,93),(127,162),(454,323),(356,389)]:
                cv2.line(frame, pt(a), pt(b), (220,255,180), 2, cv2.LINE_AA)
                cv2.line(frame, pt(a), pt(b), gp,            1, cv2.LINE_AA)
            # Forehead dots
            for idx in [10, 151, 9, 8]:
                cv2.circle(frame, pt(idx), 3, gp, -1, cv2.LINE_AA)

        elif char_key == "hulk":
            # Brow furrow lines
            for (a, b) in [(70,63),(336,296)]:
                cv2.line(frame, pt(a), pt(b), gp, 2, cv2.LINE_AA)

        elif char_key == "thanos":
            # Chin crease lines
            for (a, b) in [(172,136),(397,365)]:
                cv2.line(frame, pt(a), pt(b), (180,100,255), 2, cv2.LINE_AA)
            # Jaw ridges
            for idx in [172, 136, 150, 397, 365, 379]:
                cv2.circle(frame, pt(idx), 3, (160,80,220), -1, cv2.LINE_AA)

        elif char_key == "ghost":
            # Eye hollow glow
            for eye in [[33,160,158,133,153,145], [362,387,385,263,373,374]]:
                poly = np.array([pt(i) for i in eye], dtype=np.int32)
                cv2.polylines(frame, [poly], True, (255,255,255), 1, cv2.LINE_AA)

        # Nose bridge (all characters)
        for (a, b) in zip([168, 6, 197], [6, 197, 195]):
            cv2.line(frame, pt(a), pt(b), gp, 1, cv2.LINE_AA)

    # ── Rim light ──────────────────────────────────────────────────────────

    def _add_rim(self, frame, mask, color_bgr):
        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        dilated = cv2.dilate(mask, k, iterations=1)
        edge    = cv2.subtract(dilated, mask)
        edge_b  = cv2.GaussianBlur(edge, (11,11), 0).astype(np.float32) / 255.0
        rim     = np.zeros_like(frame, dtype=np.float32)
        for c, val in enumerate(color_bgr):
            rim[:, :, c] = edge_b * val
        return np.clip(frame.astype(np.float32) + rim, 0, 255).astype(np.uint8)

    # ══════════════════════════════════════════════════════════════════════════
    # Other Modes
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_minecraft(self, frame):
        h, w = frame.shape[:2]
        bs = 16
        s = cv2.resize(frame, (w//bs, h//bs), interpolation=cv2.INTER_LINEAR)
        p = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
        g = p.copy()
        for x in range(0, w, bs): cv2.line(g, (x,0), (x,h), (0,0,0), 1)
        for y in range(0, h, bs): cv2.line(g, (0,y), (w,y), (0,0,0), 1)
        return cv2.addWeighted(p, 0.85, g, 0.15, 0)

    def _apply_cyberpunk(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        neon  = np.zeros_like(frame)
        neon[:,:,0] = edges
        neon[:,:,2] = edges
        neon[:,:,1] = (edges*0.5).astype(np.uint8)
        dark   = cv2.convertScaleAbs(frame, alpha=0.4, beta=0)
        result = cv2.add(dark, neon)
        result[:,:,1] = np.clip(result[:,:,1].astype(np.int32)+20, 0, 255).astype(np.uint8)
        return result

    def _apply_faceswap(self, frame, face_key):
        if not INSIGHTFACE_AVAILABLE or not self.face_app or not self.swapper:
            return self._faceswap_fallback(frame, face_key)
        entry = self.face_catalog.get(face_key)
        if not entry: return frame
        tf = self.face_app.get(entry["img"])
        if not tf: return frame
        lf = self.face_app.get(frame)
        if not lf: return frame
        result = frame.copy()
        for f in lf:
            result = self.swapper.get(result, f, tf[0], paste_back=True)
        return result

    def _faceswap_fallback(self, frame, face_key):
        entry = self.face_catalog.get(face_key)
        if not entry: return frame
        h, w = frame.shape[:2]
        cv2.putText(frame, f"[{entry['label']}]", (20, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,180), 2)
        return frame
