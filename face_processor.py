"""
face_processor.py — AMD Adapt Kiosk
Clean live feed with person detection + selection overlay.
Transform Me triggers AI generation on selected people only.
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path

# ── onnxruntime patch ─────────────────────────────────────────────────────────
try:
    import onnxruntime as _ort
    if not hasattr(_ort, 'set_default_logger_severity'):
        _ort.set_default_logger_severity = lambda level: None
except Exception:
    pass

# ── rembg ─────────────────────────────────────────────────────────────────────
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MP_AVAILABLE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        MP_AVAILABLE = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
CHAR_CONFIG = {
    "navi":     {"label":"Na'vi",    "subtitle":"Avatar · Pandora",     "emoji":"🔵","color":"#00d4ff","bg":"pandora"},
    "hulk":     {"label":"Hulk",     "subtitle":"Gamma · Avengers",     "emoji":"💚","color":"#22dd44","bg":"gamma"},
    "thanos":   {"label":"Thanos",   "subtitle":"Infinity · Mad Titan", "emoji":"💜","color":"#9b30ff","bg":"space"},
    "predator": {"label":"Predator", "subtitle":"Thermal · Cloaked",    "emoji":"👁️","color":"#ff6600","bg":"thermal"},
    "ghost":    {"label":"Ghost",    "subtitle":"Spectral · Ethereal",  "emoji":"👻","color":"#aaddff","bg":"spirit"},
    "groot":    {"label":"Groot",    "subtitle":"Guardians · Flora",    "emoji":"🌿","color":"#8B5E3C","bg":"forest"},
}
CHARACTER_ORDER = ["navi","hulk","thanos","predator","ghost","groot"]

# ── Segmenter ─────────────────────────────────────────────────────────────────
class Segmenter:
    def __init__(self):
        self._mask   = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._ready  = False
        if REMBG_AVAILABLE:
            t = threading.Thread(target=self._init_model, daemon=True)
            t.start()

    def _init_model(self):
        try:
            self._session = new_session("u2net_human_seg")
            self._ready   = True
            print("[OK] Body segmentation ready!")
            self._worker_thread = threading.Thread(target=self._run, daemon=True)
            self._worker_thread.start()
        except Exception as e:
            print(f"[WARN] Segmentation init failed: {e}")

    def _run(self):
        while True:
            with self._lock:
                frame = self._frame
            if frame is not None:
                try:
                    from PIL import Image
                    pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    out  = remove(pil, session=self._session)
                    arr  = np.array(out)
                    mask = (arr[:,:,3] / 255.0).astype(np.float32)
                    with self._lock:
                        self._mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                except Exception:
                    pass
            time.sleep(0.07)

    def update(self, frame):
        if self._ready:
            with self._lock:
                self._frame = cv2.resize(frame, (320, 240))

    def get(self, h, w):
        with self._lock:
            m = self._mask
        if m is None: return None
        return cv2.resize(m, (w, h))


# ── FaceProcessor ─────────────────────────────────────────────────────────────
class FaceProcessor:
    def __init__(self, faces_dir="faces"):
        self.faces_dir  = Path(faces_dir)
        self.mode       = "passthrough"
        self.character  = "navi"
        self.face_key   = None
        self._lock      = threading.Lock()
        self.catalog    = {}
        self._load_catalog()

        self.seg        = Segmenter()

        # Face detector for person selection
        self.face_det   = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Person selection state
        self._detected_faces  = []   # list of (x,y,w,h) in current frame
        self._selected_people = set()  # set of indices into _detected_faces
        self._frame_size      = (0, 0)  # (w, h) of current frame

    # ── Catalog ───────────────────────────────────────────────────────────────
    def _load_catalog(self):
        if not self.faces_dir.exists():
            self.faces_dir.mkdir(parents=True)
            return
        ext = {".jpg",".jpeg",".png",".webp"}
        for cat in self.faces_dir.iterdir():
            if not cat.is_dir(): continue
            for p in cat.iterdir():
                if p.suffix.lower() not in ext: continue
                img = cv2.imread(str(p))
                if img is None: continue
                key = f"{cat.name}/{p.stem}"
                self.catalog[key] = {
                    "label": p.stem.replace("_"," ").title(),
                    "category": cat.name.title(),
                    "img": img, "path": str(p)
                }
        print(f"[OK] {len(self.catalog)} face(s) in catalog")

    def reload_catalog(self):
        with self._lock:
            self.catalog.clear()
            self._load_catalog()

    def get_catalog(self):
        return [{"key":k,"label":v["label"],"category":v["category"]}
                for k,v in self.catalog.items()]

    def get_characters(self):
        return [{"key":k, "label":CHAR_CONFIG[k]["label"],
                 "subtitle":CHAR_CONFIG[k]["subtitle"],
                 "emoji":CHAR_CONFIG[k]["emoji"],
                 "color":CHAR_CONFIG[k]["color"],
                 "swap_ready":True, "ref_photo":""}
                for k in CHARACTER_ORDER]

    def set_mode(self, mode, face_key=None, character=None):
        with self._lock:
            self.mode = mode
            if face_key:  self.face_key = face_key
            if character and character in CHAR_CONFIG:
                self.character = character

    # ── Person selection ──────────────────────────────────────────────────────
    def get_detected_faces(self):
        """Return current detected face boxes as list of dicts."""
        with self._lock:
            faces = list(self._detected_faces)
            sel   = set(self._selected_people)
            fw, fh = self._frame_size
        return [{"index":i,"x":x,"y":y,"w":w,"h":h,
                 "selected": i in sel,
                 "frame_w": fw, "frame_h": fh}
                for i,(x,y,w,h) in enumerate(faces)]

    def toggle_person(self, click_x, click_y, frame_w, frame_h):
        """Toggle selection of person at click coordinates."""
        with self._lock:
            faces = list(self._detected_faces)
            fw, fh = self._frame_size

        if fw == 0 or fh == 0:
            return

        # Scale click to frame coordinates
        sx = int(click_x * fw / frame_w)
        sy = int(click_y * fh / frame_h)

        # Expand hit area by 20px
        pad = 20
        for i, (x, y, w, h) in enumerate(faces):
            if (x - pad) <= sx <= (x + w + pad) and (y - pad) <= sy <= (y + h + pad):
                with self._lock:
                    if i in self._selected_people:
                        self._selected_people.discard(i)
                    else:
                        self._selected_people.add(i)
                return

    def clear_selection(self):
        with self._lock:
            self._selected_people.clear()

    def get_selected_mask(self, frame):
        """Return a mask covering only selected people, or full frame if none selected."""
        with self._lock:
            faces = list(self._detected_faces)
            sel   = set(self._selected_people)

        h, w = frame.shape[:2]
        if not sel:
            return np.ones((h, w), dtype=np.float32)

        mask = np.zeros((h, w), dtype=np.float32)
        for i in sel:
            if i < len(faces):
                x, y, fw, fh = faces[i]
                # Expand box to cover body (2.5x height below face)
                body_y1 = max(0, y - fh//2)
                body_y2 = min(h, y + fh * 3)
                body_x1 = max(0, x - fw//2)
                body_x2 = min(w, x + fw + fw//2)
                mask[body_y1:body_y2, body_x1:body_x2] = 1.0
        return mask

    # ── process_frame ─────────────────────────────────────────────────────────
    def process_frame(self, frame):
        if frame is None:
            return frame

        self.seg.update(frame)
        h, w = frame.shape[:2]

        with self._lock:
            self._frame_size = (w, h)

        # Detect faces every frame for selection UI
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        with self._lock:
            self._detected_faces = [tuple(f) for f in faces] if len(faces) > 0 else []
            # Clamp selected indices to valid range
            self._selected_people = {
                i for i in self._selected_people
                if i < len(self._detected_faces)
            }
            sel = set(self._selected_people)
            faces_snap = list(self._detected_faces)

        # Always show clean live feed — draw selection overlay on top
        result = frame.copy()
        self._draw_selection_overlay(result, faces_snap, sel)

        return result

    def _draw_selection_overlay(self, frame, faces, selected):
        """Draw bounding boxes around detected people. Selected = bright, others = dim."""
        for i, (x, y, w, h) in enumerate(faces):
            is_sel = i in selected
            # Expand box slightly
            pad = 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)

            if is_sel:
                color     = (0, 220, 255)   # bright cyan
                thickness = 3
                label     = f"Person {i+1} SELECTED"
                # Glow effect
                cv2.rectangle(frame, (x1-2,y1-2), (x2+2,y2+2), (0,100,120), 6)
            else:
                color     = (80, 80, 80)    # dim grey
                thickness = 1
                label     = f"Person {i+1}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

            # Label above box
            lx, ly = x1, max(0, y1-10)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Instruction hint when people detected but none selected
        if len(faces) > 0 and not selected:
            cv2.putText(frame, "TAP a person to select for transformation",
                        (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,255), 2, cv2.LINE_AA)
