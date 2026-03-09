"""
face_processor.py — AMD Adapt Kiosk
Clean live feed, person selection, face recognition with persistent names.
"""

import cv2
import numpy as np
import threading
import time
import json
import pickle
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

CHAR_CONFIG = {
    "navi":     {"label":"Na'vi",    "subtitle":"Avatar · Pandora",     "emoji":"🔵","color":"#00d4ff","bg":"pandora"},
    "hulk":     {"label":"Hulk",     "subtitle":"Gamma · Avengers",     "emoji":"💚","color":"#22dd44","bg":"gamma"},
    "thanos":   {"label":"Thanos",   "subtitle":"Infinity · Mad Titan", "emoji":"💜","color":"#9b30ff","bg":"space"},
    "predator": {"label":"Predator", "subtitle":"Thermal · Cloaked",    "emoji":"👁️","color":"#ff6600","bg":"thermal"},
    "ghost":    {"label":"Ghost",    "subtitle":"Spectral · Ethereal",  "emoji":"👻","color":"#aaddff","bg":"spirit"},
    "groot":    {"label":"Groot",    "subtitle":"Guardians · Flora",    "emoji":"🌿","color":"#8B5E3C","bg":"forest"},
}
CHARACTER_ORDER = ["navi","hulk","thanos","predator","ghost","groot"]

KNOWN_FACES_DIR  = Path("known_faces")
KNOWN_FACES_DB   = Path("known_faces/faces.json")
RECOGNIZER_FILE  = Path("known_faces/recognizer.pkl")


# ── Segmenter ─────────────────────────────────────────────────────────────────
class Segmenter:
    def __init__(self):
        self._mask  = None
        self._frame = None
        self._lock  = threading.Lock()
        self._ready = False
        if REMBG_AVAILABLE:
            threading.Thread(target=self._init_model, daemon=True).start()

    def _init_model(self):
        try:
            self._session = new_session("u2net_human_seg")
            self._ready   = True
            print("[OK] Body segmentation ready!")
            threading.Thread(target=self._run, daemon=True).start()
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


# ── FaceRecognizer ────────────────────────────────────────────────────────────
class FaceRecognizer:
    """
    LBPH-based face recognizer. Learns faces on-demand when user names them.
    Persists across restarts via pickle + JSON.
    """
    CONFIDENCE_THRESHOLD = 75   # lower = stricter match

    def __init__(self):
        KNOWN_FACES_DIR.mkdir(exist_ok=True)
        self._lock       = threading.Lock()
        self._names      = {}       # label_id (int) → name (str)
        self._samples    = {}       # label_id → list of gray face arrays
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._trained    = False
        self._load()

    def _load(self):
        if KNOWN_FACES_DB.exists():
            try:
                data = json.loads(KNOWN_FACES_DB.read_text())
                self._names = {int(k): v for k, v in data.items()}
            except Exception:
                pass
        if RECOGNIZER_FILE.exists() and self._names:
            try:
                with open(RECOGNIZER_FILE, "rb") as f:
                    self._recognizer = pickle.load(f)
                self._trained = True
                print(f"[OK] Face recognizer loaded — {len(self._names)} known face(s)")
            except Exception:
                pass

    def _save(self):
        KNOWN_FACES_DIR.mkdir(exist_ok=True)
        KNOWN_FACES_DB.write_text(json.dumps(self._names))
        try:
            with open(RECOGNIZER_FILE, "wb") as f:
                pickle.dump(self._recognizer, f)
        except Exception:
            pass

    def _next_id(self):
        return max(self._names.keys(), default=-1) + 1

    def learn(self, face_img, name):
        """Add a face sample and retrain. face_img is BGR crop."""
        with self._lock:
            # Find existing label for this name, or create new
            label = next((k for k,v in self._names.items() if v == name), None)
            if label is None:
                label = self._next_id()
                self._names[label] = name
                self._samples[label] = []

            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            gray = cv2.equalizeHist(gray)

            # Store up to 20 samples per person
            if label not in self._samples:
                self._samples[label] = []
            self._samples[label].append(gray)
            if len(self._samples[label]) > 20:
                self._samples[label].pop(0)

            self._retrain()
            self._save()
            return label

    def _retrain(self):
        all_imgs, all_labels = [], []
        for label, imgs in self._samples.items():
            for img in imgs:
                all_imgs.append(img)
                all_labels.append(label)
        if all_imgs:
            self._recognizer.train(all_imgs, np.array(all_labels))
            self._trained = True

    def recognize(self, face_img):
        """Returns (name, confidence) or (None, None) if unknown."""
        if not self._trained:
            return None, None
        with self._lock:
            try:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (100, 100))
                gray = cv2.equalizeHist(gray)
                label, conf = self._recognizer.predict(gray)
                if conf < self.CONFIDENCE_THRESHOLD:
                    return self._names.get(label, "Unknown"), conf
            except Exception:
                pass
        return None, None

    def get_known_names(self):
        with self._lock:
            return dict(self._names)

    def forget(self, name):
        with self._lock:
            label = next((k for k,v in self._names.items() if v == name), None)
            if label is not None:
                del self._names[label]
                self._samples.pop(label, None)
                self._retrain()
                self._save()


# ── FaceProcessor ─────────────────────────────────────────────────────────────
class FaceProcessor:
    def __init__(self, faces_dir="faces"):
        self.faces_dir = Path(faces_dir)
        self.mode      = "passthrough"
        self.character = "navi"
        self.face_key  = None
        self._lock     = threading.Lock()
        self.catalog   = {}
        self._load_catalog()

        self.seg        = Segmenter()
        self.recognizer = FaceRecognizer()

        self.face_det = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detection state
        self._detected_faces   = []   # [(x,y,w,h), ...]
        self._recognized_names = {}   # face index → name string
        self._selected_people  = set()
        self._frame_size       = (0, 0)

        # Recognition runs in background thread
        self._recog_frame = None
        self._recog_lock  = threading.Lock()
        threading.Thread(target=self._recog_loop, daemon=True).start()

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
                self.catalog[key] = {"label":p.stem.replace("_"," ").title(),
                                     "category":cat.name.title(),"img":img,"path":str(p)}
        print(f"[OK] {len(self.catalog)} face(s) in catalog")

    def reload_catalog(self):
        with self._lock:
            self.catalog.clear()
            self._load_catalog()

    def get_catalog(self):
        return [{"key":k,"label":v["label"],"category":v["category"]}
                for k,v in self.catalog.items()]

    def get_characters(self):
        return [{"key":k,"label":CHAR_CONFIG[k]["label"],
                 "subtitle":CHAR_CONFIG[k]["subtitle"],
                 "emoji":CHAR_CONFIG[k]["emoji"],
                 "color":CHAR_CONFIG[k]["color"],
                 "swap_ready":True,"ref_photo":""}
                for k in CHARACTER_ORDER]

    def set_mode(self, mode, face_key=None, character=None):
        with self._lock:
            self.mode = mode
            if face_key:  self.face_key = face_key
            if character and character in CHAR_CONFIG:
                self.character = character

    # ── Person selection ──────────────────────────────────────────────────────
    def get_detected_faces(self):
        with self._lock:
            faces = list(self._detected_faces)
            sel   = set(self._selected_people)
            names = dict(self._recognized_names)
            fw, fh = self._frame_size
        return [{"index":i,"x":x,"y":y,"w":w,"h":h,
                 "selected":i in sel,
                 "name": names.get(i),
                 "frame_w":fw,"frame_h":fh}
                for i,(x,y,w,h) in enumerate(faces)]

    def toggle_person(self, click_x, click_y, frame_w, frame_h):
        with self._lock:
            faces  = list(self._detected_faces)
            fw, fh = self._frame_size
        if fw == 0 or fh == 0: return
        sx = int(click_x * fw / frame_w)
        sy = int(click_y * fh / frame_h)
        pad = 20
        for i, (x,y,w,h) in enumerate(faces):
            if (x-pad) <= sx <= (x+w+pad) and (y-pad) <= sy <= (y+h+pad):
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
        with self._lock:
            faces = list(self._detected_faces)
            sel   = set(self._selected_people)
        h, w = frame.shape[:2]
        if not sel:
            return np.ones((h,w), dtype=np.float32)
        mask = np.zeros((h,w), dtype=np.float32)
        for i in sel:
            if i < len(faces):
                x,y,fw,fh = faces[i]
                y1 = max(0, y - fh//2);   y2 = min(h, y + fh*3)
                x1 = max(0, x - fw//2);   x2 = min(w, x + fw + fw//2)
                mask[y1:y2, x1:x2] = 1.0
        return mask

    # ── Face naming ───────────────────────────────────────────────────────────
    def name_selected_face(self, index, name, frame):
        """Extract face crop at given index and teach the recognizer."""
        with self._lock:
            faces = list(self._detected_faces)
        if index >= len(faces): return False
        x, y, w, h = faces[index]
        # Pad crop slightly
        pad = 10
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(frame.shape[1], x+w+pad)
        y2 = min(frame.shape[0], y+h+pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return False
        self.recognizer.learn(crop, name)
        print(f"[OK] Learned face: {name}")
        return True

    def get_known_names(self):
        return self.recognizer.get_known_names()

    def forget_face(self, name):
        self.recognizer.forget(name)

    # ── Background recognition loop ───────────────────────────────────────────
    def _recog_loop(self):
        """Continuously recognize faces in background thread."""
        while True:
            with self._recog_lock:
                frame = self._recog_frame
            if frame is not None:
                with self._lock:
                    faces = list(self._detected_faces)
                new_names = {}
                for i, (x,y,w,h) in enumerate(faces):
                    pad = 10
                    x1 = max(0,x-pad); y1 = max(0,y-pad)
                    x2 = min(frame.shape[1],x+w+pad)
                    y2 = min(frame.shape[0],y+h+pad)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        name, conf = self.recognizer.recognize(crop)
                        if name:
                            new_names[i] = name
                with self._lock:
                    self._recognized_names = new_names
            time.sleep(0.25)  # ~4fps recognition

    # ── process_frame ─────────────────────────────────────────────────────────
    def process_frame(self, frame):
        if frame is None: return frame
        self.seg.update(frame)
        h, w = frame.shape[:2]
        with self._lock:
            self._frame_size = (w, h)

        # Detect faces
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        with self._lock:
            self._detected_faces = [tuple(f) for f in faces] if len(faces) > 0 else []
            self._selected_people = {
                i for i in self._selected_people if i < len(self._detected_faces)}
            sel   = set(self._selected_people)
            names = dict(self._recognized_names)
            faces_snap = list(self._detected_faces)

        # Feed frame to recognition loop
        with self._recog_lock:
            self._recog_frame = frame.copy()

        result = frame.copy()
        self._draw_overlay(result, faces_snap, sel, names)
        return result

    def _draw_overlay(self, frame, faces, selected, names):
        for i, (x,y,w,h) in enumerate(faces):
            is_sel = i in selected
            name   = names.get(i)
            pad    = 10
            x1,y1  = max(0,x-pad), max(0,y-pad)
            x2,y2  = min(frame.shape[1],x+w+pad), min(frame.shape[0],y+h+pad)

            if is_sel:
                color, thick = (0,220,255), 3
                # Glow
                cv2.rectangle(frame,(x1-2,y1-2),(x2+2,y2+2),(0,100,120),6)
            else:
                color, thick = (70,70,70), 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,thick)

            # Name tag — bright gold if recognized, dim if unknown
            if name:
                label      = name
                label_col  = (0,210,255)  # cyan-gold
                bg_col     = (0,80,100)
            elif is_sel:
                label      = f"Person {i+1} ✓"
                label_col  = (0,220,255)
                bg_col     = (0,60,80)
            else:
                label      = f"Person {i+1}"
                label_col  = (70,70,70)
                bg_col     = None

            # Draw label background pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            lx = x1
            ly = max(th+6, y1-6)
            if bg_col:
                cv2.rectangle(frame,(lx-2,ly-th-4),(lx+tw+6,ly+4),bg_col,-1)
            cv2.putText(frame, label, (lx+2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_col, 2, cv2.LINE_AA)

        if len(faces) > 0 and not selected:
            cv2.putText(frame, "Tap a person to select",
                        (10, frame.shape[0]-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,200), 2, cv2.LINE_AA)
