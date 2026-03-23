"""
face_processor.py — AI Avatar Kiosk
Face detection, recognition, person selection, and live preview overlay.

Simplified for 4 avatar styles: Avatar, Claymation, Anime, Ghost.
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

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MP_AVAILABLE = False
mp_pose = None
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        mp_pose = mp.solutions.pose
        MP_AVAILABLE = True
        print("[OK] MediaPipe pose available")
except ImportError:
    print("[WARN] MediaPipe not available")

# ── Style config (display info only — generation params in generator.py) ──────
STYLE_CONFIG = {
    "avatar":      {"label": "Avatar",      "subtitle": "Blue Alien · Pandora",       "emoji": "🔵", "color": "#00d4ff"},
    "anime":       {"label": "Anime",       "subtitle": "Cel Shaded · Manga",         "emoji": "✨", "color": "#ff69b4"},
    "cyberpunk":   {"label": "Cyberpunk",   "subtitle": "Neon · Chrome · Future",      "emoji": "🤖", "color": "#ff00ff"},
    "claymation":  {"label": "Claymation",  "subtitle": "Clay · Stop Motion",          "emoji": "🎨", "color": "#e67e22"},
    "oilpainting": {"label": "Oil Painting","subtitle": "Renaissance · Brushstrokes",  "emoji": "🖼️", "color": "#c8a24e"},
    "comicbook":   {"label": "Comic Book",  "subtitle": "Bold Ink · Halftone",         "emoji": "💥", "color": "#e74c3c"},
    "pixelart":    {"label": "Pixel Art",   "subtitle": "Retro · 16-Bit",              "emoji": "👾", "color": "#27ae60"},
    "steampunk":   {"label": "Steampunk",   "subtitle": "Brass · Gears · Victorian",   "emoji": "⚙️", "color": "#d4a574"},
    "watercolor":  {"label": "Watercolor", "subtitle": "Soft · Delicate",              "emoji": "🌺", "color": "#88c8e8"},
    "zombie":      {"label": "Zombie",     "subtitle": "Horror · Undead",              "emoji": "🧟", "color": "#5a7247"},
    "fantasy":     {"label": "Fantasy Elf","subtitle": "Magic · Enchanted",            "emoji": "🧙", "color": "#7b68ee"},
    "popart":      {"label": "Pop Art",    "subtitle": "Bold · Graphic",               "emoji": "🎨", "color": "#ff4081"},
    "ice":         {"label": "Ice & Frost","subtitle": "Frozen · Crystal",             "emoji": "❄️", "color": "#a0d8ef"},
    "neon":        {"label": "Neon Glow",  "subtitle": "Electric · UV",                "emoji": "🔮", "color": "#b400ff"},
    "labubu":      {"label": "Labubu",     "subtitle": "Vinyl Toy · Kawaii",           "emoji": "🧸", "color": "#ffb6c1"},
    "wizard":      {"label": "Wizard",     "subtitle": "Magic · Hogwarts",             "emoji": "🪄", "color": "#7b3f00"},
}
STYLE_ORDER = [
    "avatar", "anime", "cyberpunk", "claymation", "oilpainting", "comicbook",
    "pixelart", "steampunk", "watercolor", "zombie", "fantasy", "popart",
    "ice", "neon", "labubu", "wizard",
]

KNOWN_FACES_DIR = Path("known_faces")
KNOWN_FACES_DB  = Path("known_faces/faces.json")
RECOGNIZER_FILE = Path("known_faces/recognizer.pkl")
KNOWN_GENDER_DB = Path("known_faces/genders.json")


# ── PoseTracker ───────────────────────────────────────────────────────────────
class PoseTracker:
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (11,12),(11,23),(12,24),(23,24),
        (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
        (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
        (23,25),(25,27),(27,29),(27,31),(29,31),
        (24,26),(26,28),(28,30),(28,32),(30,32),
    ]

    def __init__(self):
        self._landmarks = None
        self._skeleton = None
        self._frame = None
        self._lock = threading.Lock()
        if MP_AVAILABLE:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        pose = mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        while True:
            with self._lock:
                frame = self._frame
            if frame is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    if results.pose_landmarks:
                        h, w = frame.shape[:2]
                        lms = results.pose_landmarks.landmark
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                        skel = self._draw_skeleton(pts, h, w)
                        with self._lock:
                            self._landmarks = pts
                            self._skeleton = skel
                    else:
                        with self._lock:
                            self._landmarks = None
                            self._skeleton = None
                except Exception:
                    pass
            time.sleep(1/15)

    def update(self, frame):
        if MP_AVAILABLE:
            with self._lock:
                self._frame = cv2.resize(frame, (640, 480))

    def _draw_skeleton(self, pts, h, w):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for a, b in self.CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (255, 255, 255), 3, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 5, (200, 200, 200), -1, cv2.LINE_AA)
        return img

    def get_body_mask(self, h, w):
        with self._lock:
            lms = list(self._landmarks) if self._landmarks else None
        if not lms:
            return None
        pts = np.array([[x, y] for x, y in lms], dtype=np.int32)
        pts[:, 0] = (pts[:, 0] * w / 640).astype(int)
        pts[:, 1] = (pts[:, 1] * h / 480).astype(int)
        pts = np.clip(pts, 0, [w-1, h-1])
        mask = np.zeros((h, w), dtype=np.float32)
        hull = cv2.convexHull(pts)
        center = hull.mean(axis=0)
        expanded = ((hull - center) * 1.4 + center).astype(np.int32)
        cv2.fillConvexPoly(mask, expanded, 1.0)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        return mask


# ── Face recognizer backends ──────────────────────────────────────────────────
_CONTRIB_AVAILABLE = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")

if not _CONTRIB_AVAILABLE:
    try:
        from sklearn.neighbors import KNeighborsClassifier
        _SKLEARN_AVAILABLE = True
        print("[OK] Face recognition: sklearn KNN")
    except ImportError:
        _SKLEARN_AVAILABLE = False
        print("[WARN] Face recognition disabled")
else:
    _SKLEARN_AVAILABLE = False
    print("[OK] Face recognition: opencv LBPH")


class _LBPHRecognizer:
    def __init__(self):
        self._rec = cv2.face.LBPHFaceRecognizer_create()
        self._trained = False

    def train(self, imgs, labels):
        self._rec.train(imgs, labels)
        self._trained = True

    def predict(self, img):
        if not self._trained:
            return -1, 9999.0
        label, conf = self._rec.predict(img)
        return label, conf

    @property
    def trained(self):
        return self._trained


class _KNNRecognizer:
    def __init__(self):
        self._knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
        self._trained = False

    def _features(self, img):
        return img.flatten().astype(np.float32) / 255.0

    def train(self, imgs, labels):
        X = np.array([self._features(img) for img in imgs])
        y = np.array(labels)
        self._knn.fit(X, y)
        self._trained = True

    def predict(self, img):
        if not self._trained:
            return -1, 9999.0
        x = self._features(img).reshape(1, -1)
        lbl = int(self._knn.predict(x)[0])
        dist, _ = self._knn.kneighbors(x, n_neighbors=1)
        conf = float(dist[0][0]) * 1000
        return lbl, conf

    @property
    def trained(self):
        return self._trained


class _NoopRecognizer:
    def train(self, imgs, labels): pass
    def predict(self, img): return -1, 9999.0
    @property
    def trained(self): return False


def _make_recognizer():
    if _CONTRIB_AVAILABLE:
        return _LBPHRecognizer()
    if _SKLEARN_AVAILABLE:
        return _KNNRecognizer()
    return _NoopRecognizer()


# ── FaceRecognizer ────────────────────────────────────────────────────────────
class FaceRecognizer:
    CONFIDENCE_THRESHOLD = 110  # Higher = more forgiving (KNN dist*1000, LBPH raw)

    def __init__(self):
        KNOWN_FACES_DIR.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._names = {}
        self._genders = {}
        self._samples = {}
        self._recognizer = _make_recognizer()
        self._load()

    def _load(self):
        if KNOWN_FACES_DB.exists():
            try:
                data = json.loads(KNOWN_FACES_DB.read_text())
                self._names = {int(k): v for k, v in data.items()}
            except Exception:
                pass
        if KNOWN_GENDER_DB.exists():
            try:
                self._genders = json.loads(KNOWN_GENDER_DB.read_text())
            except Exception:
                pass
        if RECOGNIZER_FILE.exists() and self._names:
            try:
                with open(RECOGNIZER_FILE, "rb") as f:
                    loaded = pickle.load(f)
                if hasattr(loaded, "predict") and hasattr(loaded, "trained"):
                    self._recognizer = loaded
                    print(f"[OK] Recognizer loaded — {len(self._names)} face(s)")
            except Exception as e:
                print(f"[WARN] Could not load recognizer: {e}")

    def _save(self):
        KNOWN_FACES_DIR.mkdir(exist_ok=True)
        KNOWN_FACES_DB.write_text(json.dumps(self._names))
        KNOWN_GENDER_DB.write_text(json.dumps(self._genders))
        try:
            with open(RECOGNIZER_FILE, "wb") as f:
                pickle.dump(self._recognizer, f)
        except Exception as e:
            print(f"[WARN] Could not save recognizer: {e}")

    def _next_id(self):
        return max(self._names.keys(), default=-1) + 1

    def learn(self, face_img, name, gender="unknown"):
        with self._lock:
            label = next((k for k, v in self._names.items() if v == name), None)
            if label is None:
                label = self._next_id()
                self._names[label] = name
                self._samples[label] = []

            if gender != "unknown":
                self._genders[name] = gender
            elif name not in self._genders:
                self._genders[name] = "unknown"

            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            gray = cv2.equalizeHist(gray)

            augmented = [gray, cv2.flip(gray, 1)]
            for delta in (-25, -12, 12, 25):
                shifted = np.clip(gray.astype(np.int16) + delta, 0, 255).astype(np.uint8)
                augmented.append(shifted)
            augmented.append(cv2.GaussianBlur(gray, (3, 3), 0))
            augmented.append(cv2.GaussianBlur(gray, (5, 5), 0))

            if label not in self._samples:
                self._samples[label] = []
            self._samples[label].extend(augmented)
            if len(self._samples[label]) > 80:
                self._samples[label] = self._samples[label][-80:]

            self._retrain()
            self._save()
            print(f"[FaceRecognizer] Learned '{name}' (label={label}, "
                  f"samples={len(self._samples[label])}, gender={gender})")
            return label

    def _retrain(self):
        imgs, labels = [], []
        for label, samples in self._samples.items():
            for img in samples:
                imgs.append(img)
                labels.append(label)
        if imgs:
            self._recognizer.train(imgs, np.array(labels, dtype=np.int32))

    def recognize(self, face_img):
        if not self._recognizer.trained:
            return None, None
        with self._lock:
            try:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (100, 100))
                gray = cv2.equalizeHist(gray)
                label, conf = self._recognizer.predict(gray)
                if conf < self.CONFIDENCE_THRESHOLD:
                    return self._names.get(label), conf
            except Exception:
                pass
        return None, None

    def get_gender_for_name(self, name):
        with self._lock:
            return self._genders.get(name, "unknown")

    def get_known_names(self):
        with self._lock:
            return dict(self._names)

    def forget(self, name):
        with self._lock:
            label = next((k for k, v in self._names.items() if v == name), None)
            if label is not None:
                del self._names[label]
                self._genders.pop(name, None)
                self._samples.pop(label, None)
                self._retrain()
                self._save()


# ── FaceProcessor ─────────────────────────────────────────────────────────────
class FaceProcessor:
    def __init__(self, faces_dir="faces"):
        self.faces_dir = Path(faces_dir)
        self.mode = "passthrough"
        self.character = "avatar"
        self.face_key = None
        self._lock = threading.Lock()
        self.catalog = {}

        self.poser = PoseTracker()
        self.recognizer = FaceRecognizer()

        self.face_det = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self._detected_faces = []
        self._recognized_names = {}
        self._selected_people = set()
        self._frame_size = (0, 0)
        self._crop_for_naming = None

        self._det_frame = None
        self._det_lock = threading.Lock()
        self._recog_frame = None
        self._recog_lock = threading.Lock()
        threading.Thread(target=self._detect_loop, daemon=True).start()
        threading.Thread(target=self._recog_loop, daemon=True).start()

    def _detect_loop(self):
        while True:
            with self._det_lock:
                frame = self._det_frame
            if frame is not None:
                try:
                    h, w = frame.shape[:2]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_det.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
                    body_boxes = []
                    for (fx, fy, fw, fh) in (faces if len(faces) > 0 else []):
                        bx = max(0, fx - int(fw * 0.6))
                        by = max(0, fy - int(fh * 0.3))
                        bw = min(w - bx, int(fw * 2.2))
                        bh = min(h - by, int(fh * 5.5))
                        body_boxes.append((bx, by, bw, bh))
                    with self._lock:
                        self._detected_faces = body_boxes
                        self._selected_people = {
                            i for i in self._selected_people
                            if i < len(body_boxes)}
                        self._frame_size = (w, h)
                except Exception:
                    pass
            time.sleep(0.10)

    def get_characters(self):
        return [
            {
                "key": k,
                "label": STYLE_CONFIG[k]["label"],
                "subtitle": STYLE_CONFIG[k]["subtitle"],
                "emoji": STYLE_CONFIG[k]["emoji"],
                "color": STYLE_CONFIG[k]["color"],
            }
            for k in STYLE_ORDER
        ]

    def set_mode(self, mode, face_key=None, character=None):
        with self._lock:
            self.mode = mode
            if face_key:
                self.face_key = face_key
            if character and character in STYLE_CONFIG:
                self.character = character

    # ── Person selection ──────────────────────────────────────────────────────
    def get_detected_faces(self):
        with self._lock:
            faces = list(self._detected_faces)
            sel = set(self._selected_people)
            names = dict(self._recognized_names)
            fw, fh = self._frame_size
        return [
            {
                "index": i,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "selected": i in sel,
                "name": names.get(i),
                "frame_w": int(fw),
                "frame_h": int(fh),
            }
            for i, (x, y, w, h) in enumerate(faces)
        ]

    def _extract_and_store_crop(self, frame, x, y, w, h):
        face_crop = self._extract_face_from_body(frame, x, y, w, h)
        if face_crop is not None and face_crop.size > 0:
            with self._lock:
                self._crop_for_naming = face_crop.copy()

    def toggle_person(self, click_x, click_y, frame_w, frame_h, frame=None):
        with self._lock:
            faces = list(self._detected_faces)
            fw, fh = self._frame_size
        if fw == 0 or fh == 0:
            return
        sx = int(click_x * fw / frame_w)
        sy = int(click_y * fh / frame_h)
        pad = 35
        for i, (x, y, w, h) in enumerate(faces):
            if (x-pad) <= sx <= (x+w+pad) and (y-pad) <= sy <= (y+h+pad):
                with self._lock:
                    if i in self._selected_people:
                        self._selected_people.discard(i)
                    else:
                        self._selected_people.add(i)
                if frame is not None:
                    self._extract_and_store_crop(frame, x, y, w, h)
                return

    def clear_selection(self):
        with self._lock:
            self._selected_people.clear()
            self._crop_for_naming = None

    def get_selected_mask(self, frame):
        with self._lock:
            faces = list(self._detected_faces)
            sel = set(self._selected_people)
        h, w = frame.shape[:2]
        if not sel:
            return np.ones((h, w), dtype=np.float32)

        pose_mask = self.poser.get_body_mask(h, w)
        if pose_mask is not None:
            return pose_mask

        mask = np.zeros((h, w), dtype=np.float32)
        for i in sel:
            if i < len(faces):
                x, y, fw, fh = faces[i]
                y1 = max(0, y - fh//2)
                y2 = min(h, y + fh * 4)
                x1 = max(0, x - fw)
                x2 = min(w, x + fw * 2)
                mask[y1:y2, x1:x2] = 1.0
        return mask

    # ── Face naming ───────────────────────────────────────────────────────────
    def name_face(self, index, name, gender="unknown"):
        with self._lock:
            crop = self._crop_for_naming
        if crop is None:
            return False
        self.recognizer.learn(crop, name, gender)
        with self._lock:
            self._crop_for_naming = None
        return True

    def name_selected_face(self, index, name, frame, gender="unknown"):
        if self.name_face(index, name, gender):
            return True
        with self._lock:
            faces = list(self._detected_faces)
            sel = set(self._selected_people)
        targets = [faces[i] for i in sel if i < len(faces)] or (faces[:1] if faces else [])
        for (x, y, w, h) in targets:
            # Extract just the face from the body box
            face_crop = self._extract_face_from_body(frame, x, y, w, h)
            if face_crop is not None and face_crop.size > 0:
                self.recognizer.learn(face_crop, name, gender)
                return True
        return False

    def get_known_names(self):
        return self.recognizer.get_known_names()

    def get_gender_for_name(self, name):
        return self.recognizer.get_gender_for_name(name)

    def forget_face(self, name):
        self.recognizer.forget(name)

    def _extract_face_from_body(self, frame, bx, by, bw, bh):
        """Extract just the face from a body bounding box."""
        fh_frame, fw_frame = frame.shape[:2]
        bx1 = max(0, bx)
        by1 = max(0, by)
        bx2 = min(fw_frame, bx + bw)
        by2 = min(fh_frame, by + bh)
        body_roi = frame[by1:by2, bx1:bx2]

        if body_roi.size == 0:
            return None

        # Re-detect face within this body box
        gray = cv2.cvtColor(body_roi, cv2.COLOR_BGR2GRAY)
        sub_faces = self.face_det.detectMultiScale(
            gray, 1.1, 4, minSize=(30, 30))
        if len(sub_faces) > 0:
            fx, fy, fw, fhh = max(sub_faces, key=lambda r: r[2] * r[3])
            face_crop = body_roi[fy:fy+fhh, fx:fx+fw]
            if face_crop.size > 0:
                return face_crop

        # Fallback: top third of body box (approximate head area)
        head_h = max(1, (by2 - by1) // 3)
        head_crop = frame[by1:by1+head_h, bx1:bx2]
        if head_crop.size > 0:
            return head_crop

        return None

    # ── Background recognition ────────────────────────────────────────────────
    def _recog_loop(self):
        while True:
            with self._recog_lock:
                frame = self._recog_frame
            if frame is not None:
                with self._lock:
                    faces = list(self._detected_faces)
                new_names = {}
                for i, (x, y, w, h) in enumerate(faces):
                    face_crop = self._extract_face_from_body(frame, x, y, w, h)
                    if face_crop is not None:
                        name, _ = self.recognizer.recognize(face_crop)
                        if name:
                            new_names[i] = name
                with self._lock:
                    self._recognized_names = new_names
            time.sleep(0.3)

    # ── process_frame ─────────────────────────────────────────────────────────
    def process_frame(self, frame):
        if frame is None:
            return frame
        # Pose tracking disabled — not needed for full-scene transform mode

        with self._det_lock:
            self._det_frame = frame
        with self._recog_lock:
            self._recog_frame = frame

        with self._lock:
            sel = set(self._selected_people)
            names = dict(self._recognized_names)
            faces_snap = list(self._detected_faces)

        result = frame.copy()
        self._draw_overlay(result, faces_snap, sel, names)
        return result

    def _draw_overlay(self, frame, faces, selected, names):
        for i, (x, y, w, h) in enumerate(faces):
            is_sel = i in selected
            name = names.get(i)
            pad = 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)

            if is_sel:
                color, thick = (0, 220, 255), 3
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 100, 120), 6)
            else:
                color, thick = (70, 70, 70), 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

            if name:
                label, label_col, bg_col = name, (0, 210, 255), (0, 80, 100)
            elif is_sel:
                label, label_col, bg_col = f"Person {i+1}", (0, 220, 255), (0, 60, 80)
            else:
                label, label_col, bg_col = f"Person {i+1}", (70, 70, 70), None

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            lx = x1
            ly = max(th+6, y1-6)
            if bg_col:
                cv2.rectangle(frame, (lx-2, ly-th-4), (lx+tw+6, ly+4), bg_col, -1)
            cv2.putText(frame, label, (lx+2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_col, 2, cv2.LINE_AA)

        if faces and not selected:
            cv2.putText(frame, "Click a person to select",
                        (10, frame.shape[0]-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 200), 2, cv2.LINE_AA)
