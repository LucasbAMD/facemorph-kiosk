"""
face_processor.py — AMD Adapt Kiosk
Full-body pose tracking via MediaPipe + face recognition via LBPH.
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

CHAR_CONFIG = {
    "navi":     {"label":"Na'vi",    "subtitle":"Avatar · Pandora",     "emoji":"🔵","color":"#00d4ff"},
    "hulk":     {"label":"Hulk",     "subtitle":"Gamma · Avengers",     "emoji":"💚","color":"#22dd44"},
    "thanos":   {"label":"Thanos",   "subtitle":"Infinity · Mad Titan", "emoji":"💜","color":"#9b30ff"},
    "predator": {"label":"Predator", "subtitle":"Thermal · Cloaked",    "emoji":"👁️","color":"#ff6600"},
    "ghost":    {"label":"Ghost",    "subtitle":"Spectral · Ethereal",  "emoji":"👻","color":"#aaddff"},
    "groot":    {"label":"Groot",    "subtitle":"Guardians · Flora",    "emoji":"🌿","color":"#8B5E3C"},
}
CHARACTER_ORDER = ["navi","hulk","thanos","predator","ghost","groot"]

KNOWN_FACES_DIR = Path("known_faces")
KNOWN_FACES_DB  = Path("known_faces/faces.json")
RECOGNIZER_FILE = Path("known_faces/recognizer.pkl")


# ── PoseTracker ───────────────────────────────────────────────────────────────
class PoseTracker:
    """
    Runs MediaPipe pose in a background thread.
    Provides landmarks and a skeleton overlay image for ControlNet.
    """
    # Connections to draw as skeleton lines
    CONNECTIONS = [
        # Face
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        # Torso
        (11,12),(11,23),(12,24),(23,24),
        # Left arm
        (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
        # Right arm
        (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
        # Left leg
        (23,25),(25,27),(27,29),(27,31),(29,31),
        # Right leg
        (24,26),(26,28),(28,30),(28,32),(30,32),
    ]

    def __init__(self):
        self._landmarks = None
        self._skeleton  = None
        self._frame     = None
        self._lock      = threading.Lock()
        if MP_AVAILABLE:
            threading.Thread(target=self._run, daemon=True).start()
            print("[OK] Pose tracker started")

    def _run(self):
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        while True:
            with self._lock:
                frame = self._frame
            if frame is not None:
                try:
                    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    if results.pose_landmarks:
                        h, w = frame.shape[:2]
                        lms  = results.pose_landmarks.landmark
                        pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                        skel = self._draw_skeleton(pts, h, w)
                        with self._lock:
                            self._landmarks = pts
                            self._skeleton  = skel
                    else:
                        with self._lock:
                            self._landmarks = None
                            self._skeleton  = None
                except Exception:
                    pass
            time.sleep(1/15)  # 15fps pose tracking

    def update(self, frame):
        if MP_AVAILABLE:
            with self._lock:
                self._frame = cv2.resize(frame, (640, 480))

    def _draw_skeleton(self, pts, h, w):
        """Draw white skeleton on black background for ControlNet."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for a, b in self.CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (255, 255, 255), 3, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 5, (200, 200, 200), -1, cv2.LINE_AA)
        return img

    def get_skeleton(self, h, w):
        with self._lock:
            skel = self._skeleton
        if skel is None:
            return None
        return cv2.resize(skel, (w, h))

    def get_landmarks(self):
        with self._lock:
            return list(self._landmarks) if self._landmarks else None

    def get_body_mask(self, h, w):
        """Return a rough body mask from pose landmarks."""
        lms = self.get_landmarks()
        if not lms:
            return None
        # Use convex hull of all visible landmarks
        pts = np.array([[x, y] for x, y in lms], dtype=np.int32)
        # Scale from 640x480 to actual frame size
        pts[:, 0] = (pts[:, 0] * w / 640).astype(int)
        pts[:, 1] = (pts[:, 1] * h / 480).astype(int)
        pts = np.clip(pts, 0, [w-1, h-1])
        mask = np.zeros((h, w), dtype=np.float32)
        hull = cv2.convexHull(pts)
        # Expand hull to capture clothing/body width
        center = hull.mean(axis=0)
        expanded = ((hull - center) * 1.4 + center).astype(np.int32)
        cv2.fillConvexPoly(mask, expanded, 1.0)
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        return mask


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
            print(f"[WARN] Segmentation init: {e}")

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
            time.sleep(0.1)

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
    CONFIDENCE_THRESHOLD = 75

    def __init__(self):
        KNOWN_FACES_DIR.mkdir(exist_ok=True)
        self._lock       = threading.Lock()
        self._names      = {}
        self._samples    = {}
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
                print(f"[OK] Face recognizer — {len(self._names)} known face(s)")
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
        with self._lock:
            label = next((k for k,v in self._names.items() if v == name), None)
            if label is None:
                label = self._next_id()
                self._names[label] = name
                self._samples[label] = []
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            gray = cv2.equalizeHist(gray)
            if label not in self._samples:
                self._samples[label] = []
            self._samples[label].append(gray)
            if len(self._samples[label]) > 20:
                self._samples[label].pop(0)
            self._retrain()
            self._save()
            return label

    def _retrain(self):
        imgs, labels = [], []
        for label, samples in self._samples.items():
            for img in samples:
                imgs.append(img)
                labels.append(label)
        if imgs:
            self._recognizer.train(imgs, np.array(labels))
            self._trained = True

    def recognize(self, face_img):
        if not self._trained: return None, None
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
        self.poser      = PoseTracker()
        self.recognizer = FaceRecognizer()

        self.face_det = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self._detected_faces   = []
        self._recognized_names = {}
        self._selected_people  = set()
        self._frame_size       = (0, 0)
        self._pending_crops    = {}

        # Detection runs in its own thread at ~10fps — never blocks the stream
        self._det_frame  = None
        self._det_lock   = threading.Lock()
        self._recog_frame = None
        self._recog_lock  = threading.Lock()
        threading.Thread(target=self._detect_loop, daemon=True).start()
        threading.Thread(target=self._recog_loop,  daemon=True).start()

    def _detect_loop(self):
        """Haar Cascade + body box expansion at ~10fps in background."""
        while True:
            with self._det_lock:
                frame = self._det_frame
            if frame is not None:
                try:
                    h, w = frame.shape[:2]
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                        self._detected_faces  = body_boxes
                        self._selected_people = {
                            i for i in self._selected_people
                            if i < len(body_boxes)}
                        self._frame_size = (w, h)
                except Exception:
                    pass
            time.sleep(0.10)  # 10fps detection — plenty for a kiosk

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
        print(f"[OK] {len(self.catalog)} catalog face(s)")

    def reload_catalog(self):
        with self._lock:
            self.catalog.clear(); self._load_catalog()

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
                 "selected":i in sel, "name":names.get(i),
                 "frame_w":fw, "frame_h":fh}
                for i,(x,y,w,h) in enumerate(faces)]

    def toggle_person(self, click_x, click_y, frame_w, frame_h, frame=None):
        with self._lock:
            faces  = list(self._detected_faces)
            fw, fh = self._frame_size
        if fw == 0 or fh == 0: return
        sx = int(click_x * fw / frame_w)
        sy = int(click_y * fh / frame_h)
        pad = 35
        for i, (x, y, w, h) in enumerate(faces):
            if (x-pad) <= sx <= (x+w+pad) and (y-pad) <= sy <= (y+h+pad):
                with self._lock:
                    if i in self._selected_people:
                        self._selected_people.discard(i)
                        self._pending_crops.pop(i, None)
                    else:
                        self._selected_people.add(i)
                        # Store face crop immediately at selection time
                        if frame is not None:
                            # Search for a face within the body box
                            bx1 = max(0, x); by1 = max(0, y)
                            bx2 = min(frame.shape[1], x+w)
                            by2 = min(frame.shape[0], y+h)
                            body_roi = frame[by1:by2, bx1:bx2]
                            face_crop = None
                            if body_roi.size > 0:
                                gray_roi = cv2.cvtColor(body_roi, cv2.COLOR_BGR2GRAY)
                                sub_faces = self.face_det.detectMultiScale(
                                    gray_roi, 1.1, 4, minSize=(30, 30))
                                if len(sub_faces) > 0:
                                    fx, fy, fw2, fh2 = sub_faces[0]
                                    face_crop = body_roi[fy:fy+fh2, fx:fx+fw2]
                            # Fallback: use top 25% of body box as face estimate
                            if face_crop is None or face_crop.size == 0:
                                face_top = min(by1 + h//4, frame.shape[0])
                                face_crop = frame[by1:face_top, bx1:bx2]
                            if face_crop is not None and face_crop.size > 0:
                                self._pending_crops[i] = face_crop.copy()
                return

    def clear_selection(self):
        with self._lock:
            self._selected_people.clear()
            self._pending_crops.clear()

    def get_selected_mask(self, frame):
        """Use pose landmarks for accurate body mask, fall back to face box."""
        with self._lock:
            faces = list(self._detected_faces)
            sel   = set(self._selected_people)
        h, w = frame.shape[:2]
        if not sel:
            return np.ones((h, w), dtype=np.float32)

        # Try pose-based mask first
        pose_mask = self.poser.get_body_mask(h, w)
        if pose_mask is not None and sel:
            return pose_mask

        # Fallback: expanded face box to cover body
        mask = np.zeros((h, w), dtype=np.float32)
        for i in sel:
            if i < len(faces):
                x, y, fw, fh = faces[i]
                y1 = max(0, y - fh//2);    y2 = min(h, y + fh * 4)
                x1 = max(0, x - fw);       x2 = min(w, x + fw * 2)
                mask[y1:y2, x1:x2] = 1.0
        return mask

    def get_pose_skeleton(self, h, w):
        """Return skeleton image for ControlNet guidance."""
        return self.poser.get_skeleton(h, w)

    # ── Face naming — uses stored crop, no dependency on current detection ────
    def name_face(self, index, name):
        with self._lock:
            crop = self._pending_crops.get(index)
            all_keys = list(self._pending_crops.keys())
        print(f"[NAME] index={index} saved crops={all_keys} found={'yes' if crop is not None else 'NO'}")
        if crop is None:
            return False
        self.recognizer.learn(crop, name)
        print(f"[OK] Learned: {name}")
        return True

    def name_selected_face(self, index, name, frame):
        """Compatibility method — tries stored crop first, then frame."""
        ok = self.name_face(index, name)
        if ok: return True
        # Last resort: crop from current frame
        with self._lock:
            faces = list(self._detected_faces)
        if index < len(faces):
            x, y, w, h = faces[index]
            p = 10
            x1 = max(0,x-p); y1 = max(0,y-p)
            x2 = min(frame.shape[1],x+w+p); y2 = min(frame.shape[0],y+h+p)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                self.recognizer.learn(crop, name)
                print(f"[OK] Learned (fallback): {name}")
                return True
        return False

    def get_known_names(self):
        return self.recognizer.get_known_names()

    def forget_face(self, name):
        self.recognizer.forget(name)

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
                    p = 10
                    x1=max(0,x-p); y1=max(0,y-p)
                    x2=min(frame.shape[1],x+w+p); y2=min(frame.shape[0],y+h+p)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        name, _ = self.recognizer.recognize(crop)
                        if name: new_names[i] = name
                with self._lock:
                    self._recognized_names = new_names
            time.sleep(0.3)

    # ── process_frame ─────────────────────────────────────────────────────────
    def process_frame(self, frame):
        if frame is None: return frame
        h, w = frame.shape[:2]

        # Feed background workers (non-blocking — they copy internally)
        self.seg.update(frame)
        self.poser.update(frame)

        # Feed detection + recognition threads
        with self._det_lock:
            self._det_frame = frame  # no copy — detection thread reads at its own pace
        with self._recog_lock:
            self._recog_frame = frame

        # Snapshot cached state — no detection here
        with self._lock:
            sel        = set(self._selected_people)
            names      = dict(self._recognized_names)
            faces_snap = list(self._detected_faces)

        result = frame.copy()
        self._draw_overlay(result, faces_snap, sel, names)
        return result

    def _draw_overlay(self, frame, faces, selected, names):
        for i, (x, y, w, h) in enumerate(faces):
            is_sel = i in selected
            name   = names.get(i)
            pad    = 10
            x1,y1  = max(0,x-pad), max(0,y-pad)
            x2,y2  = min(frame.shape[1],x+w+pad), min(frame.shape[0],y+h+pad)

            if is_sel:
                color, thick = (0,220,255), 3
                cv2.rectangle(frame,(x1-2,y1-2),(x2+2,y2+2),(0,100,120),6)
            else:
                color, thick = (70,70,70), 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,thick)

            if name:
                label, label_col, bg_col = name, (0,210,255), (0,80,100)
            elif is_sel:
                label, label_col, bg_col = f"Person {i+1} ✓", (0,220,255), (0,60,80)
            else:
                label, label_col, bg_col = f"Person {i+1}", (70,70,70), None

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            lx = x1
            ly = max(th+6, y1-6)
            if bg_col:
                cv2.rectangle(frame,(lx-2,ly-th-4),(lx+tw+6,ly+4),bg_col,-1)
            cv2.putText(frame, label, (lx+2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_col, 2, cv2.LINE_AA)

        if faces and not selected:
            cv2.putText(frame, "Tap a person to select",
                        (10, frame.shape[0]-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,200), 2, cv2.LINE_AA)
