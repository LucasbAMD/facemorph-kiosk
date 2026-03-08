"""
face_processor.py — AMD Adapt Kiosk
Full-body character transformation pipeline:

  1. rembg cleanly cuts person from background (threaded, ~15fps)
  2. Entire body gets character color transform (skin, clothes, everything)
  3. Pose-estimated markings placed on face + body (stripes, dots, veins, etc.)
  4. Character world replaces background
  5. Rim glow around silhouette

No face swap model needed. Person becomes the character — full body.
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path

# ── onnxruntime compatibility patch ───────────────────────────────────────────
# onnxruntime-directml 1.24+ removed set_default_logger_severity but rembg calls it
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
    print("[WARN] rembg not installed — run: venv\\Scripts\\python.exe -m pip install rembg")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MP_POSE_AVAILABLE = False
MP_FACE_AVAILABLE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        if hasattr(mp.solutions, 'pose'):
            MP_POSE_AVAILABLE = True
        if hasattr(mp.solutions, 'face_mesh'):
            MP_FACE_AVAILABLE = True
    print(f"[OK] MediaPipe pose={MP_POSE_AVAILABLE} face={MP_FACE_AVAILABLE}")
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# Character config
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

# ══════════════════════════════════════════════════════════════════════════════
# Background generators
# ══════════════════════════════════════════════════════════════════════════════
def _pandora(h, w):
    bg = np.full((h,w,3),(8,15,6),dtype=np.uint8)
    rng = np.random.default_rng(42)
    # Bioluminescent dots
    for _ in range(500):
        x,y = int(rng.integers(0,w)),int(rng.integers(0,h))
        r   = int(rng.integers(2,9))
        hv  = int(rng.integers(85,115))
        bv  = int(rng.integers(100,255))
        col = cv2.cvtColor(np.array([[[hv,220,bv]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0].tolist()
        cv2.circle(bg,(x,y),r,col,-1)
        cv2.circle(bg,(x,y),r*5,[c//6 for c in col],-1)
    # Tree silhouettes
    for i in range(8):
        x  = int(rng.integers(0,w))
        ty = int(rng.integers(int(h*0.1),int(h*0.5)))
        cv2.line(bg,(x,h),(x,ty),(15,35,10),int(rng.integers(8,20)))
        cv2.circle(bg,(x,ty),int(rng.integers(30,80)),(10,60,15),-1)
    # Light shafts from top
    shaft = np.zeros_like(bg)
    for x in range(0,w,w//5):
        cv2.line(shaft,(x+rng.integers(-20,20),0),(x,h),(15,30,8),25)
    return cv2.add(cv2.GaussianBlur(bg,(41,41),0), cv2.GaussianBlur(shaft,(31,31),0))

def _gamma(h, w):
    bg = np.zeros((h,w,3),dtype=np.uint8)
    bg[:,:] = (3,8,3)
    rng = np.random.default_rng(11)
    # Rubble
    for _ in range(20):
        pts = np.array([
            [rng.integers(0,w), h],
            [rng.integers(0,w), h-rng.integers(30,180)],
            [rng.integers(0,w), h],
        ],np.int32)
        cv2.fillPoly(bg,[pts],(5,14,5))
    # Green energy
    for _ in range(40):
        x1,y1 = rng.integers(0,w),rng.integers(0,h)
        x2,y2 = x1+rng.integers(-80,80),y1+rng.integers(-80,80)
        cv2.line(bg,(int(x1),int(y1)),(int(x2),int(y2)),(0,int(rng.integers(30,100)),0),1)
    return cv2.GaussianBlur(bg,(21,21),0)

def _space(h, w):
    bg = np.zeros((h,w,3),dtype=np.uint8)
    bg[:] = (10,3,10)
    rng = np.random.default_rng(7)
    for _ in range(700):
        x,y = int(rng.integers(0,w)),int(rng.integers(0,h))
        b   = int(rng.integers(60,255))
        bg[y,x] = (b,b,b)
    # Purple nebula
    nb = np.zeros_like(bg)
    cv2.ellipse(nb,(w//3,h//3),(w//3,h//4),20,0,360,(40,5,60),-1)
    return cv2.add(bg, cv2.GaussianBlur(nb,(121,121),0))

def _thermal_bg(h, w):
    bg = np.zeros((h,w,3),dtype=np.uint8)
    bg[:] = (0,0,8)
    return bg

def _spirit(h, w):
    bg = np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        t = y/h
        bg[y,:] = (int(5+t*15),int(3+t*8),int(15+t*30))
    rng = np.random.default_rng(5)
    for _ in range(300):
        x,y = int(rng.integers(0,w)),int(rng.integers(0,h))
        b   = int(rng.integers(40,180))
        bg[y,x] = (b,b,min(255,b+60))
    return cv2.GaussianBlur(bg,(31,31),0)

def _forest(h, w):
    bg = np.zeros((h,w,3),dtype=np.uint8)
    bg[:int(h*0.45),:] = (15,45,8)
    bg[int(h*0.45):,:] = (5,22,3)
    rng = np.random.default_rng(3)
    for _ in range(30):
        x  = int(rng.integers(0,w))
        ty = int(rng.uniform(h*0.1,h*0.65))
        r  = int(rng.integers(25,65))
        cv2.line(bg,(x,h),(x,ty),(18,45,10),max(4,r//5))
        for di in range(3):
            cv2.circle(bg,(x+rng.integers(-r//2,r//2),ty-di*r//3),
                       max(5,r-di*12),(8+di*5,65+di*20,8),-1)
    return cv2.GaussianBlur(bg,(35,35),0)

BG_FNS = {
    "pandora":_pandora,"gamma":_gamma,"space":_space,
    "thermal":_thermal_bg,"spirit":_spirit,"forest":_forest,
    "mc": None,  # built separately
}

# ══════════════════════════════════════════════════════════════════════════════
# Threaded rembg segmenter
# ══════════════════════════════════════════════════════════════════════════════
class Segmenter:
    def __init__(self):
        self._mask=None; self._frame=None
        self._lock=threading.Lock(); self._running=True
        self.ready=False
        if REMBG_AVAILABLE:
            threading.Thread(target=self._run,daemon=True).start()
        else:
            print("[WARN] rembg unavailable — body effects disabled")

    def _run(self):
        try:
            print("[..] Loading body segmentation model (~170 MB first run)...")
            sess = new_session("u2net_human_seg")
            self.ready = True
            print("[OK] Body segmentation ready!")
            while self._running:
                with self._lock: frame=self._frame
                if frame is not None:
                    try:
                        m = remove(frame,session=sess,only_mask=True,post_process_mask=True)
                        m = cv2.GaussianBlur(m,(21,21),0)
                        with self._lock: self._mask=m
                    except: pass
                time.sleep(0.06)
        except Exception as e:
            print(f"[ERROR] Segmenter: {e}")

    def update(self,frame):
        with self._lock: self._frame=frame

    def get(self,h,w):
        with self._lock: m=self._mask
        if m is None: return None
        if m.shape[:2]!=(h,w): m=cv2.resize(m,(w,h))
        return m.astype(np.float32)/255.0

    def stop(self): self._running=False

# ══════════════════════════════════════════════════════════════════════════════
# Pose tracker (MediaPipe)
# ══════════════════════════════════════════════════════════════════════════════
class PoseTracker:
    def __init__(self):
        self.pose = None
        self.face = None
        if MP_POSE_AVAILABLE:
            try:
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[OK] Pose tracking loaded")
            except Exception as e:
                print(f"[WARN] Pose: {e}")
        if MP_FACE_AVAILABLE:
            try:
                self.face = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=2,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[OK] Face mesh loaded")
            except Exception as e:
                print(f"[WARN] FaceMesh: {e}")

    def get_pose(self, frame):
        """Returns pose landmarks as dict of {name: (px, py)} or None."""
        if self.pose is None: return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks: return None
        h, w = frame.shape[:2]
        lm = res.pose_landmarks.landmark
        PL = mp.solutions.pose.PoseLandmark
        def pt(idx):
            l = lm[idx]
            return (int(l.x*w), int(l.y*h))
        return {
            "nose":          pt(PL.NOSE),
            "l_eye":         pt(PL.LEFT_EYE),
            "r_eye":         pt(PL.RIGHT_EYE),
            "l_ear":         pt(PL.LEFT_EAR),
            "r_ear":         pt(PL.RIGHT_EAR),
            "l_shoulder":    pt(PL.LEFT_SHOULDER),
            "r_shoulder":    pt(PL.RIGHT_SHOULDER),
            "l_elbow":       pt(PL.LEFT_ELBOW),
            "r_elbow":       pt(PL.RIGHT_ELBOW),
            "l_wrist":       pt(PL.LEFT_WRIST),
            "r_wrist":       pt(PL.RIGHT_WRIST),
            "l_hip":         pt(PL.LEFT_HIP),
            "r_hip":         pt(PL.RIGHT_HIP),
            "l_knee":        pt(PL.LEFT_KNEE),
            "r_knee":        pt(PL.RIGHT_KNEE),
        }

    def get_face_landmarks(self, frame):
        """Returns face mesh landmarks as list of (px,py) or None."""
        if self.face is None: return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face.process(rgb)
        if not res.multi_face_landmarks: return None
        h, w = frame.shape[:2]
        out = []
        for fl in res.multi_face_landmarks:
            pts = [(int(l.x*w), int(l.y*h)) for l in fl.landmark]
            out.append(pts)
        return out

# ══════════════════════════════════════════════════════════════════════════════
# Main FaceProcessor
# ══════════════════════════════════════════════════════════════════════════════
class FaceProcessor:
    def __init__(self, faces_dir="faces"):
        self.faces_dir = Path(faces_dir)
        self.mode      = "passthrough"
        self.character = "navi"
        self.face_key  = None
        self._lock     = threading.Lock()
        self._bg_cache = {}

        # Face library (for faceswap panel)
        self.catalog = {}
        self._load_catalog()

        self.seg   = Segmenter()
        self.poser = PoseTracker()

        # OpenCV face detector (fallback when pose not available)
        self.face_det = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _load_catalog(self):
        if not self.faces_dir.exists(): self.faces_dir.mkdir(parents=True); return
        ext={".jpg",".jpeg",".png",".webp"}
        for cat in self.faces_dir.iterdir():
            if not cat.is_dir(): continue
            for p in cat.iterdir():
                if p.suffix.lower() not in ext: continue
                img = cv2.imread(str(p))
                if img is None: continue
                key = f"{cat.name}/{p.stem}"
                self.catalog[key]={"label":p.stem.replace("_"," ").title(),
                                   "category":cat.name.title(),"img":img,"path":str(p)}
        print(f"[OK] {len(self.catalog)} face(s) in catalog")

    def reload_catalog(self):
        with self._lock: self.catalog.clear(); self._load_catalog()

    def get_catalog(self):
        return [{"key":k,"label":v["label"],"category":v["category"]} for k,v in self.catalog.items()]

    def get_characters(self):
        out=[]
        for k in CHARACTER_ORDER:
            c=CHAR_CONFIG[k]
            out.append({"key":k,"label":c["label"],"subtitle":c["subtitle"],
                        "emoji":c["emoji"],"color":c["color"],"swap_ready":True,"ref_photo":""})
        return out

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
            mode=self.mode; char=self.character; fkey=self.face_key
        try:
            if   mode=="character":           return self._character(frame,char)
            elif mode=="minecraft":           return self._minecraft(frame)
            elif mode=="cyberpunk":           return self._cyberpunk(frame)
            elif mode=="faceswap" and fkey:   return self._color_faceswap(frame,fkey)
            return frame
        except Exception as e:
            import traceback; traceback.print_exc()
            return frame

    # ══════════════════════════════════════════════════════════════════════════
    # CHARACTER — full body transform
    # ══════════════════════════════════════════════════════════════════════════
    def _character(self, frame, char):
        h,w = frame.shape[:2]
        mask = self.seg.get(h,w)
        pose = self.poser.get_pose(frame)

        # ── Step 1: Transform body color ──────────────────────────────────────
        if char == "predator":
            body = self._thermal(frame)
        elif char == "ghost":
            pale = self._hue_shift(frame, 105, -100, 1.3)
            body = cv2.addWeighted(pale, 0.45, frame, 0.55, 0)
        else:
            cfg = {
                "navi":   (103, 130, 0.80),
                "hulk":   (62,  150, 0.92),
                "thanos": (148, 100, 0.68),
                "groot":  (18,   70, 0.72),
            }
            h_,s_,v_ = cfg.get(char,(103,130,0.80))
            body = self._hue_shift(frame, h_, s_, v_)

        # ── Step 2: Character-specific markings ────────────────────────────────
        if char == "navi":     body = self._navi_markings(body, frame, pose)
        elif char == "hulk":   body = self._hulk_markings(body, frame, pose)
        elif char == "thanos": body = self._thanos_markings(body, pose)
        elif char == "ghost":  body = self._ghost_markings(body, mask)
        elif char == "groot":  body = self._groot_markings(body, mask)
        elif char == "predator": body = self._predator_markings(body, pose)

        # ── Step 3: Composite over character background ────────────────────────
        result = body
        if mask is not None:
            bg = self._get_bg(CHAR_CONFIG[char]["bg"], h, w)
            m3 = np.stack([mask]*3, axis=2)
            result = np.clip(body.astype(np.float32)*m3 +
                             bg.astype(np.float32)*(1-m3), 0,255).astype(np.uint8)
            # Rim glow
            glow_colors = {
                "navi":(255,210,40),"hulk":(30,230,50),"thanos":(200,50,255),
                "predator":(0,80,255),"ghost":(220,240,255),"groot":(40,180,40)
            }
            result = self._rim_glow(result, mask, glow_colors.get(char,(255,255,255)))
        else:
            # Still loading — show message
            cv2.putText(result,"Segmentation loading...",(10,35),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,220,200),2,cv2.LINE_AA)

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # MARKINGS — applied on top of color-shifted body
    # ══════════════════════════════════════════════════════════════════════════

    def _navi_markings(self, body, original, pose):
        """Bioluminescent stripes, dots, and eye glow — tracks with pose."""
        out = body.copy()
        h,w = out.shape[:2]

        # Get face bounding box from pose or detector
        face_box = self._get_face_box(original, pose)

        if face_box:
            fx,fy,fw,fh = face_box
            cx,cy = fx+fw//2, fy+fh//2
            CYAN  = (255, 230, 60)   # bioluminescent cyan (BGR)
            LCYAN = (200, 255, 120)

            # ── Cheek stripes (3 per side) ─────────────────────────────────
            for side in [-1, 1]:
                for i in range(3):
                    y_off = int(fh*(0.05 + i*0.10))
                    x1 = cx + side*int(fw*0.12)
                    x2 = cx + side*int(fw*0.44)
                    y1 = cy + y_off
                    y2 = cy + y_off + int(fh*0.04)
                    # Glow base (wide, dim)
                    cv2.line(out,(x1,y1),(x2,y2),(80,180,20),5,cv2.LINE_AA)
                    # Bright core
                    cv2.line(out,(x1,y1),(x2,y2),CYAN,2,cv2.LINE_AA)

            # ── Forehead dots ──────────────────────────────────────────────
            dot_positions = [
                (0, -0.38), (-0.12,-0.28),(0.12,-0.28),
                (-0.22,-0.16),(0.22,-0.16),
                (-0.08,-0.44),(0.08,-0.44),
            ]
            for (dx,dy) in dot_positions:
                px = cx+int(dx*fw); py = cy+int(dy*fh)
                cv2.circle(out,(px,py),5,(80,180,20),-1,cv2.LINE_AA)
                cv2.circle(out,(px,py),3,CYAN,-1,cv2.LINE_AA)
                cv2.circle(out,(px,py),6,(40,100,10),1,cv2.LINE_AA)

            # ── Nose bridge line ───────────────────────────────────────────
            nb_x = cx; nb_y1 = cy-int(fh*0.12); nb_y2 = cy+int(fh*0.06)
            cv2.line(out,(nb_x,nb_y1),(nb_x,nb_y2),(80,160,20),3,cv2.LINE_AA)
            for i in range(4):
                py = nb_y1+i*((nb_y2-nb_y1)//4)
                cv2.circle(out,(nb_x,py),2,CYAN,-1)

            # ── Eye glow rings ─────────────────────────────────────────────
            for ex in [cx-int(fw*0.20), cx+int(fw*0.20)]:
                ey = cy-int(fh*0.08)
                cv2.ellipse(out,(ex,ey),(int(fw*0.13),int(fh*0.09)),
                            0,0,360,(120,230,30),2,cv2.LINE_AA)
                cv2.ellipse(out,(ex,ey),(int(fw*0.09),int(fh*0.06)),
                            0,0,360,CYAN,1,cv2.LINE_AA)

        # ── Arm / body markings from pose landmarks ────────────────────────
        if pose:
            GLOW = (255, 220, 50)
            for seg_pts in [
                ("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
                ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
            ]:
                if all(k in pose for k in seg_pts):
                    p1,p2 = pose[seg_pts[0]], pose[seg_pts[1]]
                    # Glow base
                    cv2.line(out,p1,p2,(40,120,10),6,cv2.LINE_AA)
                    # Bright stripe
                    cv2.line(out,p1,p2,GLOW,2,cv2.LINE_AA)

            # Shoulder dots
            for key in ["l_shoulder","r_shoulder","l_elbow","r_elbow"]:
                if key in pose:
                    cv2.circle(out,pose[key],8,(40,140,10),-1)
                    cv2.circle(out,pose[key],5,GLOW,-1)

        return out

    def _hulk_markings(self, body, original, pose):
        """Vein lines and muscle definition."""
        out = body.copy()
        # Edge detection overlay for muscle texture
        gray  = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 25, 70)
        # Green tint edges
        vein_layer = np.zeros_like(out)
        vein_layer[:,:,1] = edges  # green channel only
        # Subtle overlay
        out = cv2.add(out, (vein_layer // 4).astype(np.uint8))

        if pose:
            VEIN = (30, 200, 30)
            # Vein lines on forearms
            for seg in [("l_elbow","l_wrist"),("r_elbow","r_wrist")]:
                if all(k in pose for k in seg):
                    p1,p2 = pose[seg[0]],pose[seg[1]]
                    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                    cv2.line(out,p1,mid,VEIN,3,cv2.LINE_AA)
                    cv2.line(out,mid,p2,VEIN,2,cv2.LINE_AA)

            # Brow furrow — use face box
            face_box = self._get_face_box(original, pose)
            if face_box:
                fx,fy,fw,fh = face_box
                cx,cy = fx+fw//2, fy+fh//2
                brow_y = cy - int(fh*0.28)
                cv2.line(out,(cx-int(fw*0.18),brow_y),(cx,brow_y+int(fh*0.06)),(20,160,20),3,cv2.LINE_AA)
                cv2.line(out,(cx+int(fw*0.18),brow_y),(cx,brow_y+int(fh*0.06)),(20,160,20),3,cv2.LINE_AA)
        return out

    def _thanos_markings(self, body, pose):
        """Chin creases and joint stone glows."""
        out = body.copy()
        if not pose: return out

        # Infinity stone glow dots at joints
        stones = [
            ("l_wrist",(0,0,255)),    # blue
            ("r_wrist",(0,160,255)),  # orange
            ("l_elbow",(0,255,255)),  # yellow/green
            ("r_elbow",(255,0,100)),  # purple
            ("l_shoulder",(255,60,0)),# red
            ("r_shoulder",(255,200,0)),# orange
        ]
        for (key,color) in stones:
            if key in pose:
                p = pose[key]
                cv2.circle(out,p,18,color,-1,cv2.LINE_AA)
                cv2.circle(out,p,22,[c//3 for c in color],8,cv2.LINE_AA)
                cv2.circle(out,p,12,(255,255,255),1,cv2.LINE_AA)
        return out

    def _ghost_markings(self, body, mask):
        """White aura halo around body silhouette."""
        if mask is None: return body
        out = body.copy()
        m8  = (mask*255).astype(np.uint8)
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
        aura_m = cv2.dilate(m8,k,iterations=4)
        aura_b = cv2.GaussianBlur(aura_m,(61,61),0).astype(np.float32)/255.0
        m3  = np.stack([aura_b]*3,axis=2)
        white = np.full_like(out,220)
        out = np.clip(out.astype(np.float32)+white*m3*0.5, 0,255).astype(np.uint8)
        return out

    def _groot_markings(self, body, mask):
        """Bark texture via noise."""
        out = body.copy()
        noise = np.random.randint(0,35,out.shape,dtype=np.uint8)
        out = cv2.subtract(out, noise)
        # Vine lines across body using contours of mask
        if mask is not None:
            m8 = (mask*200).astype(np.uint8)
            contours,_ = cv2.findContours(m8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out,contours,-1,(20,100,20),2)
        return out

    def _predator_markings(self, body, pose):
        """Scan-line grid + targeting reticle on face."""
        out = body.copy()
        h,w = out.shape[:2]
        # Scan lines
        for y in range(0,h,5):
            out[y,:] = (out[y,:].astype(np.float32)*0.7).astype(np.uint8)
        # Targeting reticle if pose available
        if pose and "nose" in pose:
            nx,ny = pose["nose"]
            r = 40
            cv2.circle(out,(nx,ny),r,(0,255,0),1,cv2.LINE_AA)
            cv2.circle(out,(nx,ny),r+8,(0,200,0),1,cv2.LINE_AA)
            cv2.line(out,(nx-r-20,ny),(nx-r+10,ny),(0,255,0),1)
            cv2.line(out,(nx+r-10,ny),(nx+r+20,ny),(0,255,0),1)
            cv2.line(out,(nx,ny-r-20),(nx,ny-r+10),(0,255,0),1)
            cv2.line(out,(nx,ny+r-10),(nx,ny+r+20),(0,255,0),1)
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # THERMAL helper (Predator body)
    # ══════════════════════════════════════════════════════════════════════════
    def _thermal(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(cv2.bitwise_not(gray), cv2.COLORMAP_JET)

    # ══════════════════════════════════════════════════════════════════════════
    # MINECRAFT
    # ══════════════════════════════════════════════════════════════════════════
    def _minecraft(self, frame):
        h,w   = frame.shape[:2]
        bs    = 22

        # Pixelate whole frame
        sm  = cv2.resize(frame,(w//bs,h//bs),interpolation=cv2.INTER_NEAREST)
        pix = cv2.resize(sm,(w,h),interpolation=cv2.INTER_NEAREST)

        # Grid
        for x in range(0,w,bs): cv2.line(pix,(x,0),(x,h),(0,0,0),1)
        for y in range(0,h,bs): cv2.line(pix,(0,y),(w,y),(0,0,0),1)

        # Steve face on detected face
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = self.face_det.detectMultiScale(gray,1.1,5,minSize=(50,50))
        for (fx,fy,fw,fh) in faces:
            pad=int(fw*0.15)
            x1,y1=max(0,fx-pad),max(0,fy-pad)
            x2,y2=min(w,fx+fw+pad),min(h,fy+fh+pad)
            sw,sh=x2-x1,y2-y1
            steve=self._steve_texture(sw,sh)
            pix[y1:y2,x1:x2]=steve

        # MC background
        mask = self.seg.get(h,w)
        if mask is not None:
            mk=(mask*255).astype(np.uint8)
            ms=cv2.resize(mk,(w//bs,h//bs),interpolation=cv2.INTER_NEAREST)
            mb=cv2.resize(ms,(w,h),interpolation=cv2.INTER_NEAREST).astype(np.float32)/255.0
            bg=self._get_bg("mc",h,w)
            m3=np.stack([mb]*3,axis=2)
            pix=np.clip(pix.astype(np.float32)*m3+bg.astype(np.float32)*(1-m3),0,255).astype(np.uint8)
        return pix

    def _steve_texture(self, tw, th):
        STEVE=[
            "HHHHHHHH",
            "HHHHHHHH",
            "SSESSSESS"[:8],
            "SSPSSSP"[:8]+"S",
            "SSSSSSSS",
            "SDDDDSS"[:8],
            "SDMMMDSS"[:8],
            "SSSSSSSS",
        ]
        C={"H":(65,90,115),"S":(115,155,195),"E":(230,230,230),
           "P":(40,40,180),"M":(30,40,80),"D":(50,55,80)}
        tex=np.zeros((th,tw,3),dtype=np.uint8)
        ch_h=max(1,th//8); ch_w=max(1,tw//8)
        for ri,row in enumerate(STEVE):
            for ci,ch in enumerate(row[:8]):
                col=C.get(ch,C["S"])
                tex[ri*ch_h:(ri+1)*ch_h, ci*ch_w:(ci+1)*ch_w]=col
        return tex

    # ══════════════════════════════════════════════════════════════════════════
    # CYBERPUNK
    # ══════════════════════════════════════════════════════════════════════════
    def _cyberpunk(self, frame):
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,60,180)
        neon  = np.zeros_like(frame)
        neon[:,:,0]=edges; neon[:,:,2]=edges
        dark  = cv2.convertScaleAbs(frame,alpha=0.3,beta=0)
        out   = cv2.add(dark,neon)
        for y in range(0,frame.shape[0],4):
            out[y,:] = (out[y,:].astype(np.float32)*0.65).astype(np.uint8)
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Color face swap (for face library panel)
    # ══════════════════════════════════════════════════════════════════════════
    def _color_faceswap(self, frame, fkey):
        entry = self.catalog.get(fkey)
        if entry is None: return frame
        # Show reference face in corner
        ref = entry["img"]
        h,w = frame.shape[:2]
        th  = h//5; tw = int(ref.shape[1]*(th/ref.shape[0]))
        thumb = cv2.resize(ref,(tw,th))
        out   = frame.copy()
        out[10:10+th, 10:10+tw] = thumb
        cv2.rectangle(out,(10,10),(10+tw,10+th),(0,220,180),2)
        cv2.putText(out,entry["label"],(10,10+th+20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,180),2)
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _hue_shift(self, img, hue, sat_boost, val_scale):
        """Force entire image to target hue, boost sat, scale val."""
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = hue
        if sat_boost > 0:
            hsv[:,:,1] = np.clip(hsv[:,:,1]+(255-hsv[:,:,1])*0.75+sat_boost, 0,255)
        else:
            hsv[:,:,1] = np.clip(hsv[:,:,1]+sat_boost, 0,255)
        hsv[:,:,2] = np.clip(hsv[:,:,2]*val_scale, 10,255)
        return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

    def _rim_glow(self, frame, mask_f, color_bgr, width=4):
        m8  = (mask_f*255).astype(np.uint8)
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        dil = cv2.dilate(m8,k,iterations=width)
        edge= cv2.subtract(dil,m8)
        ef  = cv2.GaussianBlur(edge,(15,15),0).astype(np.float32)/255.0
        rim = np.zeros_like(frame,dtype=np.float32)
        for c,v in enumerate(color_bgr): rim[:,:,c]=ef*v
        return np.clip(frame.astype(np.float32)+rim,0,255).astype(np.uint8)

    def _get_face_box(self, frame, pose=None):
        """Get face bounding box — from pose landmarks or OpenCV detector."""
        h,w = frame.shape[:2]
        if pose and all(k in pose for k in ["l_ear","r_ear","nose"]):
            le,re,no = pose["l_ear"],pose["r_ear"],pose["nose"]
            fw = abs(re[0]-le[0])
            fh = int(fw*1.3)
            cx = (le[0]+re[0])//2
            cy = no[1]
            return (cx-fw//2, cy-fh//2, fw, fh)
        # Fallback: OpenCV
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = self.face_det.detectMultiScale(gray,1.1,5,minSize=(40,40))
        if len(faces)>0:
            return tuple(faces[0])
        return None

    def _get_bg(self, name, h, w):
        key=f"{name}_{h}_{w}"
        if key not in self._bg_cache:
            if name=="mc":
                self._bg_cache[key]=self._mc_bg(h,w)
            elif name in BG_FNS and BG_FNS[name]:
                self._bg_cache[key]=BG_FNS[name](h,w)
            else:
                self._bg_cache[key]=np.zeros((h,w,3),dtype=np.uint8)
        return self._bg_cache[key]

    def _mc_bg(self, h, w):
        bg=np.zeros((h,w,3),dtype=np.uint8)
        sh=int(h*0.55)
        for y in range(sh):
            t=y/sh
            bg[y,:]=(int(200+t*35),int(210+t*25),int(240+t*15))
        for cx in [w//5,w//2,4*w//5]:
            for dx,dy in [(-40,0),(0,-15),(40,0),(0,0),(-20,-12),(20,-12)]:
                cv2.rectangle(bg,(cx+dx-25,sh//3+dy),(cx+dx+25,sh//3+dy+20),(240,240,240),-1)
        bg[sh:sh+int(h*0.06),:] = (34,130,34)
        bg[sh+int(h*0.06):,:] = (67,96,134)
        bs=20
        s=cv2.resize(bg,(w//bs,h//bs),interpolation=cv2.INTER_NEAREST)
        bg=cv2.resize(s,(w,h),interpolation=cv2.INTER_NEAREST)
        for x in range(0,w,bs): cv2.line(bg,(x,0),(x,h),(0,0,0),1)
        for y in range(0,h,bs): cv2.line(bg,(0,y),(w,y),(0,0,0),1)
        return bg
