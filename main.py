"""
main.py — AMD Adapt Kiosk Backend
Clean live feed + person selection + AI generation on selected people only.
"""

import cv2
import asyncio
import threading
import time
import shutil
import base64
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from face_processor import FaceProcessor
from comfy_bridge import ComfyBridge

app = FastAPI(title="AMD Adapt Kiosk")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

processor = FaceProcessor(faces_dir="faces")
comfy     = ComfyBridge()

cap: Optional[cv2.VideoCapture] = None
cap_lock        = threading.Lock()
latest_frame    = None
frame_lock      = threading.Lock()
capture_running = False
ai_result_frame = None
ai_result_lock  = threading.Lock()

# ── Camera ────────────────────────────────────────────────────────────────────
def start_capture(camera_index: int = 0):
    global cap, latest_frame, capture_running
    capture_running = True
    opened_index = camera_index
    with cap_lock:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            for alt in [2, 1, 0, 3]:
                if alt == camera_index: continue
                cap = cv2.VideoCapture(alt)
                if cap.isOpened():
                    opened_index = alt
                    print(f"[OK] Camera opened at index {alt}")
                    break
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 60)

    def loop():
        global latest_frame
        while capture_running:
            with cap_lock:
                ok = cap and cap.isOpened()
            if not ok:
                time.sleep(0.1); continue
            with cap_lock:
                ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame
            else:
                time.sleep(0.033)

    threading.Thread(target=loop, daemon=True).start()
    print(f"[OK] Camera started (index={opened_index})")

def _placeholder(w=1280, h=720):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for camera...", (w//2-220, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80,80,80), 2)
    return img

# ── MJPEG stream ──────────────────────────────────────────────────────────────
def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            processed = _placeholder()
        else:
            processed = processor.process_frame(frame)
        _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(1/60)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path("index.html")
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists()
                        else "<h1>index.html not found</h1>")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/set_mode")
async def set_mode(mode: str = Form(...), face_key: str = Form(None),
                   character: str = Form(None)):
    valid = {"passthrough","character","minecraft","cyberpunk","faceswap"}
    if mode not in valid:
        raise HTTPException(400, f"Invalid mode: {valid}")
    processor.set_mode(mode, face_key, character)
    return {"status":"ok","mode":mode}

@app.get("/characters")
async def get_characters():
    chars = processor.get_characters()
    for c in chars:
        c["ai_ready"] = comfy.available
    return JSONResponse({"characters": chars})

@app.get("/catalog")
async def get_catalog():
    return JSONResponse({"faces": processor.get_catalog()})

# ── Person selection ──────────────────────────────────────────────────────────
@app.get("/faces")
async def get_faces():
    """Return current detected face positions + selection state."""
    return JSONResponse({"faces": processor.get_detected_faces()})

@app.post("/select_person")
async def select_person(click_x: float = Form(...), click_y: float = Form(...),
                        frame_w: float = Form(...), frame_h: float = Form(...)):
    """Toggle selection of person at click coordinates."""
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    processor.toggle_person(click_x, click_y, frame_w, frame_h, frame)
    return JSONResponse({"faces": processor.get_detected_faces()})

@app.post("/clear_selection")
async def clear_selection():
    processor.clear_selection()
    return JSONResponse({"status": "ok"})

# ── AI Generation ─────────────────────────────────────────────────────────────
@app.post("/generate")
async def generate(character: str = Form(...)):
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        raise HTTPException(400, "No camera frame available")
    if not comfy.check_available():
        raise HTTPException(503, "ComfyUI not running — start ~/start_comfyui.sh")

    # Get selection info
    detected = processor.get_detected_faces()
    sel_count = sum(1 for f in detected if f["selected"])

    # If people are detected but none selected, block generation
    if len(detected) > 0 and sel_count == 0:
        raise HTTPException(400, "Please select at least one person before transforming")

    selected_mask = processor.get_selected_mask(frame)
    started = comfy.generate(frame, character, selected_mask)
    if not started:
        return JSONResponse({"status": "busy", "message": "Already generating"})
    return JSONResponse({"status": "generating", "message": "Transforming..."})

@app.get("/generate_status")
async def generate_status():
    status = comfy.get_status()
    if status["status"] == "done":
        result = comfy.get_result()
        if result is not None:
            with ai_result_lock:
                global ai_result_frame
                ai_result_frame = result
            _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 92])
            b64 = base64.b64encode(buf.tobytes()).decode()
            return JSONResponse({"status":"done",
                                 "image": f"data:image/jpeg;base64,{b64}"})
    return JSONResponse(status)

@app.get("/comfy_status")
async def comfy_status():
    return JSONResponse({"available": comfy.check_available()})

@app.post("/upload_face")
async def upload_face(file: UploadFile = File(...),
                      label: str = Form(...),
                      category: str = Form("custom")):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg",".jpeg",".png",".webp"}:
        raise HTTPException(400, "Only JPG/PNG/WEBP")
    dest = Path("faces") / category
    dest.mkdir(parents=True, exist_ok=True)
    safe = label.replace(" ","_").lower()
    path = dest / f"{safe}{ext}"
    with open(path,"wb") as f: shutil.copyfileobj(file.file, f)
    processor.reload_catalog()
    return {"status":"ok","key":f"{category}/{safe}"}

@app.get("/face_thumb")
async def face_thumb(key: str):
    entry = processor.catalog.get(key)
    if not entry: raise HTTPException(404, "Not found")
    return FileResponse(entry["path"])

@app.post("/reload_catalog")
async def reload_catalog():
    processor.reload_catalog()
    return {"status":"ok","count":len(processor.catalog)}

@app.get("/status")
async def status():
    with frame_lock:
        has_frame = latest_frame is not None
    return {"camera": has_frame, "mode": processor.mode,
            "face_key": processor.face_key,
            "face_count": len(processor.catalog),
            "ai_ready": comfy.available}


@app.post("/name_face")
async def name_face(index: int = Form(...), name: str = Form(...)):
    """Name a detected face at given index using current camera frame."""
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        raise HTTPException(400, "No camera frame")
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")
    success = processor.name_selected_face(index, name, frame)
    return JSONResponse({"status": "ok" if success else "error",
                         "name": name, "index": index})

@app.get("/known_names")
async def known_names():
    return JSONResponse({"names": processor.get_known_names()})

@app.post("/forget_face")
async def forget_face(name: str = Form(...)):
    processor.forget_face(name)
    return JSONResponse({"status": "ok"})

@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, start_capture, 0)

@app.on_event("shutdown")
async def on_shutdown():
    global capture_running, cap
    capture_running = False
    with cap_lock:
        if cap: cap.release()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AMD ADAPT KIOSK")
    print("  http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
