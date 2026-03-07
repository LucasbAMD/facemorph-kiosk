"""
main.py  —  FaceSwap Kiosk Backend
FastAPI server that:
  - Streams processed webcam video as MJPEG
  - Accepts mode/face selection from the browser UI
  - Serves the kiosk frontend
Run with:  python main.py
"""

import cv2
import asyncio
import threading
import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import (
    StreamingResponse, HTMLResponse, JSONResponse, FileResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from face_processor import FaceProcessor

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="FaceSwap Kiosk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Global State ──────────────────────────────────────────────────────────────

processor = FaceProcessor(faces_dir="faces")

# Webcam capture (shared across all stream consumers)
cap: Optional[cv2.VideoCapture] = None
cap_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()
capture_running = False


def start_capture(camera_index: int = 0):
    global cap, latest_frame, capture_running
    capture_running = True

    # Try higher resolution for Logitech Brio
    with cap_lock:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def capture_loop():
        global latest_frame, capture_running
        while capture_running:
            with cap_lock:
                if cap is None or not cap.isOpened():
                    time.sleep(0.1)
                    continue
                ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame
            else:
                time.sleep(0.033)

    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print(f"[OK] Camera capture started (index={camera_index})")


def generate_frames():
    """Generator that yields MJPEG frames to the browser."""
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            # Send a black placeholder frame
            placeholder = _black_placeholder()
            _, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
        else:
            processed = processor.process_frame(frame)
            _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 82])

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1 / 30)  # ~30 fps cap


def _black_placeholder(w=1280, h=720):
    img = __import__("numpy").zeros((h, w, 3), dtype=__import__("numpy").uint8)
    cv2.putText(img, "Waiting for camera...", (w // 2 - 220, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 80), 2)
    return img


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/set_mode")
async def set_mode(mode: str = Form(...), face_key: str = Form(None),
                   character: str = Form(None)):
    """Switch processing mode: passthrough | character | minecraft | cyberpunk | faceswap"""
    valid = {"passthrough", "character", "minecraft", "cyberpunk", "faceswap"}
    if mode not in valid:
        raise HTTPException(400, f"Invalid mode. Choose from: {valid}")
    processor.set_mode(mode, face_key, character)
    return {"status": "ok", "mode": mode, "face_key": face_key, "character": character}


@app.get("/characters")
async def get_characters():
    """Return list of available full-body character presets."""
    return JSONResponse({"characters": processor.get_characters()})


@app.get("/catalog")
async def get_catalog():
    """Return list of available face swap targets."""
    return JSONResponse({"faces": processor.get_catalog()})


@app.post("/upload_face")
async def upload_face(
    file: UploadFile = File(...),
    label: str = Form(...),
    category: str = Form("custom")
):
    """Upload a new face image to the catalog."""
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Only JPG/PNG/WEBP supported")

    dest_dir = Path("faces") / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_").lower()
    dest = dest_dir / f"{safe_label}{ext}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    processor.reload_catalog()
    return {"status": "ok", "key": f"{category}/{safe_label}", "label": label}


@app.get("/face_thumb")
async def face_thumb(key: str):
    """Serve face thumbnail from catalog."""
    entry = processor.face_catalog.get(key)
    if not entry:
        raise HTTPException(404, "Face not found")
    return FileResponse(entry["path"])


@app.post("/reload_catalog")
async def reload_catalog():
    processor.reload_catalog()
    return {"status": "ok", "count": len(processor.face_catalog)}


@app.get("/status")
async def status():
    with frame_lock:
        has_frame = latest_frame is not None
    return {
        "camera": has_frame,
        "mode": processor.current_mode,
        "face_key": processor.current_face_key,
        "face_count": len(processor.face_catalog),
    }


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    # Start camera in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, start_capture, 0)


@app.on_event("shutdown")
async def on_shutdown():
    global capture_running, cap
    capture_running = False
    with cap_lock:
        if cap:
            cap.release()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🎭  FaceSwap Kiosk — Starting up")
    print("  📡  Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
