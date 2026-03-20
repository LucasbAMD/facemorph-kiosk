"""
main.py — AI Scene Style Kiosk Backend
Live webcam feed + AI scene style transfer.
"""

import cv2
import asyncio
import threading
import time
import base64
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from generator import ComfyBridge

app = FastAPI(title="AMD-Adapt")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

comfy = ComfyBridge()

cap: Optional[cv2.VideoCapture] = None
cap_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()
capture_running = False

# ── Motion detection state ────────────────────────────────────────────────────
prev_gray = None
motion_lock = threading.Lock()
stable_since: float = 0.0        # timestamp when scene became stable
motion_score: float = 999.0      # current frame-diff score (lower = more stable)
MOTION_THRESHOLD = 3.0           # below this = "holding still"


# ── Co-brand overlay state ────────────────────────────────────────────────────
_cobrand_name: Optional[str] = None
_cobrand_lock = threading.Lock()


def _apply_cobrand_overlay(img: np.ndarray, name: str) -> np.ndarray:
    """Render 'AMD X <name>' badge on the top-left of the result image."""
    label = f"AMD X {name}"
    h, w = img.shape[:2]

    # Scale font relative to image width so it's visible on stickers (~3-4% of width)
    # At 2048px wide this gives roughly 60-80px tall text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 900          # ~2.3 at 2048px, ~1.1 at 1024px
    thickness = max(2, int(w / 500))  # ~4 at 2048px

    # Measure text
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Padding around text
    pad_x = int(w * 0.015)
    pad_y = int(h * 0.010)
    margin = int(w * 0.02)

    # Box coordinates
    x1 = margin
    y1 = margin
    x2 = x1 + tw + 2 * pad_x
    y2 = y1 + th + baseline + 2 * pad_y

    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # White text
    text_x = x1 + pad_x
    text_y = y1 + pad_y + th
    cv2.putText(img, label, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    return img


# ── Camera ────────────────────────────────────────────────────────────────────
def start_capture(camera_index: int = 0):
    global cap, latest_frame, capture_running
    capture_running = True
    opened_index = camera_index
    with cap_lock:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            for alt in [2, 1, 0, 3]:
                if alt == camera_index:
                    continue
                cap = cv2.VideoCapture(alt)
                if cap.isOpened():
                    opened_index = alt
                    print(f"[OK] Camera opened at index {alt}")
                    break
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

    def loop():
        global latest_frame, prev_gray, stable_since, motion_score
        while capture_running:
            with cap_lock:
                ok = cap and cap.isOpened()
            if not ok:
                time.sleep(0.1)
                continue
            with cap_lock:
                ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame
                # Motion detection: compare grayscale frames
                small = cv2.resize(frame, (320, 180))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                with motion_lock:
                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray)
                        score = float(np.mean(diff))
                        motion_score = score
                        if score < MOTION_THRESHOLD:
                            if stable_since == 0.0:
                                stable_since = time.time()
                        else:
                            stable_since = 0.0
                    else:
                        stable_since = 0.0
                    prev_gray = gray
            else:
                time.sleep(0.033)

    threading.Thread(target=loop, daemon=True).start()
    print(f"[OK] Camera started (index={opened_index})")


def _placeholder(w=1280, h=720):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for camera...", (w//2-220, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 80), 2)
    return img


# ── MJPEG stream ──────────────────────────────────────────────────────────────
def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            processed = _placeholder()
        else:
            processed = frame
        # Downscale for streaming to reduce JPEG encoding cost
        h, w = processed.shape[:2]
        if w > 1280:
            scale = 1280 / w
            processed = cv2.resize(processed, (1280, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(1/30)


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


# ── AI Generation ─────────────────────────────────────────────────────────────
@app.post("/generate")
async def generate(character: str = Form(...),
                   cobrand: Optional[str] = Form(None)):
    global _cobrand_name
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        raise HTTPException(400, "No camera frame available")
    if not comfy.check_available():
        raise HTTPException(503, "AI engine not ready - loading model...")

    with _cobrand_lock:
        _cobrand_name = cobrand.strip() if cobrand else None

    started = comfy.generate(frame, character)
    if not started:
        return JSONResponse({"status": "busy", "message": "Already generating"})
    return JSONResponse({"status": "generating", "message": "Transforming scene..."})


@app.get("/generate_status")
async def generate_status():
    status = comfy.get_status()
    if status["status"] == "done":
        result = comfy.get_result()
        if result is not None:
            try:
                # Apply co-brand overlay if a name was provided
                with _cobrand_lock:
                    cobrand = _cobrand_name
                if cobrand:
                    result = _apply_cobrand_overlay(result, cobrand)
                _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                b64 = base64.b64encode(buf.tobytes()).decode()
                print(f"[Status] Result ready: {result.shape}, JPEG size: {len(buf)}bytes")
                return JSONResponse({"status": "done",
                                     "image": f"data:image/jpeg;base64,{b64}"})
            except Exception as e:
                print(f"[Status] Error encoding result: {e}")
                return JSONResponse({"status": "error", "message": f"Encoding failed: {e}"})
        else:
            # Result was already consumed or missing
            print("[Status] Done but result is None (already consumed?)")
            return JSONResponse({"status": "idle", "message": ""})
    return JSONResponse(status)


@app.get("/comfy_status")
async def comfy_status():
    return JSONResponse({
        "available": comfy.check_available(),
        "mode": comfy.get_mode(),
    })


@app.get("/motion_status")
async def motion_status_endpoint():
    with motion_lock:
        dur = (time.time() - stable_since) if stable_since > 0 else 0.0
        score = motion_score
    return JSONResponse({"stable_seconds": round(dur, 2), "motion_score": round(score, 2)})


@app.post("/motion_reset")
async def motion_reset():
    """Reset stable timer (called when generation starts to prevent re-trigger)."""
    global stable_since
    with motion_lock:
        stable_since = 0.0
    return JSONResponse({"ok": True})


@app.get("/status")
async def status():
    with frame_lock:
        has_frame = latest_frame is not None
    return {"camera": has_frame, "ai_ready": comfy.available,
            "mode": comfy.get_mode()}


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, start_capture, 0)


@app.on_event("shutdown")
async def on_shutdown():
    global capture_running, cap
    capture_running = False
    with cap_lock:
        if cap:
            cap.release()


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AMD-ADAPT")
    print("  http://localhost:8000")
    print("="*55 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
