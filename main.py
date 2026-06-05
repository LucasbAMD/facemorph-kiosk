"""
main.py — AI Scene Style Kiosk Backend
Live webcam feed + AI scene style transfer.
"""

import cv2
import asyncio
import threading
import time
import base64
import io
import os
import socket
import uuid
from collections import OrderedDict
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import (StreamingResponse, HTMLResponse, JSONResponse,
                               Response)
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

# ── Result store + QR-to-phone delivery ───────────────────────────────────────
# After a transform, the result image is kept in memory under a short random id.
# The result overlay shows a QR code that encodes  http://<kiosk-lan-ip>:<port>/r/<id>
# A visitor on the same Wi-Fi scans it, opens that page on their phone, and
# downloads the image. No internet, accounts, or cloud storage required.
#
# Tuning via env:
#   KIOSK_PUBLIC_HOST  — override the host/IP put in the QR URL (e.g. a DNS name
#                        or a tunnel). If unset, the LAN IP is auto-detected.
#   KIOSK_PORT         — port advertised in the QR URL (default 8000).
#   KIOSK_RESULT_TTL   — seconds to keep a result in memory (default 1800 = 30m).
#   KIOSK_MAX_RESULTS  — max results kept before evicting oldest (default 50).

KIOSK_PORT = int(os.environ.get("KIOSK_PORT", "8000"))
RESULT_TTL = int(os.environ.get("KIOSK_RESULT_TTL", "1800"))
MAX_RESULTS = int(os.environ.get("KIOSK_MAX_RESULTS", "50"))

# id -> {"jpg": bytes, "ts": float}
_results = OrderedDict()
_results_lock = threading.Lock()


def _detect_lan_ip():
    """Best-effort detection of this machine's primary LAN IPv4 address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def _public_base_url():
    """Base URL phones should use to reach this kiosk."""
    host = os.environ.get("KIOSK_PUBLIC_HOST", "").strip()
    if not host:
        host = _detect_lan_ip()
    return f"http://{host}:{KIOSK_PORT}"


def _store_result(jpg_bytes):
    """Save a JPEG result, return its short id. Evicts old/expired entries."""
    rid = uuid.uuid4().hex[:10]
    now = time.time()
    with _results_lock:
        expired = [k for k, v in _results.items() if now - v["ts"] > RESULT_TTL]
        for k in expired:
            _results.pop(k, None)
        while len(_results) >= MAX_RESULTS:
            _results.popitem(last=False)
        _results[rid] = {"jpg": jpg_bytes, "ts": now}
    return rid


def _get_result(rid):
    with _results_lock:
        entry = _results.get(rid)
        if entry is None:
            return None
        if time.time() - entry["ts"] > RESULT_TTL:
            _results.pop(rid, None)
            return None
        return entry["jpg"]


def _make_qr_data_url(url):
    """Return a base64 PNG data URL of a QR code for `url`, or None if the
    qrcode library isn't installed (feature degrades gracefully)."""
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except ImportError:
        print("[QR] qrcode library not installed — QR disabled. "
              "Install with: pip install qrcode[pil]")
        return None
    except Exception as e:
        print(f"[QR] Failed to render QR: {e}")
        return None


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


# ── Run counter ───────────────────────────────────────────────────────────────
_run_count: int = 0
_run_lock = threading.Lock()

# ── Co-brand overlay state ────────────────────────────────────────────────────
_cobrand_name: Optional[str] = None
_cobrand_lock = threading.Lock()


_FONT_PATH = str(Path(__file__).parent / "assets" / "fonts" / "Nunito-Variable.ttf")


def _get_font(size: int, variation: str = None) -> ImageFont.FreeTypeFont:
    """Load Nunito at the requested pixel size, fall back to default."""
    try:
        font = ImageFont.truetype(_FONT_PATH, size)
        if variation:
            font.set_variation_by_name(variation)
        return font
    except OSError:
        return ImageFont.load_default()


def _apply_cobrand_overlay(img: np.ndarray, name: str) -> np.ndarray:
    """Render 'AMD x <name>' on the top-left using Nunito (rounded sans-serif)."""
    label = f"AMD x {name}"
    h, w = img.shape[:2]

    font_size = max(16, int(w / 28))  # ~73px at 2048, ~37px at 1024
    font = _get_font(font_size)

    # Convert BGR -> RGBA for compositing
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    margin = int(w * 0.02)
    x, y = margin, margin

    # Soft dark shadow for readability (semi-transparent, offset 2px)
    draw.text((x + 2, y + 2), label, font=font, fill=(0, 0, 0, 120))
    # White text on top, slightly transparent so it's not harsh
    draw.text((x, y), label, font=font, fill=(255, 255, 255, 210))

    result = Image.alpha_composite(pil_img, overlay)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGR)


def _apply_watermark(img: np.ndarray) -> np.ndarray:
    """Render a polaroid-style bottom banner with centered label."""
    label = "AMD Customer Engagement Program"
    h, w = img.shape[:2]

    font_size = max(12, int(w / 45))
    font = _get_font(font_size, variation="SemiBold")

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Measure text height for banner sizing
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    padding = int(text_h * 0.8)
    banner_h = text_h + padding * 2

    # Semi-transparent black banner across full width at bottom (~80% opaque)
    banner_y = h - banner_h
    draw.rectangle([(0, banner_y), (w, h)], fill=(0, 0, 0, 204))

    # Centered white text on a separate layer so we can blur edges
    text_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    text_x = (w - text_w) // 2
    text_y = banner_y + (banner_h - text_h) // 2
    text_draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255, 230))
    text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=0.8))

    overlay = Image.alpha_composite(overlay, text_layer)

    result = Image.alpha_composite(pil_img, overlay)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGR)


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
                # Bump run counter
                global _run_count
                with _run_lock:
                    _run_count += 1
                # Apply co-brand overlay if a name was provided
                with _cobrand_lock:
                    cobrand = _cobrand_name
                if cobrand:
                    result = _apply_cobrand_overlay(result, cobrand)
                # Always apply the AMD CEC watermark (bottom-left, subtle)
                result = _apply_watermark(result)
                _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                jpg_bytes = buf.tobytes()
                b64 = base64.b64encode(jpg_bytes).decode()
                # Store result + build a QR pointing to the phone-download page.
                rid = _store_result(jpg_bytes)
                share_url = f"{_public_base_url()}/r/{rid}"
                qr_data_url = _make_qr_data_url(share_url)
                print(f"[Status] Result ready: {result.shape}, "
                      f"JPEG size: {len(jpg_bytes)}bytes, id={rid}, "
                      f"share={share_url}")
                return JSONResponse({"status": "done",
                                     "image": f"data:image/jpeg;base64,{b64}",
                                     "result_id": rid,
                                     "share_url": share_url,
                                     "qr": qr_data_url})
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


@app.get("/run_count")
async def run_count():
    with _run_lock:
        count = _run_count
    return JSONResponse({"count": count})


@app.get("/status")
async def status():
    with frame_lock:
        has_frame = latest_frame is not None
    return {"camera": has_frame, "ai_ready": comfy.available,
            "mode": comfy.get_mode()}


@app.get("/health")
async def health():
    """Diagnostics: which backend/profile is active and how phones reach us."""
    info = {"ai_ready": comfy.check_available(), "mode": comfy.get_mode(),
            "share_base_url": _public_base_url()}
    try:
        import generator
        dev, kind, _ = generator.get_device()
        pname, prof = generator.get_profile()
        info.update({"device": str(dev), "device_kind": kind,
                     "profile": pname, "profile_detail": prof})
    except Exception as e:
        info["device_error"] = str(e)
    return JSONResponse(info)


# ── QR result delivery (phone download) ───────────────────────────────────────
@app.get("/r/{rid}", response_class=HTMLResponse)
async def result_page(rid: str):
    """Mobile-friendly page shown when a visitor scans the QR with their phone."""
    jpg = _get_result(rid)
    if jpg is None:
        return HTMLResponse(
            "<!doctype html><meta name='viewport' content='width=device-width,"
            "initial-scale=1'><body style='font-family:system-ui;background:#111;"
            "color:#eee;text-align:center;padding:3rem 1.5rem'>"
            "<h2>Link expired</h2><p>This photo is no longer available. "
            "Please take a new one at the kiosk.</p></body>",
            status_code=404)
    img_src = f"/r/{rid}/image"
    dl_name = f"amd-facemorph-{rid}.jpg"
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your AMD FaceMorph</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin:0;
         background:#0b0b0d; color:#f4f4f5; text-align:center; }}
  header {{ padding:1.1rem 1rem .4rem; font-weight:700; letter-spacing:.02em; }}
  .sub {{ color:#a1a1aa; font-size:.85rem; margin:.1rem 0 1rem; }}
  .wrap {{ max-width:560px; margin:0 auto; padding:0 1rem 2.5rem; }}
  img.result {{ width:100%; height:auto; border-radius:14px;
               box-shadow:0 8px 30px rgba(0,0,0,.5); }}
  a.btn {{ display:block; margin:1.25rem auto 0; padding:1rem 1.25rem;
          background:#e2231a; color:#fff; text-decoration:none; font-weight:700;
          border-radius:12px; font-size:1.05rem; }}
  a.btn:active {{ filter:brightness(.9); }}
  .hint {{ color:#a1a1aa; font-size:.8rem; margin-top:1rem; line-height:1.4; }}
</style></head>
<body>
  <header>Your AMD FaceMorph is ready</header>
  <div class="sub">AMD Customer Engagement Program</div>
  <div class="wrap">
    <img class="result" src="{img_src}" alt="Your generated image">
    <a class="btn" href="{img_src}" download="{dl_name}">Save to my phone</a>
    <p class="hint">If the button doesn't save it, press and hold the image
       above and choose &ldquo;Save Image&rdquo;.</p>
  </div>
</body></html>"""
    return HTMLResponse(html)


@app.get("/r/{rid}/image")
async def result_image(rid: str):
    """Raw JPEG, shown inline on the phone page."""
    jpg = _get_result(rid)
    if jpg is None:
        raise HTTPException(404, "Result expired")
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/r/{rid}/download")
async def result_download(rid: str):
    """JPEG with an attachment header to force a download."""
    jpg = _get_result(rid)
    if jpg is None:
        raise HTTPException(404, "Result expired")
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Content-Disposition":
                             f'attachment; filename="amd-facemorph-{rid}.jpg"'})


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
    print(f"  Local:  http://localhost:{KIOSK_PORT}")
    print(f"  Phones: {_public_base_url()}  (QR target)")
    print("="*55 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=KIOSK_PORT,
                reload=False, workers=1)
