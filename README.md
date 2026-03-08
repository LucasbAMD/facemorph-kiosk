# AMD ADAPT KIOSK

Real-time webcam character transformation powered by AMD W7900 + ROCm.

Walk up, pick a character, press **⚡ TRANSFORM ME** — see yourself as a photorealistic Na'vi, Hulk, Thanos, and more in ~2 seconds.

---

## How It Works

**Live preview (instant):** Body segmentation + color transforms + pose-tracked markings appear in real time as you stand in front of the camera.

**AI transform (~2 sec):** Press ENTER or click ⚡ TRANSFORM ME. SDXL-Turbo + ControlNet on the W7900 generates a photorealistic version of you as the character, preserving your body pose.

---

## Characters

| # | Character | Live Effect | AI Effect |
|---|-----------|-------------|-----------|
| 1 | **Na'vi** | Blue body + bioluminescent markings | Photorealistic Avatar Na'vi |
| 2 | **Hulk** | Gamma green + muscle veins | Photorealistic Hulk |
| 3 | **Thanos** | Purple body + Infinity Stone glows | Photorealistic Thanos |
| 4 | **Predator** | Thermal heatmap + targeting reticle | Thermal alien warrior |
| 5 | **Ghost** | Pale body + spectral aura | Translucent spectral figure |
| 6 | **Groot** | Bark texture + vine outlines | Living tree humanoid |
| — | **Minecraft** | Steve face + full pixelation | — |
| — | **Cyberpunk** | Neon edge detection | — |

Keyboard: `1–6` pick character · `ENTER` generate AI · `ESC` live view · `F` face swap panel

---

## Setup — Ubuntu (Recommended for AI generation)

```bash
git clone https://github.com/WudiRL/facemorph-kiosk.git
cd facemorph-kiosk
bash setup_ubuntu.sh
```

The script installs everything: ROCm, PyTorch, ComfyUI, SDXL-Turbo (~3.1GB), and ControlNet. Takes 20–40 min on first run.

**Requirements:** Ubuntu 22.04/24.04 · AMD GPU with ROCm support (W7900 recommended) · 15GB disk

### Run

```bash
# Terminal 1 — AI engine
~/start_comfyui.sh

# Terminal 2 — Kiosk
source venv/bin/activate && python start.py
```

Browser opens automatically at http://localhost:8000

---

## Setup — Windows (Live mode only)

```bash
python install.py
python start.py
```

Live color transforms and body markings work on Windows via DirectML. AI generation requires Ubuntu + ROCm.

---

## Hardware

Built for: **AMD Threadripper 9995X + Radeon Pro W7900**
- W7900 handles all ML inference via ROCm (rembg segmentation + SDXL-Turbo generation)
- Threadripper handles parallel camera/pose/compositing threads
- DirectML on Windows for live-only mode

---

## Files

| File | Purpose |
|------|---------|
| `setup_ubuntu.sh` | Full Ubuntu + ROCm + ComfyUI setup |
| `install.py` | Windows live-mode setup |
| `start.py` | Launch kiosk |
| `launch_comfy.py` | Start ComfyUI from Python |
| `main.py` | FastAPI backend |
| `face_processor.py` | Live body transforms |
| `comfy_bridge.py` | ComfyUI AI generation bridge |
| `index.html` | Kiosk UI |
| `faces/fantasy/` | Character reference photos |
