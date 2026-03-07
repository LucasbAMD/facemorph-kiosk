# 🎭 FaceMorph Kiosk

Real-time full-body character transformation via webcam. Walk up, pick a character, become them instantly.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/facemorph-kiosk.git
cd facemorph-kiosk
python install.py      # one-time setup (~20-30 min, mostly downloading PyTorch)
python start.py        # every time after
```

Open **http://localhost:8000**

## Characters

| Key | Character | Effect |
|-----|-----------|--------|
| `1` | 🔵 Na'vi | Blue skin, bioluminescent markings, eye glow |
| `2` | 💚 Hulk | Gamma green, vein lines, brow furrow |
| `3` | 💜 Thanos | Deep purple, Infinity Stone joint dots |
| `4` | 👁️ Predator | Thermal heatmap + scan grid |
| `5` | 👻 Ghost | Semi-transparent, white aura, hollow eyes |
| `6` | 🌿 Groot | Bark brown, forest green nature markings |

Plus **Face Swap** mode — swap to any celebrity photo you drop in.

## Hardware

- **Tested on:** AMD Threadripper 9995X + W7900 GPU
- **Camera:** Logitech Brio 4K (any USB webcam works)
- **GPU:** Auto-detects ROCm (AMD) → CUDA (NVIDIA) → CPU

## Face Swap Setup

1. Drop face photos into `faces/celebrity/` (e.g. `Lisa_Su.jpg`)
2. Download the swap model:
```bash
wget -P models/ https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1–6` | Switch character |
| `F` | Open Face Swap panel |
| `ESC` | Return to live view |
