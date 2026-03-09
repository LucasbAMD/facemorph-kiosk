#!/bin/bash
# download_models.sh — Download AI models for AMD Adapt Kiosk
# Run this once on the Threadripper machine

COMFY_DIR="$HOME/ComfyUI"
MODELS="$COMFY_DIR/models"

echo ""
echo "======================================================"
echo "  AMD Adapt Kiosk — Model Download"
echo "======================================================"
echo ""

mkdir -p "$MODELS/checkpoints" "$MODELS/controlnet"

# SDXL-Turbo
SDXL="$MODELS/checkpoints/sd_xl_turbo_1.0_fp16.safetensors"
if [ ! -f "$SDXL" ]; then
    echo "[..] Downloading SDXL-Turbo (~3.1 GB)..."
    wget --show-progress -q \
        "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors" \
        -O "$SDXL"
    echo "[OK] SDXL-Turbo done"
else
    echo "[OK] SDXL-Turbo already present"
fi

# ControlNet
CNET="$MODELS/controlnet/control-lora-canny-rank128.safetensors"
if [ ! -f "$CNET" ]; then
    echo "[..] Downloading ControlNet (~660 MB)..."
    wget --show-progress -q \
        "https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors" \
        -O "$CNET"
    echo "[OK] ControlNet done"
else
    echo "[OK] ControlNet already present"
fi

echo ""
echo "[OK] All models ready. Start the kiosk:"
echo "     Terminal 1: ~/start_comfyui.sh"
echo "     Terminal 2: cd ~/facemorph-kiosk && source venv/bin/activate && python start.py"
echo ""
