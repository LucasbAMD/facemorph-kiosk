#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_instantid.sh — AMD Adapt Kiosk
#
# Installs all three InstantID components into your existing ComfyUI setup:
#   1. ComfyUI_InstantID custom node  (from cubiq/ComfyUI_InstantID)
#   2. ip-adapter_instantid.bin       (face identity model, ~900 MB)
#   3. instantid-controlnet.safetensors (face keypoint ControlNet, ~1.4 GB)
#   4. insightface + onnxruntime      (installed into the ComfyUI venv)
#
# Run ONCE after setup_ubuntu.sh has completed.
# ComfyUI does NOT need to be running during install.
# Restart ComfyUI after this script finishes.
#
# Usage:
#   bash ~/facemorph-kiosk/install_instantid.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

COMFY_DIR="$HOME/ComfyUI"
VENV="$HOME/comfyui-venv"
NODES_DIR="$COMFY_DIR/custom_nodes"
IPADAPTER_DIR="$COMFY_DIR/models/instantid"
CONTROLNET_DIR="$COMFY_DIR/models/controlnet"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}[OK]${RESET}  $1"; }
info() { echo -e "${CYAN}[..]${RESET}  $1"; }
err()  { echo -e "${RED}[ERR]${RESET} $1"; exit 1; }

echo -e "\n${BOLD}AMD Adapt Kiosk — InstantID Installer${RESET}\n"

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -d "$COMFY_DIR" ]]  || err "ComfyUI not found at $COMFY_DIR. Run setup_ubuntu.sh first."
[[ -d "$VENV" ]]       || err "ComfyUI venv not found at $VENV. Run setup_ubuntu.sh first."

# ── 1. Custom node ────────────────────────────────────────────────────────────
NODE_PATH="$NODES_DIR/ComfyUI_InstantID"
if [[ -d "$NODE_PATH" ]]; then
    info "ComfyUI_InstantID node already present — pulling latest"
    git -C "$NODE_PATH" pull --quiet
else
    info "Cloning ComfyUI_InstantID custom node..."
    git clone --depth 1 https://github.com/cubiq/ComfyUI_InstantID.git "$NODE_PATH"
fi
ok "Custom node ready at $NODE_PATH"

# ── 2. Python dependencies ────────────────────────────────────────────────────
info "Installing insightface + onnxruntime into ComfyUI venv..."
source "$VENV/bin/activate"

# onnxruntime-rocm is ideal but insightface only needs CPU onnxruntime for face analysis
pip install --quiet insightface onnxruntime

# buffalo_l face analysis model (auto-downloaded by insightface on first use)
# Force the download now so the kiosk doesn't stall on first generation
python3 - <<'PYEOF'
import insightface
app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("[insightface] buffalo_l model ready")
PYEOF

deactivate
ok "insightface + onnxruntime installed, buffalo_l model cached"

# ── 3. InstantID ip-adapter model ─────────────────────────────────────────────
mkdir -p "$IPADAPTER_DIR"
IID_MODEL="$IPADAPTER_DIR/ip-adapter_instantid.bin"
if [[ -f "$IID_MODEL" ]]; then
    ok "ip-adapter_instantid.bin already present ($(du -h "$IID_MODEL" | cut -f1))"
else
    info "Downloading ip-adapter_instantid.bin (~900 MB)..."
    wget -q --show-progress \
        "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin" \
        -O "$IID_MODEL"
    ok "ip-adapter_instantid.bin downloaded"
fi

# ── 4. InstantID ControlNet ───────────────────────────────────────────────────
mkdir -p "$CONTROLNET_DIR"
IID_CNET="$CONTROLNET_DIR/instantid-controlnet.safetensors"
if [[ -f "$IID_CNET" ]]; then
    ok "instantid-controlnet.safetensors already present ($(du -h "$IID_CNET" | cut -f1))"
else
    info "Downloading instantid-controlnet.safetensors (~1.4 GB)..."
    wget -q --show-progress \
        "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors" \
        -O "$IID_CNET"
    ok "instantid-controlnet.safetensors downloaded"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}InstantID install complete.${RESET}"
echo ""
echo "  Node:       $NODE_PATH"
echo "  Model:      $IID_MODEL  ($(du -h "$IID_MODEL" | cut -f1))"
echo "  ControlNet: $IID_CNET   ($(du -h "$IID_CNET" | cut -f1))"
echo ""
echo -e "${CYAN}Next step:${RESET} restart ComfyUI, then the kiosk will automatically"
echo "use InstantID for all character generations."
echo ""
echo "  Terminal 1:  ~/start_comfyui.sh"
echo "  Terminal 2:  bash ~/facemorph-kiosk/start.sh"
echo ""
