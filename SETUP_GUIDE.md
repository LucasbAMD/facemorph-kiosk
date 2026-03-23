# Strix Halo Setup Guide

## Machine Requirements
- Ubuntu 22.04 or 24.04
- AMD Strix Halo APU (gfx1150 — auto-detected)
- Internet connection
- ~20 GB free disk space

## Step 1: Install (one-time, ~20-30 minutes)

```bash
git clone https://github.com/LucasbAMD/facemorph-kiosk.git && cd facemorph-kiosk
bash bootstrap.sh
```

This will automatically:
1. Install system packages (build tools, OpenCV deps)
2. Detect ROCm — if missing, installs ROCm 7.2 automatically
3. Detect your GPU (should show `gfx1150` for Strix Halo)
4. Create a Python virtual environment
5. Install PyTorch with the correct ROCm wheels
6. Install all Python dependencies
7. Download AI models (~15 GB)
8. Set GPU device permissions

**If ROCm was just installed, you must reboot before continuing:**
```bash
sudo reboot
```

## Step 2: Start the kiosk

```bash
cd facemorph-kiosk
source kiosk_venv/bin/activate
python start.py
```

Then open **http://localhost:8000** in a browser.

You should see:
```
[OK] Detected AMD GPU: gfx1150
     HSA_OVERRIDE_GFX_VERSION=11.0.0
[OK] ControlNet Depth SDXL — found in cache
[OK] Mode: ControlNet + SDXL Base (best quality)
[OK] Starting kiosk at http://localhost:8000
```

## After a Reboot

ROCm and models persist. Just start the app:
```bash
cd facemorph-kiosk
source kiosk_venv/bin/activate
python start.py
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ROCm drivers not found` after reboot | Run `bootstrap.sh` again — it will skip what's already done |
| `GPU not detected by PyTorch` | Reboot after ROCm install, then re-run `bootstrap.sh` |
| `No module named torch` | You forgot to activate the venv: `source kiosk_venv/bin/activate` |
| Black screen / no camera | Check `ls /dev/video*` — camera must be plugged in |
| Slow generation | First run downloads/compiles kernels — second run will be faster |
| `Permission denied` on `/dev/kfd` | Run: `sudo chmod 666 /dev/kfd /dev/dri/renderD*` |

## Quick Reference

```bash
# Full setup (first time only)
git clone https://github.com/LucasbAMD/facemorph-kiosk.git && cd facemorph-kiosk
bash bootstrap.sh
# reboot if ROCm was installed

# Start (every time)
cd facemorph-kiosk
source kiosk_venv/bin/activate
python start.py
# open http://localhost:8000
```
