# AMD AI Scene Style Kiosk

AI-powered photo booth that transforms live camera feeds into stylized artwork using Stable Diffusion XL, ControlNet, and IP-Adapter FaceID — running locally on AMD GPUs with ROCm.

## Requirements

- **OS:** Ubuntu 22.04+
- **GPU:** AMD GPU (RX 7000 series tested) — ROCm drivers are installed automatically by the setup script
- **Disk:** ~20 GB free (for AI models)
- **Camera:** USB webcam
- **Internet:** Required for first-time model downloads

## Quick Setup (2 commands)

```bash
git clone https://github.com/LucasbAMD/facemorph-kiosk.git && cd facemorph-kiosk
bash bootstrap.sh
```

That's it. The bootstrap script handles everything:
1. Installs system packages (build tools, OpenCV deps)
2. **Installs ROCm drivers** if not already present (Ubuntu 22.04/24.04)
3. Creates a Python virtual environment (`kiosk_venv/`)
4. Installs PyTorch with ROCm support
5. Installs all Python dependencies
6. Downloads all AI models (~15 GB)
7. Sets GPU device permissions

> **First-time ROCm install?** If the script installs ROCm drivers for the first time, you'll need to **reboot** before starting the kiosk. After rebooting, you do NOT need to re-run `bootstrap.sh` — just start the kiosk (see below).

## Starting the Kiosk

```bash
cd facemorph-kiosk
source kiosk_venv/bin/activate
python start.py
```

Then open **http://localhost:8000** in a browser.

## Setup Instructions (for someone else)

> **Give these instructions to anyone setting up the kiosk on a fresh Ubuntu machine:**
>
> 1. Plug in a USB webcam
> 2. Open a terminal and run:
>    ```
>    git clone https://github.com/LucasbAMD/facemorph-kiosk.git
>    cd facemorph-kiosk
>    bash bootstrap.sh
>    ```
> 3. Wait for the setup to finish (~15-30 min depending on internet speed — it downloads ~15 GB of AI models)
> 4. **If ROCm was just installed for the first time**, the script will tell you to reboot. Reboot, then skip to step 5.
> 5. Start the kiosk:
>    ```
>    cd facemorph-kiosk
>    source kiosk_venv/bin/activate
>    python start.py
>    ```
> 6. Open **http://localhost:8000** in Chrome or Firefox
> 7. Pick a style from the sidebar and click **Generate** (or just stand still and it auto-triggers)
>
> **To restart the kiosk later** (e.g. after a reboot):
> ```
> cd facemorph-kiosk
> source kiosk_venv/bin/activate
> python start.py
> ```

## Project Structure

```
facemorph-kiosk/
├── bootstrap.sh        # One-command setup script
├── start.py            # App launcher (configures ROCm env)
├── main.py             # FastAPI web server
├── generator.py        # AI generation pipeline
├── face_processor.py   # Face detection & recognition
├── setup_models.py     # Model downloader
├── requirements.txt    # Python dependencies
├── index.html          # Web UI
└── assets/fonts/       # Nunito font files
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GPU not detected` | Reboot if ROCm was just installed. Otherwise install manually: `sudo amdgpu-install --usecase=rocm` then reboot |
| `No camera found` | Check `ls /dev/video*` — plug in a USB webcam |
| `Missing packages` | Re-run `source kiosk_venv/bin/activate && pip install -r requirements.txt` |
| `Models not found` | Run `python setup_models.py` to re-download |
| `Permission denied on /dev/kfd` | Run `sudo chmod 666 /dev/kfd /dev/dri/renderD128` |
