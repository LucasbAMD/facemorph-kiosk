# AMD AI Scene Style Kiosk

AI-powered photo booth that transforms live camera feeds into stylized artwork using Stable Diffusion XL, ControlNet, and IP-Adapter FaceID — running locally on AMD GPUs with ROCm.

## Requirements

- **OS:** Ubuntu 22.04+
- **GPU:** AMD GPU with ROCm support (RX 7000 series tested)
- **Disk:** ~20 GB free (for AI models)
- **Camera:** USB webcam
- **Internet:** Required for first-time model downloads

## Quick Setup (2 commands)

```bash
git clone <repo-url> && cd facemorph-kiosk
bash bootstrap.sh
```

That's it. The bootstrap script handles everything:
1. Installs system packages (build tools, OpenCV deps)
2. Creates a Python virtual environment (`kiosk_venv/`)
3. Installs PyTorch with ROCm support
4. Installs all Python dependencies
5. Downloads all AI models (~15 GB)
6. Sets GPU device permissions

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
> 1. Make sure you have an AMD GPU with ROCm drivers installed
> 2. Plug in a USB webcam
> 3. Open a terminal and run:
>    ```
>    git clone <repo-url>
>    cd facemorph-kiosk
>    bash bootstrap.sh
>    ```
> 4. Wait for the setup to finish (~15-30 min depending on internet speed — it downloads ~15 GB of AI models)
> 5. When it says "Setup complete!", start the kiosk:
>    ```
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
| `GPU not detected` | Install ROCm drivers: `sudo apt install rocm-dev` then reboot |
| `No camera found` | Check `ls /dev/video*` — plug in a USB webcam |
| `Missing packages` | Re-run `source kiosk_venv/bin/activate && pip install -r requirements.txt` |
| `Models not found` | Run `python setup_models.py` to re-download |
| `Permission denied on /dev/kfd` | Run `sudo chmod 666 /dev/kfd /dev/dri/renderD128` |
