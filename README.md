# AMD AI Scene Style Kiosk — Windows / R9700 build

AI-powered photo booth that transforms a live camera feed into stylized
artwork using Stable Diffusion XL, ControlNet, and IP-Adapter FaceID — running
locally on the **AMD Radeon AI PRO R9700** (RDNA 4 / gfx1201) under
**Windows 11**.

> **Linux build?** This branch (`claude/windows-r9700-support-3swMZ`) targets
> Windows. For the original Ubuntu + ROCm build, use `main`.

## Requirements

- **OS:** Windows 11 (22H2 or newer)
- **GPU:** AMD Radeon AI PRO R9700 (other RDNA 3 / RDNA 4 cards should also work)
- **Driver:** Latest AMD Adrenalin
- **AMD HIP SDK 6.4+** — [download here](https://www.amd.com/en/developer/resources/rocm-hub.html)
- **Python 3.10, 3.11, or 3.12** — [python.org](https://www.python.org/downloads/) (check "Add to PATH" during install)
- **Disk:** ~20 GB free (for AI models)
- **Camera:** USB webcam
- **Internet:** Required for first-time model downloads

## Quick Setup

Open **PowerShell** in the repo directory and run:

```powershell
# One-time only — allow local scripts:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Bootstrap:
.\bootstrap.ps1
```

The bootstrap script will:
1. Verify Python is installed
2. Check for the AMD HIP SDK (warns if missing — falls back to DirectML)
3. Create a Python venv (`kiosk_venv\`)
4. Install PyTorch with ROCm-Windows wheels (or `torch-directml` as a fallback)
5. Install all Python dependencies
6. Download all AI models (~15 GB)

> **HIP SDK not installed?** The bootstrap will still run by falling back to
> `torch-directml`, which works on any DX12 GPU. For the best performance on
> the R9700, install the HIP SDK and re-run `bootstrap.ps1`.

## Starting the Kiosk

```powershell
.\start.ps1
```

Or manually:

```powershell
.\kiosk_venv\Scripts\Activate.ps1
python start.py
```

Then open **http://localhost:8000** in a browser.

> A `start.bat` is also provided for `cmd.exe` users.

## Cloning the repo on the demo desktop

```powershell
git clone https://github.com/LucasbAMD/facemorph-kiosk.git
cd facemorph-kiosk
git checkout claude/windows-r9700-support-3swMZ
.\bootstrap.ps1
```

If the repo is private, sign in first with the [GitHub CLI](https://cli.github.com/):
`gh auth login`, then clone.

## Project Structure

```
facemorph-kiosk/
├── bootstrap.ps1       # Windows setup script  (PowerShell)
├── bootstrap.sh        # Linux setup script    (bash, kept for reference)
├── start.ps1           # Windows launcher       (PowerShell)
├── start.bat           # Windows launcher       (cmd.exe)
├── start.py            # Cross-platform app launcher
├── main.py             # FastAPI web server
├── generator.py        # AI generation pipeline
├── face_processor.py   # Face detection & recognition
├── setup_models.py     # Model downloader
├── check_gpu.py        # GPU diagnostic (cross-platform)
├── requirements.txt    # Python dependencies
├── index.html          # Web UI
└── assets/fonts/       # Nunito font files
```

## Diagnostics

Run the GPU check to confirm the backend is alive:

```powershell
.\kiosk_venv\Scripts\Activate.ps1
python check_gpu.py
```

You should see something like:

```
Platform: Windows 11
PyTorch:  2.x.x+rocm6.4
CUDA/ROCm available: True
GPU:  AMD Radeon AI PRO R9700
VRAM: 32.0 GB
Tensor on GPU: cuda:0 -- GPU is working
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Set-ExecutionPolicy` error | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once |
| GPU not detected by PyTorch | Install AMD HIP SDK 6.4+ and reboot. Verify with `hipinfo` in a new shell |
| `No camera found` | Plug in a USB webcam, then check Windows **Settings → Privacy → Camera** |
| `Missing packages` | `.\kiosk_venv\Scripts\Activate.ps1; pip install -r requirements.txt` |
| `Models not found` | `python setup_models.py` |
| ROCm wheels can't install | Override the index URL: `$env:TORCH_INDEX="https://download.pytorch.org/whl/nightly/rocm6.4"; .\bootstrap.ps1` |
| Need a totally clean reinstall | Delete `kiosk_venv\` and re-run `.\bootstrap.ps1` |

## Backend notes

The R9700 (gfx1201) is officially supported by ROCm 6.4+ on Windows, so no
`HSA_OVERRIDE_GFX_VERSION` is needed. If the ROCm-Windows PyTorch wheels can't
be installed (network issue, AMD changed the index URL, etc.), the bootstrap
falls back to `torch-directml` which still runs on the R9700 via DirectX 12 —
slower, but reliable.

To force a different PyTorch wheel index, set `$env:TORCH_INDEX` before
running `bootstrap.ps1`.
