# AMD AI Scene Style Kiosk — Dev Box build (`devbox`)

AI-powered photo booth that turns a live camera feed into stylized artwork
using Stable Diffusion XL, ControlNet, and IP-Adapter FaceID — running fully
**locally** on Windows. After each transform, the visitor can **scan a QR code
to save the image to their phone**.

This `devbox` branch is built to run on a **vanilla Windows machine with no
discrete GPU** — for example an **AMD Ryzen AI dev box** (Strix Halo APU). It
auto-detects the best available graphics backend, so the same build also runs on
an AMD Radeon AI PRO R9700 or any DirectX 12 GPU, and falls back to CPU if
needed.

> **Which branch?** `devbox` = portable Windows build (this one, for the demo).
> `main` = original Ubuntu + ROCm build. Other branches target specific GPUs.

---

## TL;DR — fastest path on a fresh dev box

```powershell
# 1. Install Git and Python (one time) — see Step 1 below if not installed
# 2. Get the code
git clone https://github.com/LucasbAMD/facemorph-kiosk.git
cd facemorph-kiosk
git checkout devbox

# 3. Allow local scripts to run (one time)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 4. Install everything (PyTorch + dependencies + ~15 GB of AI models)
.\bootstrap.ps1

# 5. Run it
.\start.ps1
```

Then open **http://localhost:8000** in a browser. First launch downloads the AI
models and loads them into memory, so give it a few minutes before the first
transform.

---

## What you need

- **Windows 11** (Windows 10 works too)
- **Any AMD GPU or APU** with a current **Adrenalin driver** (the Ryzen AI dev
  box's built-in graphics is fine — no separate graphics card required)
- **Python 3.10, 3.11, or 3.12** with "Add Python to PATH" checked at install
- **A USB webcam**
- **~20 GB free disk space** (for the AI models)
- **Internet** for the one-time setup (model downloads). The booth itself runs
  offline after that.

> You do **not** need the AMD HIP SDK for the dev box. This build defaults to
> the **DirectML** backend, which runs on any DirectX 12 device through Windows.
> The HIP SDK is only relevant if you opt into native ROCm on a supported card
> like the R9700 (see "Advanced: native ROCm" below).

---

## Step-by-step first-time setup (nothing installed yet)

### Step 1 — Install Git and Python

Open the **Start menu**, type `PowerShell`, open **Windows PowerShell**, and run:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
```

**Close PowerShell and open a new window** (newly installed tools only appear in
a fresh window). Confirm both work:

```powershell
git --version
python --version
```

If either isn't recognized, install it manually: Git from
<https://git-scm.com/download/win>, Python from
<https://www.python.org/downloads/> (tick **Add Python to PATH**).

### Step 2 — Tell Git who you are (first time on this machine)

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Step 3 — Download the code

```powershell
cd $HOME\Documents
git clone https://github.com/LucasbAMD/facemorph-kiosk.git
cd facemorph-kiosk
git checkout devbox
```

The first time you reach GitHub, a browser window opens to sign in and authorize
access — do that once and Windows remembers it.

### Step 4 — Allow local scripts (one time)

Windows blocks unsigned scripts by default. Allow them for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Step 5 — Install everything

```powershell
.\bootstrap.ps1
```

This creates a private Python environment (`kiosk_venv\`), installs PyTorch
(DirectML by default) and all dependencies, then downloads ~15 GB of AI models.
It can take a while on the first run — that's normal.

### Step 6 — Run the kiosk

```powershell
.\start.ps1
```

Open **http://localhost:8000**. Hold still in front of the camera, pick a style,
and let it transform. When the result appears, a QR code shows next to it.

---

## The QR "save to your phone" feature

After a transform, the kiosk displays a QR code. A visitor scans it with their
phone camera, which opens a simple page showing their image with a **"Save to my
phone"** button.

**How phones reach the kiosk:** the QR encodes the kiosk's address on the local
network (e.g. `http://192.168.1.50:8000/r/<id>`). For this to work, **the
visitor's phone and the kiosk must be on the same Wi-Fi network** (the venue /
booth Wi-Fi). No internet, cloud account, or app is required — the image is
served directly by the kiosk.

The kiosk auto-detects its own network address. If it guesses wrong (for example
the machine has several network adapters), set it manually before launching:

```powershell
$env:KIOSK_PUBLIC_HOST = "192.168.1.50"   # the kiosk's IP on the venue Wi-Fi
.\start.ps1
```

To find the kiosk's IP, run `ipconfig` and look for the IPv4 address on the
active Wi-Fi adapter.

---

## Tuning: speed vs. quality

Each transform is tuned to land in roughly a **15-45 second** window. The build
auto-picks a profile based on the detected hardware — a strong GPU gets the full
quality pipeline, an APU gets a faster one. To override it, set `KIOSK_PROFILE`
before launching:

```powershell
$env:KIOSK_PROFILE = "fast"       # fastest, lighter output (good for a busy booth)
$env:KIOSK_PROFILE = "balanced"   # middle ground
$env:KIOSK_PROFILE = "quality"    # best output, slowest
.\start.ps1
```

If transforms are too slow on the dev box, use `fast`. If you have time to
spare and want nicer images, try `balanced` or `quality`.

---

## Environment variables (all optional)

Set any of these in PowerShell before `.\start.ps1` (or `.\bootstrap.ps1` for
the backend one):

| Variable | What it does | Default |
|---|---|---|
| `KIOSK_PROFILE` | `fast` / `balanced` / `quality` | auto by hardware |
| `KIOSK_DEVICE` | Force `dml` / `cuda` / `cpu` | auto-detect |
| `KIOSK_PUBLIC_HOST` | IP/hostname put in the QR link | auto LAN IP |
| `KIOSK_PORT` | Port the kiosk + QR link use | `8000` |
| `KIOSK_RESULT_TTL` | Seconds a photo stays downloadable | `1800` (30 min) |
| `KIOSK_BACKEND` | (bootstrap only) `directml` or `rocm` | `directml` |

---

## Checking it works

Confirm the graphics backend is alive:

```powershell
.\kiosk_venv\Scripts\Activate.ps1
python check_gpu.py
```

Once the kiosk is running, you can also open **http://localhost:8000/health** in
a browser — it reports the active device (e.g. `dml`) and profile.

---

## Advanced: native ROCm on a supported card (e.g. R9700)

The dev box should just use the default DirectML backend. But if you're running
on an **AMD Radeon AI PRO R9700** and want native ROCm performance:

1. Install the **AMD HIP SDK** and latest Adrenalin driver.
2. Point the bootstrap at an AMD **Windows** PyTorch wheel index and opt in:

```powershell
$env:KIOSK_BACKEND = "rocm"
$env:TORCH_INDEX   = "<AMD Windows wheel index for your card>"
.\bootstrap.ps1
```

Use AMD's `repo.radeon.com/rocm/windows/` wheels or a TheRock `gfx120X` index.
> Note: the `download.pytorch.org` ROCm indexes are **Linux-only** and will not
> install on Windows. If the ROCm wheels can't load, the bootstrap automatically
> falls back to DirectML so the kiosk still runs.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Set-ExecutionPolicy` / "scripts disabled" error | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once |
| `git` or `python` "not recognized" | Close and reopen PowerShell after installing; if still missing, reinstall and ensure "Add to PATH" |
| No GPU detected (runs slowly on CPU) | Update the AMD Adrenalin driver, reboot, re-run `.\bootstrap.ps1` |
| `No camera found` | Plug in a USB webcam; check Windows **Settings → Privacy → Camera** |
| QR scan doesn't open the page | Make sure the phone is on the **same Wi-Fi** as the kiosk; set `KIOSK_PUBLIC_HOST` to the kiosk's IP |
| Transforms too slow | `$env:KIOSK_PROFILE="fast"` before `.\start.ps1` |
| Out-of-memory on first transform | Use `fast` profile; close other GPU apps |
| `Missing packages` | `.\kiosk_venv\Scripts\Activate.ps1; pip install -r requirements.txt` |
| `Models not found` | `python setup_models.py` |
| Clean reinstall | Delete the `kiosk_venv\` folder and re-run `.\bootstrap.ps1` |

---

## Project structure

```
facemorph-kiosk/
├── bootstrap.ps1       # Windows setup (PowerShell) — installs everything
├── start.ps1 / .bat    # Windows launchers
├── start.py            # Cross-platform app launcher
├── main.py             # Web server + QR result delivery
├── generator.py        # AI pipeline (auto-detects GPU backend)
├── face_processor.py   # Face detection & recognition
├── setup_models.py     # Model downloader
├── check_gpu.py        # GPU diagnostic
├── requirements.txt    # Python dependencies
├── index.html          # Kiosk web UI
└── assets/fonts/       # Fonts
```
