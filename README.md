# AMD AI Scene Style Kiosk — Dev Box build (`devbox`)

AI-powered photo booth that turns a live webcam feed into stylized artwork
using Stable Diffusion XL (Juggernaut XL v9), ControlNet, and IP-Adapter FaceID,
running fully **locally** on Windows. After each transform, the visitor can
**scan a QR code to save the image to their phone**.

This `devbox` branch is built to run on a **Windows machine with an AMD APU
(integrated GPU)** — e.g. an **AMD Ryzen AI dev box** (Strix Halo). It
auto-detects the best graphics backend (DirectML on APUs, ROCm/CUDA on dGPUs,
CPU as a last resort) and auto-selects a speed profile to match.

> **Which branch?** `devbox` = portable Windows build (this one, for the demo).
> `main` = original Ubuntu + ROCm build.

---

## TL;DR — get it running

```powershell
# 1. Install Git + Python 3.12 (one time) — see Step 1 if not installed
# 2. Get the code
git clone https://github.com/LucasbAMD/facemorph-kiosk.git
cd facemorph-kiosk
git checkout devbox

# 3. Allow local scripts (one time)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 4. Install everything (PyTorch + deps + ~15 GB of AI models)
.\bootstrap.ps1

# 5. Allow phones to reach the kiosk (one time, as Administrator)
New-NetFirewallRule -DisplayName "FaceMorph Kiosk 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow

# 6. Run it
.\start.ps1
```

Then open **http://localhost:8000** on the kiosk machine. First launch loads the
models into memory, so give it a minute before the first transform.

---

## What you need

- **Windows 11** (Windows 10 works too)
- **AMD APU or GPU** with a current **Adrenalin driver** (the Ryzen AI dev box's
  built-in graphics is the target; a discrete AMD GPU also works)
- **Python 3.12** — see note below; **3.13 is NOT supported** (the AI libraries
  don't have compatible builds for it yet)
- **A USB or built-in webcam**
- **~20 GB free disk space** (for the AI models)
- **Internet** for one-time setup (downloads). The booth itself runs offline.

> **Important — Python version.** Use **Python 3.12** (3.10/3.11 also fine).
> Do NOT use 3.13: `torch-directml` and the pinned AI libraries have no 3.13
> wheels. If python.org only offers a source-only 3.12, get the last 3.12 that
> shipped an installer: <https://www.python.org/downloads/release/python-31210/>
> (Files table -> "Windows installer (64-bit)"), or run
> `winget install --id Python.Python.3.12 -e`.

> **You do NOT need the AMD HIP SDK** for the dev box. This build defaults to the
> DirectML backend, which runs on any DirectX 12 device. The HIP SDK is only for
> opting into native ROCm on a supported discrete card (see "Advanced").

---

## First-time setup, step by step (nothing installed yet)

### Step 1 — Install Git and Python 3.12

Open **Start -> type "PowerShell" -> Windows PowerShell**, then:

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.12 -e
```

**Close PowerShell and open a new window** (new tools only appear in a fresh
window). Confirm:

```powershell
git --version
py -3.12 --version
```

### Step 2 — Identify yourself to Git (first time on this machine)

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

First contact with GitHub opens a browser to sign in — do it once.

### Step 4 — Allow local scripts (one time)

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Step 5 — Install everything

```powershell
.\bootstrap.ps1
```

This creates a private Python env (`kiosk_venv\`), installs PyTorch (DirectML by
default) and all pinned dependencies, then downloads ~15 GB of models. The model
download is the slow part (10-30+ min). If it ever shows
`ERROR: No models found`, the model step didn't finish — just run it directly:

```powershell
.\kiosk_venv\Scripts\Activate.ps1
python setup_models.py
```

### Step 6 — Allow phones to reach the kiosk (one time)

The QR-to-phone feature needs inbound connections on port 8000. Windows blocks
these by default, which makes the QR page hang on phones. Open PowerShell **as
Administrator** and run:

```powershell
New-NetFirewallRule -DisplayName "FaceMorph Kiosk 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

### Step 7 — Run the kiosk

```powershell
.\start.ps1
```

Open **http://localhost:8000**. To stop it later, press `Ctrl+C` in PowerShell.

Day-to-day after the first install, you only need:

```powershell
cd $HOME\Documents\facemorph-kiosk
.\start.ps1
```

---

## The QR "save to your phone" feature

After a transform, the kiosk shows a QR code. A visitor scans it; their phone
opens a page with their image and a **"Save to my phone"** button.

**The phone and the kiosk must be on the same Wi-Fi network.** The QR encodes the
kiosk's LAN address (e.g. `http://192.168.1.42:8000/r/<id>`). No internet or
cloud is involved — the kiosk serves the image directly.

**If the QR page hangs / shows a white page**, it's almost always one of these
two things (both came up during testing — check them on the dev box too):

1. **Firewall** — did you run the Step 6 firewall rule (as Administrator)?
   Without it, phones can't reach the kiosk and the page hangs.

2. **Wrong IP in the QR** — if the machine has extra network adapters (VPN, WSL,
   Hyper-V, Docker), the kiosk may auto-pick the wrong address. Check it: open
   `http://localhost:8000/health` and look at `share_base_url`. If it is NOT your
   real Wi-Fi IP (e.g. it shows `172.x.x.x` from a VPN instead of `192.168.x.x`),
   force the correct one. Find your Wi-Fi IPv4 with `ipconfig` (under
   "Wireless LAN adapter Wi-Fi"), then restart with:

   ```powershell
   $env:KIOSK_PUBLIC_HOST="192.168.1.42"   # your real Wi-Fi IPv4
   .\start.ps1
   ```

   Re-open `/health` to confirm `share_base_url` now shows the right address,
   then generate a FRESH transform and scan the NEW QR. (Old QRs stop working
   after a restart — results are stored in memory and cleared on restart.)

---

## Speed vs. quality (profiles)

Each transform targets roughly **15-45 seconds**. The build auto-picks a profile
by hardware: a strong dGPU gets `quality` (full ControlNet + IP-Adapter), an APU
gets `fast` (a lightweight img2img pipeline — see note below). Override with:

```powershell
$env:KIOSK_PROFILE="fast"       # lightest, fastest
$env:KIOSK_PROFILE="balanced"   # middle
$env:KIOSK_PROFILE="quality"    # best, slowest
.\start.ps1
```

> **Why APUs use a lightweight mode.** The full pipeline (ControlNet + depth +
> IP-Adapter + CLIP vision, ~10 GB) can exhaust a shared-memory iGPU and trigger
> a DirectML device reset ("GPU will not respond to more commands"). The `fast`
> profile therefore runs Juggernaut as a plain img2img pipeline. Trade-off: it
> preserves less of the original pose/structure (e.g. a hand gesture may not
> carry through) and does not lock facial identity as tightly. If the dev box's
> iGPU has enough graphics memory, try `KIOSK_PROFILE=balanced` for ControlNet
> structure preservation. See "If the GPU runs out of memory" below.

---

## Environment variables (all optional)

Set before `.\start.ps1` (or `.\bootstrap.ps1` for `KIOSK_BACKEND`):

| Variable | What it does | Default |
|---|---|---|
| `KIOSK_PROFILE` | `fast` / `balanced` / `quality` | auto by hardware |
| `KIOSK_DEVICE` | Force `dml` / `cuda` / `cpu` | auto-detect |
| `KIOSK_PUBLIC_HOST` | IP/hostname put in the QR link | auto LAN IP |
| `KIOSK_PORT` | Port the kiosk + QR use | `8000` |
| `KIOSK_RESULT_TTL` | Seconds a photo stays downloadable | `1800` (30 min) |
| `KIOSK_BACKEND` | (bootstrap) `directml` or `rocm` | `directml` |

---

## Checking it works

```powershell
.\kiosk_venv\Scripts\Activate.ps1
python check_gpu.py
```

And once running, open **http://localhost:8000/health** — it reports the active
device (`dml`/`cuda`/`cpu`), the profile, and the `share_base_url` the QR uses.

---

## Differences on the dev box vs. a thin laptop

This was developed/tested first on a Zenbook (Radeon 890M), which has a small
iGPU. The Ryzen AI dev box has a much larger iGPU and far more memory it can give
to graphics, so:

- **Don't force CPU on the dev box.** Let it auto-detect DirectML. SDXL should
  load onto the GPU and run much faster than CPU.
- The firewall rule (Step 6) and the `KIOSK_PUBLIC_HOST` check still apply.
- If the GPU has the headroom, try `KIOSK_PROFILE=balanced` or `quality` for
  better structure/identity preservation than the `fast` img2img default.

---

## If the GPU runs out of memory (DirectML device reset)

Symptom: startup logs show `The GPU device instance has been suspended` or
`GPU will not respond to more commands`, and it falls back / fails to load.

The iGPU does not have enough graphics memory for the model. Options, best first:

1. **Increase Variable Graphics Memory (VGM).** In **AMD Adrenalin ->
   Performance/Gaming**, raise "Variable Graphics Memory" (or set the BIOS
   "UMA Frame Buffer Size") to 8-16 GB if you have the system RAM. This gives
   DirectML room to load SDXL. Best option — keeps full quality, reversible.
2. **Use the `fast` profile** (default on APUs) — lighter pipeline.
3. **Force CPU** as a guaranteed-but-slow fallback for testing:
   `$env:KIOSK_DEVICE="cpu"; .\start.ps1` (minutes per image — not for a booth).

---

## Advanced: native ROCm on a supported dGPU (e.g. R9700)

The dev box should use the default DirectML backend. For an **AMD Radeon AI PRO
R9700** wanting native ROCm:

```powershell
$env:KIOSK_BACKEND="rocm"
$env:TORCH_INDEX="<AMD Windows wheel index for your card>"
.\bootstrap.ps1
```

Use AMD's `repo.radeon.com/rocm/windows/` wheels or a TheRock `gfx120X` index.
> The `download.pytorch.org` ROCm indexes are **Linux-only** and won't install on
> Windows. If ROCm wheels can't load, the bootstrap falls back to DirectML.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Set-ExecutionPolicy` / "scripts disabled" | Run the Step 4 command once |
| `git` / `python` not recognized | Close & reopen PowerShell; reinstall with "Add to PATH" |
| Bootstrap parse errors / weird characters | Make sure you have the `devbox` versions of the scripts |
| Pip / import errors about `diffusers`, `transformers`, `PreTrainedModel`, "Parameter q has unsupported type" | Dependency mismatch — the `requirements.txt` pins must be installed: `.\kiosk_venv\Scripts\Activate.ps1; pip install -r requirements.txt` |
| `ERROR: No models found` | `python setup_models.py` (inside the venv) |
| "Waiting for camera..." but webcam works elsewhere | Check Windows camera privacy (Settings -> Privacy -> Camera, allow desktop apps); the app tries DirectShow/MSMF automatically |
| QR page hangs on phone | Firewall rule (Step 6) + check `share_base_url` at `/health`; set `KIOSK_PUBLIC_HOST` to the real Wi-Fi IP |
| GPU device suspended / reset | See "If the GPU runs out of memory" above |
| Transforms too slow | `$env:KIOSK_PROFILE="fast"`; on APUs increase VGM |
| Clean reinstall | Delete `kiosk_venv\` and re-run `.\bootstrap.ps1` |

---

## Project structure

```
facemorph-kiosk/
|-- bootstrap.ps1       # Windows setup (installs everything)
|-- start.ps1 / .bat    # Windows launchers
|-- start.py            # Cross-platform app launcher
|-- main.py             # Web server, camera, QR result delivery
|-- generator.py        # AI pipeline (auto device + profile selection)
|-- face_processor.py   # Face detection & recognition
|-- setup_models.py     # Model downloader
|-- check_gpu.py        # GPU diagnostic
|-- requirements.txt    # Pinned Python dependencies
|-- index.html          # Kiosk web UI
|-- assets/fonts/       # Fonts
```
