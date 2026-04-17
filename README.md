# AMD AI Scene Style Kiosk — Strix Halo APU Edition

AI-powered photo booth that transforms live camera feeds into stylized artwork using Stable Diffusion XL, ControlNet, and IP-Adapter FaceID — running locally on AMD Strix Halo APUs with ROCm.

## Requirements

- **OS:** Ubuntu 24.04 or Linux Mint 22.x
- **Hardware:** AMD Strix Halo APU (gfx1150 or gfx1151)
  - Tested: GMKtec box (Radeon 8060S, 96 GB), HP ZBook Ultra G1a (Radeon 8050S, 48 GB)
- **Disk:** ~20 GB free (for AI models)
- **Camera:** USB webcam
- **Internet:** Required for first-time setup
- **Power:** Keep plugged in during generation (battery mode throttles the GPU)

> **For discrete AMD GPUs** (RX 7000 series etc), use the `main` branch instead.

## Quick Setup

```bash
git clone -b claude/strix-halo-apu-yZRkP https://github.com/LucasbAMD/facemorph-kiosk.git && cd facemorph-kiosk
bash bootstrap.sh
```

The bootstrap script handles everything automatically:
1. Installs system packages
2. Installs ROCm drivers with `--no-dkms` (required for APU)
3. Detects gfx1150/gfx1151 and applies Strix Halo fixes:
   - Removes broken `amdgpu-dkms-firmware` package (MES 0x83 bug)
   - Downloads latest GPU firmware from kernel.org
   - Adds `amdgpu.cwsr_enable=0` kernel parameter to GRUB
   - Rebuilds initramfs
4. Creates Python virtual environment
5. Installs TheRock PyTorch wheel (gfx1150/gfx1151-specific build)
6. Installs all Python dependencies
7. Downloads AI models (~15 GB)
8. Sets GPU device permissions

> **Reboot required** after first run — the script will tell you. After rebooting, you do NOT need to re-run `bootstrap.sh`.

## Starting the Kiosk

```bash
cd facemorph-kiosk
source kiosk_venv/bin/activate
python start.py
```

Then open **http://localhost:8000** in a browser.

> **First generation is slow** (~3-5 min) because GPU kernels are being compiled and cached. Subsequent generations take ~45-90 sec depending on your hardware.

> **Laptop users:** Keep the charger plugged in and set power mode to Performance. Close unused apps. Don't close the lid during generation.

## What start.py Does Automatically

- Detects gfx1150/gfx1151 APU architecture
- Sets `HSA_ENABLE_SDMA=0` (fixes compute hangs on unified memory)
- Sets `HSA_XNACK=1` (required for unified memory APUs)
- Sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (enables attention kernels)
- Sets memory allocation config for APU
- Sets GPU device permissions

## Setup Instructions (for someone else)

> Give these instructions to anyone setting up the kiosk on a fresh Strix Halo machine:
>
> 1. Plug in a USB webcam
> 2. Open a terminal and run:
>    ```
>    git clone -b claude/strix-halo-apu-yZRkP https://github.com/LucasbAMD/facemorph-kiosk.git
>    cd facemorph-kiosk
>    bash bootstrap.sh
>    ```
> 3. Wait for setup (~15-30 min depending on internet speed)
> 4. **Reboot** when prompted: `sudo reboot`
> 5. Start the kiosk:
>    ```
>    cd facemorph-kiosk
>    source kiosk_venv/bin/activate
>    python start.py
>    ```
> 6. Open **http://localhost:8000** in Chrome or Firefox
> 7. Pick a style and click **Generate**
>
> **To restart later** (e.g. after a reboot):
> ```
> cd facemorph-kiosk
> source kiosk_venv/bin/activate
> python start.py
> ```

## Project Structure

```
facemorph-kiosk/
├── bootstrap.sh        # One-command setup (handles all Strix Halo fixes)
├── start.py            # App launcher (auto-configures ROCm env vars)
├── main.py             # FastAPI web server
├── generator.py        # AI generation pipeline (SDXL + ControlNet)
├── face_processor.py   # Face detection & recognition
├── setup_models.py     # Model downloader
├── requirements.txt    # Python dependencies
├── index.html          # Web UI
├── check_system.sh     # Diagnostic script (run with sudo)
├── fix_firmware.sh     # Manual firmware fix (if bootstrap missed it)
├── fix_torch.sh        # Manual PyTorch fix (reinstalls TheRock wheel)
└── assets/fonts/       # Nunito font files
```

## Troubleshooting

Run the diagnostic first:
```bash
sudo bash check_system.sh
```

| Problem | Fix |
|---------|-----|
| GPU compute hangs | Check MES firmware isn't 0x83: `sudo bash check_system.sh`. Run `bash fix_firmware.sh` then reboot |
| `No HIP GPUs found` | Reboot if ROCm was just installed. Check permissions: `sudo chmod 666 /dev/kfd /dev/dri/renderD*` |
| Wrong PyTorch wheel | Run `bash fix_torch.sh` to reinstall TheRock gfx1151 wheel |
| Generation stuck at 0% | Make sure `HSA_XNACK=1` is set (start.py does this automatically) |
| Generation stuck at 100% | VAE decode step is slow on first run (kernel compilation). Wait 2-3 min. Fixed in this branch (fp32 VAE + tiling) |
| PC freezes during generation | Normal on APU — CPU/GPU share resources. Freezes are brief |
| Slow generation on laptop | Plug in power + set to performance mode. Laptops thermal-throttle under sustained GPU load |
| Model downloads fail with DNS error | Firefox uses its own DNS. Fix terminal DNS: `echo "nameserver 8.8.8.8" \| sudo tee /etc/resolv.conf` |
| System freezes / suspend won't wake | Disable suspend: `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target` |
| `No camera found` | Check `ls /dev/video*` — plug in a USB webcam |
| `Missing packages` | `source kiosk_venv/bin/activate && pip install -r requirements.txt` |
| `Models not found` | Run `python setup_models.py` to re-download |

## Strix Halo Technical Notes

- **PyTorch source:** TheRock wheels from `rocm.nightlies.amd.com/v2/gfx1150/` or `/gfx1151/` (auto-detected) — standard PyTorch wheels do NOT work
- **No HSA_OVERRIDE_GFX_VERSION:** TheRock wheels have native gfx1150/1151 support, overriding to 11.0.0 causes hangs
- **HSA_ENABLE_SDMA=0:** Critical for APUs — the SDMA engine is buggy on unified memory
- **HSA_XNACK=1:** Required for unified memory architecture
- **VAE in fp32:** The fp16 VAE decoder hangs on gfx1151, so it's upcast to float32 with tiling
- **Resolution:** 768x768 (reduced from 1024x1024 for APU compatibility)
- **No Real-ESRGAN upscaler:** Disabled on APU to avoid additional GPU hangs
- **First run is slow:** MIOpen compiles GPU kernels on first generation (~3-5 min), cached after that
