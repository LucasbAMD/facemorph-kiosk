#!/usr/bin/env python3
"""Fix PyTorch for Strix Halo (gfx1151).

The standard PyTorch nightly doesn't support gfx1151.
This installs AMD's TheRock wheel built specifically for Strix Halo.
"""
import subprocess
import sys

print("=" * 55)
print("  Fixing PyTorch for Strix Halo (gfx1151)")
print("=" * 55)

print("\n[1/2] Uninstalling broken PyTorch...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                "torch", "torchvision", "torchaudio"])

print("\n[2/2] Installing AMD TheRock PyTorch for gfx1151...")
print("      This may take a few minutes...")
result = subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--index-url", "https://rocm.nightlies.amd.com/v2/gfx1151/",
    "torch",
])

if result.returncode != 0:
    print("\n[ERR] Install failed. Try manually:")
    print("  pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch")
    sys.exit(1)

print("\n[OK] Testing GPU compute...")
import importlib
import torch
importlib.reload(torch)

print(f"  PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    x = torch.zeros(512, 512).cuda()
    print(f"  Compute test: {x.sum().item()}")
    print("\n  GPU compute works! Run: python start.py")
else:
    print("\n  [WARN] CUDA not available after install")
    print("  Try rebooting and running start.py")
