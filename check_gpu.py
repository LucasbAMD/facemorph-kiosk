#!/usr/bin/env python3
"""Quick GPU diagnostic — works on Linux (ROCm) and Windows (HIP SDK / DirectML)."""
import platform
import sys

print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python:   {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch:  {torch.__version__}")
except ImportError:
    print("[ERR] PyTorch is not installed in this environment.")
    sys.exit(1)

print(f"CUDA/ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU:  {name}")
    print(f"VRAM: {vram:.1f} GB")
    x = torch.zeros(1).cuda()
    print(f"Tensor on GPU: {x.device} -- GPU is working")
    sys.exit(0)

# Windows DirectML fallback
try:
    import torch_directml
    dev = torch_directml.device()
    print(f"DirectML device: {dev}  (count={torch_directml.device_count()})")
    x = torch.zeros(1).to(dev)
    print(f"Tensor on DirectML: {x.device} -- DirectML is working")
    sys.exit(0)
except ImportError:
    pass

print("[ERR] No GPU backend detected.")
if platform.system() == "Windows":
    print("      Install AMD HIP SDK 6.4+ from https://www.amd.com/en/developer/resources/rocm-hub.html")
    print("      Or run: pip install torch-directml")
else:
    print("      Check ROCm drivers (rocminfo) and reboot if you just installed them.")
sys.exit(1)
