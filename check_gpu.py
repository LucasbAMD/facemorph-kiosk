#!/usr/bin/env python3
"""Quick GPU check."""
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    x = torch.zeros(1).cuda()
    print(f"Tensor on GPU: {x.device} -- GPU is working")
else:
    print("GPU NOT available -- check ROCm drivers")
