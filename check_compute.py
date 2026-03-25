#!/usr/bin/env python3
"""Test GPU compute with HSA override."""
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Testing GPU compute...")
x = torch.zeros(1024, 1024).cuda()
print(f"Sum: {x.sum()}")
print("GPU compute works!")
