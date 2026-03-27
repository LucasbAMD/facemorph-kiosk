#!/usr/bin/env python3
"""Test GPU compute with HSA_ENABLE_SDMA=0 (critical for APUs).

The TheRock gfx1151 wheel works natively without HSA_OVERRIDE_GFX_VERSION,
but APUs need SDMA disabled to prevent compute hangs on unified memory.
"""
import subprocess
import sys
import os

# Clean env: no HSA override, but DO disable SDMA for APU stability
env = {k: v for k, v in os.environ.items() if k != "HSA_OVERRIDE_GFX_VERSION"}
env["HSA_ENABLE_SDMA"] = "0"
env["GPU_MAX_HEAP_SIZE"] = "100"
env["GPU_MAX_ALLOC_PERCENT"] = "100"

print("Testing GPU compute with HSA_ENABLE_SDMA=0 (no HSA_OVERRIDE)...")
try:
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(f'PyTorch: {torch.__version__}'); "
         "x = torch.zeros(512,512).cuda(); print(f'Sum: {x.sum().item()}'); "
         "print('SUCCESS - GPU works!')"],
        capture_output=True, text=True, timeout=30, env=env
    )
    print(result.stdout.strip())
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            if 'error' in line.lower() or 'fault' in line.lower():
                print(line)
    if "SUCCESS" in result.stdout:
        print("\nTheRock wheel works natively with SDMA disabled.")
    else:
        print(f"\nFailed (exit code {result.returncode})")
except subprocess.TimeoutExpired:
    print("Timed out after 30s.")
    print("\nCheck: amdgpu.cwsr_enable=0 in GRUB, firmware updated, /dev/kfd permissions")
