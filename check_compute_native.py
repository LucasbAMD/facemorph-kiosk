#!/usr/bin/env python3
"""Test GPU compute WITHOUT any HSA override.
The TheRock gfx1151 wheel might work natively."""
import subprocess
import sys
import os

# Remove any existing override
env = {k: v for k, v in os.environ.items() if k != "HSA_OVERRIDE_GFX_VERSION"}

print("Testing GPU compute with NO HSA_OVERRIDE_GFX_VERSION...")
try:
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(f'PyTorch: {torch.__version__}'); "
         "x = torch.zeros(512,512).cuda(); print(f'Sum: {x.sum().item()}'); "
         "print('SUCCESS - GPU works natively!')"],
        capture_output=True, text=True, timeout=30, env=env
    )
    print(result.stdout.strip())
    if result.stderr:
        # Only print errors, not warnings
        for line in result.stderr.strip().split('\n'):
            if 'error' in line.lower() or 'fault' in line.lower():
                print(line)
    if "SUCCESS" in result.stdout:
        print("\nNo override needed! The TheRock wheel works natively for gfx1151.")
    else:
        print(f"\nFailed (exit code {result.returncode})")
except subprocess.TimeoutExpired:
    print("Timed out after 30s — native mode not working either")
    print("\nThe TheRock wheel may not be installed correctly.")
    print("Check: python -c \"import torch; print(torch.__version__)\"")
