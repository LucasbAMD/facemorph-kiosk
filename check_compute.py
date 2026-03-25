#!/usr/bin/env python3
"""Test GPU compute with HSA override for gfx1151 (Strix Halo)."""
import os
import sys
import subprocess

overrides = ["11.5.1", "11.0.0", "11.0.1", "11.0.2"]

for ver in overrides:
    print(f"\nTrying HSA_OVERRIDE_GFX_VERSION={ver}...")
    env = {**os.environ, "HSA_OVERRIDE_GFX_VERSION": ver}
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; x = torch.zeros(512,512).cuda(); print(f'Sum: {x.sum().item()}'); print('SUCCESS')"],
            capture_output=True, text=True, timeout=30, env=env
        )
        if "SUCCESS" in result.stdout:
            print(result.stdout.strip())
            print(f"\nWORKING override: HSA_OVERRIDE_GFX_VERSION={ver}")
            sys.exit(0)
        else:
            print(f"  Failed (exit code {result.returncode})")
            if result.stderr:
                print(f"  {result.stderr.strip()[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"  Timed out after 30s — not working")
    except Exception as e:
        print(f"  Error: {e}")

print("\nNo working override found.")
print("This GPU may need a newer PyTorch/ROCm build.")
print("Try: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2")
