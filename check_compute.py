#!/usr/bin/env python3
"""Test GPU compute with HSA override for gfx1151 (Strix Halo)."""
import os
import sys

# gfx1151 needs to map to a supported target
# Try 11.0.0 first, then 11.5.1 (native)
overrides = ["11.5.1", "11.0.0", "11.0.1", "11.0.2"]

for ver in overrides:
    print(f"\nTrying HSA_OVERRIDE_GFX_VERSION={ver}...")
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = ver

    # Fork a subprocess so a hang doesn't block us
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; x = torch.zeros(512,512).cuda(); print(f'Sum: {x.sum().item()}'); print('SUCCESS')"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "HSA_OVERRIDE_GFX_VERSION": ver}
    )

    if "SUCCESS" in result.stdout:
        print(result.stdout.strip())
        print(f"\nWORKING override: HSA_OVERRIDE_GFX_VERSION={ver}")
        sys.exit(0)
    else:
        stderr = result.stderr.strip()[-200:] if result.stderr else ""
        print(f"  Failed: {stderr}")

print("\nNo working override found. This GPU may need a newer PyTorch/ROCm version.")
