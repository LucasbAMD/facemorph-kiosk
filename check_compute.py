#!/usr/bin/env python3
"""Test GPU compute with various configurations for gfx1151 (Strix Halo).

Tries multiple combinations of HSA_OVERRIDE_GFX_VERSION and HSA_ENABLE_SDMA
to find a working configuration.
"""
import os
import sys
import subprocess

# Test configurations: (description, env overrides)
configs = [
    ("No override + SDMA disabled (TheRock native)", {
        "HSA_ENABLE_SDMA": "0",
    }),
    ("HSA_OVERRIDE=11.5.1 + SDMA disabled", {
        "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
        "HSA_ENABLE_SDMA": "0",
    }),
    ("No override + SDMA enabled (TheRock native)", {
    }),
    ("HSA_OVERRIDE=11.0.0 + SDMA disabled", {
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
        "HSA_ENABLE_SDMA": "0",
    }),
]

test_code = (
    "import torch; "
    "print(f'PyTorch: {torch.__version__}'); "
    "print(f'CUDA: {torch.cuda.is_available()}'); "
    "x = torch.zeros(512,512).cuda(); "
    "print(f'Sum: {x.sum().item()}'); "
    "print('SUCCESS')"
)

for desc, overrides in configs:
    print(f"\n{'='*50}")
    print(f"Testing: {desc}")
    # Start from clean env (remove any existing HSA vars)
    env = {k: v for k, v in os.environ.items()
           if k not in ("HSA_OVERRIDE_GFX_VERSION", "HSA_ENABLE_SDMA")}
    env.update(overrides)
    env["GPU_MAX_HEAP_SIZE"] = "100"
    env["GPU_MAX_ALLOC_PERCENT"] = "100"

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True, text=True, timeout=30, env=env
        )
        if "SUCCESS" in result.stdout:
            print(result.stdout.strip())
            print(f"\nWORKING config: {desc}")
            for k, v in overrides.items():
                print(f"  {k}={v}")
            sys.exit(0)
        else:
            print(f"  Failed (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-3:]:
                    print(f"  {line}")
    except subprocess.TimeoutExpired:
        print(f"  Timed out after 30s")
    except Exception as e:
        print(f"  Error: {e}")

print("\nNo working configuration found.")
print("Make sure you have:")
print("  1. TheRock gfx1151 wheel: pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/")
print("  2. Kernel parameter: amdgpu.cwsr_enable=0")
print("  3. GPU permissions: sudo chmod 666 /dev/kfd /dev/dri/renderD*")
print("  4. Updated firmware: sudo apt install linux-firmware")
