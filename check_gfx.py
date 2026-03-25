#!/usr/bin/env python3
"""Detect GPU GFX version from sysfs."""
import os
import glob

kfd = "/sys/class/kfd/kfd/topology/nodes"
if not os.path.isdir(kfd):
    print("No KFD nodes found - ROCm driver not loaded")
else:
    for props in sorted(glob.glob(os.path.join(kfd, "*/properties"))):
        with open(props) as f:
            for line in f:
                if line.startswith("gfx_target_version"):
                    val = int(line.split()[1])
                    if val == 0:
                        continue
                    major = val // 10000
                    minor = (val % 10000) // 100
                    patch = val % 100
                    print(f"Raw value: {val}")
                    print(f"Decoded: gfx{major}{minor}{patch:02d}")
                    print(f"Major.Minor.Patch: {major}.{minor}.{patch}")
                    print(f"Node: {props}")
