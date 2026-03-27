#!/usr/bin/env bash
echo "=== /dev/kfd ==="
ls -la /dev/kfd 2>&1

echo ""
echo "=== amdgpu module ==="
lsmod | grep amdgpu

echo ""
echo "=== dmesg amdgpu (last 15 lines) ==="
dmesg | grep -i amdgpu | tail -15

echo ""
echo "=== ROCm version ==="
cat /opt/rocm/.info/version 2>/dev/null || echo "ROCm version file not found"

echo ""
echo "=== rocminfo (first 30 lines) ==="
rocminfo 2>&1 | head -30
