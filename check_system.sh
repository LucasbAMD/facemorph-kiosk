#!/usr/bin/env bash
# Comprehensive diagnostic for Strix Halo (gfx1151) GPU compute
echo "======================================================="
echo "  Strix Halo GPU Diagnostic"
echo "======================================================="
echo ""

echo "=== Kernel ==="
uname -r
echo ""

echo "=== Boot Parameters ==="
cat /proc/cmdline
echo ""

echo "=== GPU Architecture ==="
for node in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -f "$node" ] || continue
    ver=$(grep "^gfx_target_version" "$node" 2>/dev/null | awk '{print $2}')
    [ -n "$ver" ] && [ "$ver" != "0" ] && echo "gfx_target_version: $ver (node: $node)"
done
echo ""

echo "=== ROCm Version ==="
cat /opt/rocm/.info/version 2>/dev/null || echo "Not found"
echo ""

echo "=== MES Firmware ==="
if [ -r /sys/kernel/debug/dri/1/amdgpu_firmware_info ]; then
    grep -i "MES" /sys/kernel/debug/dri/1/amdgpu_firmware_info 2>/dev/null
else
    echo "debugfs not readable (try: sudo bash check_system.sh)"
fi
echo ""

echo "=== amdgpu-dkms-firmware ==="
if dpkg -l amdgpu-dkms-firmware 2>/dev/null | grep -q '^ii'; then
    echo "INSTALLED (this can cause hangs — remove with: sudo apt autoremove --purge amdgpu-dkms-firmware)"
else
    echo "Not installed (good)"
fi
echo ""

echo "=== /dev/kfd Permissions ==="
ls -la /dev/kfd 2>&1
echo ""

echo "=== User Groups ==="
groups
echo ""

echo "=== PyTorch Version ==="
if [ -d kiosk_venv ]; then
    source kiosk_venv/bin/activate 2>/dev/null
fi
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'HIP version: {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
" 2>/dev/null || echo "PyTorch not installed or not working"
echo ""

echo "=== dmesg GPU Errors (last 10) ==="
dmesg 2>/dev/null | grep -iE "page.fault|PROTECTION_FAULT|MES failed|amdgpu.*error" | tail -10
if [ $? -ne 0 ]; then
    echo "(no errors found, or need sudo)"
fi
echo ""

echo "=== Quick GPU Compute Test ==="
echo "(30s timeout, HSA_ENABLE_SDMA=0, no HSA_OVERRIDE)"
export HSA_ENABLE_SDMA=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
unset HSA_OVERRIDE_GFX_VERSION 2>/dev/null
timeout 30 python3 -c "
import torch
if not torch.cuda.is_available():
    print('FAIL: No HIP GPUs found')
else:
    x = torch.zeros(512,512).cuda()
    print(f'Sum: {x.sum().item()}')
    print('SUCCESS - GPU compute works!')
" 2>&1
if [ $? -ne 0 ]; then
    echo ""
    echo "FAIL: GPU compute hung or crashed."
    echo ""
    echo "Most likely causes:"
    echo "  1. MES firmware 0x83 (check above)"
    echo "  2. amdgpu-dkms-firmware installed (check above)"
    echo "  3. Missing amdgpu.cwsr_enable=0 (check boot params above)"
    echo "  4. Wrong PyTorch wheel (need TheRock gfx1151 build)"
fi
echo ""
echo "======================================================="
