# ----------------------------------------------------------------------
#  bootstrap.ps1 -- One-command Windows setup for AI Scene Style Kiosk
#
#  Usage (in PowerShell, from the repo directory):
#      .\bootstrap.ps1
#
#  If you get an execution-policy error, run once:
#      Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#
#  What it does:
#    1. Verifies Python 3.10/3.11 is installed
#    2. Verifies the AMD HIP SDK (ROCm for Windows) is installed
#    3. Creates a Python venv (kiosk_venv\)
#    4. Installs PyTorch with ROCm-Windows wheels
#       (falls back to torch-directml if ROCm wheels aren't available)
#    5. Installs all Python dependencies
#    6. Downloads all AI models (~15 GB)
#
#  Requirements:
#    - Windows 11
#    - AMD Radeon AI PRO R9700 (or any RDNA 4 / RDNA 3 GPU)
#    - Latest Adrenalin driver
#    - AMD HIP SDK 6.4+  (https://www.amd.com/en/developer/resources/rocm-hub.html)
#    - Python 3.10 or 3.11 (https://www.python.org/downloads/)
#    - ~20 GB free disk space
# ----------------------------------------------------------------------

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir   = Join-Path $ScriptDir "kiosk_venv"
$PyExe     = Join-Path $VenvDir "Scripts\python.exe"
$PipExe    = Join-Path $VenvDir "Scripts\pip.exe"
$Activate  = Join-Path $VenvDir "Scripts\Activate.ps1"

# -- PyTorch backend selection -------------------------------------------------
# This kiosk must run on a range of machines, importantly including AMD APUs
# with NO discrete GPU (e.g. a Ryzen AI Max "Strix Halo" dev box). Native
# ROCm-on-Windows wheels only cover specific dGPUs (the R9700/gfx1201 is the
# main one) and are unreliable on APUs, so the DEFAULT, most-portable backend
# here is DirectML, which runs the model on any DirectX 12 GPU/iGPU.
#
# Choose with $env:KIOSK_BACKEND before running:
#   directml  (default) -- torch + torch-directml. Works on APUs and any DX12 GPU.
#   rocm                -- native ROCm-Windows wheels (only for supported dGPUs
#                          like the R9700). Set $env:TORCH_INDEX to the AMD index,
#                          e.g. https://repo.radeon.com/rocm/windows/...  or a
#                          TheRock gfx120X index. The pytorch.org rocm indexes are
#                          LINUX-ONLY and will not install on Windows.
$Backend = if ($env:KIOSK_BACKEND) { $env:KIOSK_BACKEND.ToLower() } else { "directml" }

# Only used when $Backend = "rocm". No safe universal Windows default exists,
# so this must be supplied by the user for their specific GPU.
$TorchIndex = $env:TORCH_INDEX

function Fail($msg) {
    Write-Host ""
    Write-Host "  [ERROR] $msg" -ForegroundColor Red
    Write-Host "  Bootstrap failed. Fix the issue above and re-run: .\bootstrap.ps1"
    Write-Host ""
    exit 1
}

function Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
}

function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  AI Scene Style Kiosk -- Windows Bootstrap" -ForegroundColor Cyan
Write-Host "  Target GPU: AMD Radeon AI PRO R9700 (RDNA 4 / gfx1201)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# -- 1. Python check ---------------------------------------------------------
Step "1/6" "Checking Python install..."

$Python = $null
foreach ($candidate in @("py -3.11", "py -3.10", "python")) {
    try {
        $verRaw = & cmd /c "$candidate --version 2>&1"
        if ($LASTEXITCODE -eq 0 -and $verRaw -match "Python\s+3\.(1[0-2])") {
            $Python = $candidate
            Ok "Found $verRaw (using '$candidate')"
            break
        }
    } catch { }
}

if (-not $Python) {
    Fail "Python 3.10, 3.11, or 3.12 not found. Install from https://www.python.org/downloads/ and check 'Add to PATH'."
}

# -- 2. HIP SDK check --------------------------------------------------------
Step "2/6" "Checking AMD HIP SDK (ROCm for Windows)..."

$HipPath  = $env:HIP_PATH
$RocmRoot = $null
if ($HipPath -and (Test-Path $HipPath)) {
    $RocmRoot = $HipPath
} else {
    foreach ($p in @("C:\Program Files\AMD\ROCm\6.4", "C:\Program Files\AMD\ROCm\6.3", "C:\Program Files\AMD\ROCm\6.2")) {
        if (Test-Path $p) { $RocmRoot = $p; break }
    }
}

if ($RocmRoot) {
    Ok "HIP SDK detected at $RocmRoot"
    if (-not $env:HIP_PATH) { $env:HIP_PATH = $RocmRoot }
} else {
    Warn "HIP SDK not detected."
    Write-Host "         The R9700 needs the AMD HIP SDK 6.4+ for native PyTorch ROCm acceleration."
    Write-Host "         Download:  https://www.amd.com/en/developer/resources/rocm-hub.html"
    Write-Host ""
    Write-Host "         Continuing anyway -- will fall back to torch-directml if ROCm wheels"
    Write-Host "         can't load. This is slower but will still run on the R9700 via DirectX 12."
}

# -- 3. Create venv ----------------------------------------------------------
Step "3/6" "Creating Python virtual environment..."

if (Test-Path $VenvDir) {
    Ok "venv already exists at $VenvDir"
} else {
    & cmd /c "$Python -m venv `"$VenvDir`""
    if ($LASTEXITCODE -ne 0) { Fail "Failed to create venv at $VenvDir" }
    Ok "Created venv at $VenvDir"
}

if (-not (Test-Path $PyExe)) { Fail "venv python not found at $PyExe" }

& $PyExe -m pip install --upgrade pip setuptools wheel -q
if ($LASTEXITCODE -ne 0) { Fail "Failed to upgrade pip/setuptools" }

# -- 4. PyTorch --------------------------------------------------------------
Step "4/6" "Installing PyTorch (backend: $Backend)..."

$TorchOk = $false

if ($Backend -eq "rocm") {
    # Native ROCm-on-Windows path -- only for supported dGPUs (e.g. R9700).
    if (-not $TorchIndex) {
        Warn 'KIOSK_BACKEND=rocm requires $env:TORCH_INDEX to point at an AMD Windows wheel index.'
        Warn 'The pytorch.org rocm indexes are Linux-only. For a supported AMD dGPU use either:'
        Warn '  - AMD official:  https://repo.radeon.com/rocm/windows/  (see AMD install-pytorch docs)'
        Warn '  - TheRock gfx120X nightly index for gfx1200/gfx1201'
        Warn 'Then set $env:TORCH_INDEX and re-run bootstrap.ps1, or run without KIOSK_BACKEND for DirectML.'
        Fail 'Missing $env:TORCH_INDEX for KIOSK_BACKEND=rocm.'
    }
    Write-Host "  Installing ROCm-Windows wheels from: $TorchIndex"
    & $PipExe install --pre torch torchvision torchaudio --index-url $TorchIndex
    if ($LASTEXITCODE -eq 0) {
        & $PyExe -c "import torch; assert torch.cuda.is_available()" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $name = & $PyExe -c "import torch; print(torch.cuda.get_device_name(0))"
            Ok "PyTorch ROCm-Windows installed -- GPU: $name"
            $TorchOk = $true
        } else {
            Warn "ROCm wheels installed but torch.cuda.is_available() is False."
            Warn "Falling back to DirectML so the kiosk still runs."
            & $PipExe uninstall -y torch torchvision torchaudio 2>$null | Out-Null
        }
    } else {
        Warn "ROCm-Windows wheels could not be installed from $TorchIndex. Falling back to DirectML."
    }
}

# Default / fallback: torch-directml (runs on any DX12 GPU or AMD APU iGPU).
if (-not $TorchOk) {
    Write-Host ""
    Write-Host "  Installing torch-directml (portable DirectML backend)..."
    # torch-directml pins compatible torch/torchvision versions itself.
    & $PipExe install torch-directml
    if ($LASTEXITCODE -ne 0) { Fail "Failed to install torch-directml" }
    # Verify DirectML actually sees a device before continuing.
    & $PyExe -c "import torch_directml as d; assert d.device_count() > 0" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $dname = & $PyExe -c "import torch_directml as d; print(d.device_name(0))" 2>$null
        Ok "torch-directml installed -- device: $dname"
    } else {
        Warn "torch-directml installed but reports 0 devices. The kiosk will run on CPU (slow)."
        Warn "Check that the latest AMD Adrenalin driver is installed."
    }
    $TorchOk = $true
}

# -- 5. Python dependencies --------------------------------------------------
Step "5/6" "Installing Python dependencies..."

& $PipExe install -r (Join-Path $ScriptDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "Failed to install requirements.txt" }
& $PipExe install huggingface-hub
if ($LASTEXITCODE -ne 0) { Fail "Failed to install huggingface-hub" }

# InsightFace (IP-Adapter FaceID) runs on ONNXRuntime, separate from torch.
# On the DirectML backend, install onnxruntime-directml so face detection can
# use the GPU too; otherwise it falls back to CPU automatically.
if (-not ($Backend -eq "rocm" -and $TorchOk)) {
    Write-Host "  Installing onnxruntime-directml for GPU face detection..."
    & $PipExe install onnxruntime-directml 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "onnxruntime-directml installed" }
    else { Warn "onnxruntime-directml not installed -- InsightFace will use CPU." }
}
Ok "All Python packages installed"

# -- 6. Download models ------------------------------------------------------
Step "6/6" "Downloading AI models (~15 GB, may take a while)..."
Write-Host "       Models cached to %USERPROFILE%\.cache\huggingface\ and %USERPROFILE%\kiosk_models\"

& $PyExe (Join-Path $ScriptDir "setup_models.py")
if ($LASTEXITCODE -ne 0) { Fail "Model download failed. Check internet and disk space." }

# -- Done -------------------------------------------------------------------
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  To start the kiosk:"
Write-Host "      .\start.ps1"
Write-Host ""
Write-Host "  Or manually:"
Write-Host "      .\kiosk_venv\Scripts\Activate.ps1"
Write-Host "      python start.py"
Write-Host ""
Write-Host "  Then open http://localhost:8000 in a browser."
Write-Host "=======================================================" -ForegroundColor Green
Write-Host ""