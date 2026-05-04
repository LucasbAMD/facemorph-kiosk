# ──────────────────────────────────────────────────────────────────────
#  bootstrap.ps1 — One-command Windows setup for AI Scene Style Kiosk
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
# ──────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir   = Join-Path $ScriptDir "kiosk_venv"
$PyExe     = Join-Path $VenvDir "Scripts\python.exe"
$PipExe    = Join-Path $VenvDir "Scripts\pip.exe"
$Activate  = Join-Path $VenvDir "Scripts\Activate.ps1"

# Default ROCm-Windows PyTorch wheel index. Override by setting $env:TORCH_INDEX
# before running this script if AMD publishes a newer index URL.
$DefaultTorchIndex = "https://download.pytorch.org/whl/nightly/rocm6.4"
$TorchIndex = if ($env:TORCH_INDEX) { $env:TORCH_INDEX } else { $DefaultTorchIndex }

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
Write-Host "  AI Scene Style Kiosk — Windows Bootstrap" -ForegroundColor Cyan
Write-Host "  Target GPU: AMD Radeon AI PRO R9700 (RDNA 4 / gfx1201)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# ── 1. Python check ─────────────────────────────────────────────────────────
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

# ── 2. HIP SDK check ────────────────────────────────────────────────────────
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
    Write-Host "         Continuing anyway — will fall back to torch-directml if ROCm wheels"
    Write-Host "         can't load. This is slower but will still run on the R9700 via DirectX 12."
}

# ── 3. Create venv ──────────────────────────────────────────────────────────
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

# ── 4. PyTorch ──────────────────────────────────────────────────────────────
Step "4/6" "Installing PyTorch..."

$TorchOk = $false

# First try: ROCm-Windows wheels
Write-Host "  Trying ROCm-Windows wheels: $TorchIndex"
& $PipExe install --pre torch torchvision torchaudio --index-url $TorchIndex
if ($LASTEXITCODE -eq 0) {
    & $PyExe -c "import torch; assert torch.cuda.is_available()" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $name = & $PyExe -c "import torch; print(torch.cuda.get_device_name(0))"
        Ok "PyTorch ROCm-Windows installed — GPU: $name"
        $TorchOk = $true
    } else {
        Warn "PyTorch ROCm wheels installed but GPU not detected. Falling back to DirectML."
        & $PipExe uninstall -y torch torchvision torchaudio 2>$null | Out-Null
    }
} else {
    Warn "ROCm-Windows wheels not available from $TorchIndex"
}

# Fallback: torch-directml (works on any DX12 GPU including R9700)
if (-not $TorchOk) {
    Write-Host ""
    Write-Host "  Falling back to torch-directml (DirectML backend)..."
    & $PipExe install torch torchvision torch-directml
    if ($LASTEXITCODE -ne 0) { Fail "Failed to install torch-directml" }
    Ok "torch-directml installed"
}

# ── 5. Python dependencies ──────────────────────────────────────────────────
Step "5/6" "Installing Python dependencies..."

& $PipExe install -r (Join-Path $ScriptDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "Failed to install requirements.txt" }
& $PipExe install huggingface-hub
if ($LASTEXITCODE -ne 0) { Fail "Failed to install huggingface-hub" }
Ok "All Python packages installed"

# ── 6. Download models ──────────────────────────────────────────────────────
Step "6/6" "Downloading AI models (~15 GB, may take a while)..."
Write-Host "       Models cached to %USERPROFILE%\.cache\huggingface\ and %USERPROFILE%\kiosk_models\"

& $PyExe (Join-Path $ScriptDir "setup_models.py")
if ($LASTEXITCODE -ne 0) { Fail "Model download failed. Check internet and disk space." }

# ── Done ───────────────────────────────────────────────────────────────────
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
