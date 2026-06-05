# preflight.ps1 — check prerequisites, then run bootstrap.ps1
#
# Usage (in PowerShell, from the repo directory):
#     .\preflight.ps1
#
# Checks Python, AMD HIP SDK, GPU, disk space, and execution policy.
# If everything is satisfied, offers to run bootstrap.ps1 automatically.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Pass($msg)  { Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function FailLn($m)  { Write-Host "  [MISS] $m" -ForegroundColor Red }
function Info($msg)  { Write-Host "         $msg" -ForegroundColor DarkGray }

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  AI Scene Style Kiosk — Windows Preflight" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

$problems = @()

# ── 1. Windows version ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "[1/6] Windows version"
$os = Get-CimInstance Win32_OperatingSystem
if ($os.Caption -match "Windows 11") {
    Pass "$($os.Caption) ($($os.Version))"
} else {
    Warn "$($os.Caption) — this branch targets Windows 11"
}

# ── 2. Execution policy ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/6] PowerShell execution policy"
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -in @("RemoteSigned", "Unrestricted", "Bypass")) {
    Pass "CurrentUser policy = $policy"
} else {
    FailLn "CurrentUser policy = $policy (scripts will be blocked)"
    Info "Fix: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
    $problems += "execution-policy"
}

# ── 3. Python ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/6] Python 3.10 / 3.11 / 3.12"
$pythonOk = $false
foreach ($candidate in @("py -3.12", "py -3.11", "py -3.10", "python")) {
    try {
        $verRaw = & cmd /c "$candidate --version 2>&1"
        if ($LASTEXITCODE -eq 0 -and $verRaw -match "Python\s+3\.(1[0-2])") {
            Pass "$verRaw  (use: $candidate)"
            $pythonOk = $true
            break
        }
    } catch { }
}
if (-not $pythonOk) {
    FailLn "Python 3.10/3.11/3.12 not found on PATH"
    Info "Download: https://www.python.org/downloads/"
    Info "IMPORTANT: check 'Add Python to PATH' during install"
    $problems += "python"
}

# ── 4. AMD GPU detection ────────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/6] AMD GPU"
$gpus = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" }
if ($gpus) {
    foreach ($g in $gpus) {
        $vram = if ($g.AdapterRAM) { "{0:N1} GB" -f ($g.AdapterRAM / 1GB) } else { "?" }
        Pass "$($g.Name)  (VRAM reported: $vram)"
        if ($g.Name -match "R9700") {
            Info "Target GPU detected: Radeon AI PRO R9700 (gfx1201, RDNA 4)"
        }
    }
    Info "Note: Windows under-reports VRAM > 4 GB on some drivers; check Adrenalin for the real number."
} else {
    FailLn "No AMD GPU found by Win32_VideoController"
    Info "Make sure the latest AMD Adrenalin driver is installed."
    $problems += "gpu"
}

# ── 5. AMD HIP SDK ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[5/6] AMD HIP SDK (ROCm for Windows)"
$hipFound = $false
$hipPath  = $null
if ($env:HIP_PATH -and (Test-Path $env:HIP_PATH)) {
    $hipPath = $env:HIP_PATH
    $hipFound = $true
}
if (-not $hipFound) {
    foreach ($p in @(
        "C:\Program Files\AMD\ROCm\6.4",
        "C:\Program Files\AMD\ROCm\6.3",
        "C:\Program Files\AMD\ROCm\6.2"
    )) {
        if (Test-Path $p) { $hipPath = $p; $hipFound = $true; break }
    }
}
if ($hipFound) {
    Pass "HIP SDK at $hipPath"
} else {
    Warn "HIP SDK not detected"
    Info "Download:  https://www.amd.com/en/developer/resources/rocm-hub.html"
    Info "(pick 'HIP SDK for Windows', ~2 GB)"
    Info "Without it, the bootstrap falls back to torch-directml — works but slower."
}

# ── 6. Disk space ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] Disk space"
$drive = (Get-Item $ScriptDir).PSDrive.Name
$free  = (Get-PSDrive $drive).Free / 1GB
if ($free -ge 25) {
    Pass ("{0}: has {1:N1} GB free  (need ~20 GB for models)" -f $drive, $free)
} else {
    FailLn ("{0}: only {1:N1} GB free, need ~20 GB" -f $drive, $free)
    $problems += "disk"
}

# ── Summary ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
if ($problems.Count -eq 0) {
    Write-Host "  All checks passed." -ForegroundColor Green
    Write-Host "=======================================================" -ForegroundColor Cyan
    Write-Host ""
    $ans = Read-Host "Run .\bootstrap.ps1 now? [Y/n]"
    if ($ans -eq "" -or $ans -match "^[Yy]") {
        & (Join-Path $ScriptDir "bootstrap.ps1")
    } else {
        Write-Host "Skipped. Run .\bootstrap.ps1 when you're ready." -ForegroundColor DarkGray
    }
} else {
    Write-Host "  Blocking issues: $($problems -join ', ')" -ForegroundColor Red
    Write-Host "=======================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Fix the items marked [MISS] above, then re-run .\preflight.ps1"
    exit 1
}
