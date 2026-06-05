# start.ps1 — activate venv and launch the kiosk on Windows
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Activate = Join-Path $ScriptDir "kiosk_venv\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Host "[ERROR] venv not found. Run .\bootstrap.ps1 first." -ForegroundColor Red
    exit 1
}

. $Activate
python (Join-Path $ScriptDir "start.py")
