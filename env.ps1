# ============================================================
#  Activate project venv + set required env vars
#  Usage:  powershell -ExecutionPolicy Bypass -File .\env.ps1
#          (or:  . .\env.ps1   if execution policy allows)
# ============================================================
$PROJ = $PSScriptRoot

$env:HF_HOME = Join-Path $PROJ ".hf"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:TRANSFORMERS_VERBOSITY = "error"

$activate = Join-Path $PROJ ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host ""
    Write-Host "[ERROR] venv not found at: $($PROJ).venv" -ForegroundColor Red
    Write-Host "        Run setup_env.bat first." -ForegroundColor Red
    Write-Host ""
    exit 1
}

& $activate

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " [Env ready]" -ForegroundColor Green
Write-Host "   Python       = $env:VIRTUAL_ENV\Scripts\python.exe"
Write-Host "   HF_HOME      = $env:HF_HOME"
Write-Host "   HF_ENDPOINT  = $env:HF_ENDPOINT"
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
