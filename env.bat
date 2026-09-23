@echo off
REM ============================================================
REM  Activate project venv + set required env vars
REM  Usage:  env.bat
REM ============================================================
set "PROJ=%~dp0"
set "HF_HOME=%PROJ%.hf"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "TRANSFORMERS_VERBOSITY=error"

if not exist "%PROJ%.venv\Scripts\activate.bat" (
    echo.
    echo [ERROR] venv not found at: %PROJ%.venv
    echo         Run setup_env.bat first.
    echo.
    exit /b 1
)

call "%PROJ%.venv\Scripts\activate.bat"

echo.
echo ============================================================
echo  [Env ready]
echo    Python       = %VIRTUAL_ENV%\Scripts\python.exe
echo    HF_HOME      = %HF_HOME%
echo    HF_ENDPOINT  = %HF_ENDPOINT%
echo ============================================================
echo.
