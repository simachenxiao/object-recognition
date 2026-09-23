@echo off
REM ============================================================
REM  One-click environment setup (for a fresh machine)
REM  Usage:  setup_env.bat
REM
REM  Creates .venv and installs requirements.txt
REM  Note: model weights in .hf\ are NOT re-downloaded if present.
REM ============================================================
set "PROJ=%~dp0"
cd /d "%PROJ%"

echo ============================================================
echo  Project : %PROJ%
echo ============================================================
echo.

echo [1/3] Creating venv ...
if exist ".venv\Scripts\python.exe" (
    echo       .venv already exists - skipping
) else (
    python -m venv .venv
    if errorlevel 1 goto :err
    echo       created
)
echo.

echo [2/3] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
if errorlevel 1 goto :err
echo       done
echo.

echo [3/3] Installing requirements ...
echo       (torch is ~250MB, this may take a few minutes)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :err
echo       done
echo.

if exist ".hf\hub" (
    echo [OK] Local model cache found at .hf\hub - no download needed
) else (
    echo [NOTE] No local model cache ^(.hf\hub^).
    echo        First run will download ~3.5GB via HF_ENDPOINT mirror.
)
echo.
echo ============================================================
echo  Setup complete.  Now run:  env.bat
echo ============================================================
echo.
goto :eof

:err
echo.
echo [ERROR] Setup failed. Check the messages above.
echo.
exit /b 1
