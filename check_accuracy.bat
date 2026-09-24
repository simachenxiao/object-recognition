@echo off
setlocal
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set HF_HOME=%~dp0.hf
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] venv not found. Run setup_env.bat first.
  pause
  exit /b 1
)
echo ============================================================
echo   Accuracy check  /  jian cha shi bie zhun que lv
echo   (edit descriptions in the web UI, then run this to verify)
echo ============================================================
echo.
".venv\Scripts\python.exe" src_eval_tray.py images %*
echo.
pause
