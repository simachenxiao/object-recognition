@echo off
REM ============================================================
REM  Item Recognition Demo - Web UI launcher
REM  Usage:  run_web.bat
REM          run_web.bat --port 8080
REM          run_web.bat --no-browser
REM ============================================================
chcp 65001 >nul
call "%~dp0env.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] venv not ready. Run setup_env.bat first.
    pause
    exit /b 1
)
echo.
echo  Starting web demo ... press Ctrl+C to stop
echo.
python -u "%~dp0src\web_demo.py" %*
echo.
pause
