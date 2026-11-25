@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================================
echo    HydroLab - Hydrology Workbench
echo  ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found. Install Python 3.10+ and add it to PATH.
  pause
  exit /b 1
)

python -c "import pandas, numpy, scipy, matplotlib, flask" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] Installing core Python dependencies for the first run...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed. Check your network and retry.
    pause
    exit /b 1
  )
)

python workbench\start.py %*
if errorlevel 1 (
  echo.
  echo Workbench stopped. If the port is busy, try: python workbench\start.py --port 8766
  pause
)
endlocal
