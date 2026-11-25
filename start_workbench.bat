@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================================
echo    HydroLab - 水文计算工作台
echo  ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 未找到 Python。请先安装 Python 3.10+ 并加入 PATH。
  pause
  exit /b 1
)

python -c "import flask" >nul 2>nul
if errorlevel 1 (
  echo [SETUP] 首次运行，正在安装 Python 依赖...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo [ERROR] 依赖安装失败，请检查网络后重试。
    pause
    exit /b 1
  )
)

python workbench\start.py %*
if errorlevel 1 (
  echo.
  echo 工作台已退出。若端口被占用，可尝试：python workbench\start.py --port 8766
  pause
)
endlocal
