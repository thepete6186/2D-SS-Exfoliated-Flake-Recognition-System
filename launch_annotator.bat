@echo off
echo ========================================
echo Camera Annotator Launcher
echo ========================================
echo.
echo Killing Labscope service...
taskkill /F /IM LabscopeService.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo.
echo Starting Camera Annotator...
python camera/camera_annotator.py
echo.
echo Camera Annotator closed.
pause