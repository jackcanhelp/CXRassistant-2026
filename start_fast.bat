@echo off
REM ==========================================
REM  CXR Server - Quick Start (XRV only)
REM ==========================================

call venv\Scripts\activate.bat
echo Starting CXR Analysis Server (TorchXRayVision mode)...
echo.
echo   URL: http://localhost:8765
echo   Press Ctrl+C to stop
echo.
python server.py
