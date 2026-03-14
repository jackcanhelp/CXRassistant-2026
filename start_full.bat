@echo off
cd /d "%~dp0"
set VLM_BACKEND=chexone
set VLM_DEVICE=cuda
set CXR_SECRET_TOKEN=19910209
echo Starting CXR Analysis Server (CheXOne mode)...
echo URL: http://localhost:8765
echo Press Ctrl+C to stop
echo.
venv\Scripts\python.exe server.py
