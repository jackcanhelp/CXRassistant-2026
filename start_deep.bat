@echo off
REM ==========================================
REM  CXR Server - Deep Mode (XRV + llama.cpp GGUF GPU+CPU hybrid)
REM  需先下載 GGUF 模型到 models\ 資料夾
REM  建議: Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf (~7GB)
REM  來源: https://huggingface.co/bartowski/Llama-3.2-11B-Vision-Instruct-GGUF
REM ==========================================

call venv\Scripts\activate.bat
set ENABLE_CHEXAGENT=true
set VLM_BACKEND=llamacpp
set LLAMACPP_MODEL_PATH=models\Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf
set LLAMACPP_N_GPU_LAYERS=35
set LLAMACPP_N_CTX=8192
echo Starting CXR Analysis Server (Deep mode: llama.cpp GPU+CPU hybrid)...
echo.
echo   URL: http://localhost:8765
echo   VLM: Llama-3.2-11B-Vision (GGUF Q4, GPU layers=35, rest on CPU/RAM)
echo   NOTE: First run will be slow (model loading ~1-2min)
echo   Press Ctrl+C to stop
echo.
python server.py
