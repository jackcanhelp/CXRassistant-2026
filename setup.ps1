<# 
  CXR Analysis Server - Windows Setup Script
  ============================================
  Run in PowerShell as Administrator
#>

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  CXR Analysis Server Setup (Windows)" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: Check Python ---
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "  ERROR: Python not found. Install from https://python.org (3.10 or 3.11)" -ForegroundColor Red
    exit 1
}
python --version
Write-Host ""

# --- Step 2: Create virtual environment ---
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
}
.\venv\Scripts\Activate.ps1
Write-Host "  Virtual environment activated." -ForegroundColor Green
Write-Host ""

# --- Step 3: Install base dependencies ---
Write-Host "[3/6] Installing base dependencies (FastAPI + TorchXRayVision)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install fastapi uvicorn python-multipart jinja2 Pillow numpy scikit-image
Write-Host ""

# --- Step 4: Install PyTorch ---
Write-Host "[4/6] Installing PyTorch..." -ForegroundColor Yellow
Write-Host "  Detecting GPU..." -ForegroundColor Gray

$hasNvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($hasNvidia) {
    Write-Host "  NVIDIA GPU detected! Installing CUDA version..." -ForegroundColor Green
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "  No NVIDIA GPU. Installing CPU version..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}
Write-Host ""

# --- Step 5: Install TorchXRayVision ---
Write-Host "[5/6] Installing TorchXRayVision..." -ForegroundColor Yellow
pip install torchxrayvision
Write-Host ""

# --- Step 6: CheXagent (Optional) ---
Write-Host "[6/6] CheXagent-2-3b Setup (Optional)" -ForegroundColor Yellow
$installChex = Read-Host "  Install CheXagent-2-3b? Requires NVIDIA GPU with 8GB+ VRAM (y/N)"
if ($installChex -eq 'y') {
    Write-Host "  Installing CheXagent dependencies..." -ForegroundColor Green
    pip install transformers==4.39.0 accelerate opencv-python albumentations
    Write-Host "  CheXagent model will be downloaded on first use (~6GB)." -ForegroundColor Gray
}
Write-Host ""

# --- Done ---
Write-Host "=====================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the server:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python server.py"
Write-Host ""
Write-Host "To start with CheXagent enabled:" -ForegroundColor Cyan
Write-Host '  $env:ENABLE_CHEXAGENT="true"'
Write-Host "  python server.py"
Write-Host ""
Write-Host "Server will be at: http://localhost:8765" -ForegroundColor White
