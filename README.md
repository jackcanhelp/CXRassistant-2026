# CXR Analysis Server

遠端胸腔 X 光 AI 分析系統 — 在家中電腦運行，從醫院瀏覽器使用。

```
醫院瀏覽器 ──HTTPS──> Cloudflare Tunnel ──> 家中電腦 (FastAPI + AI Models)
```

## 系統架構

```
cxr-server/
├── server.py                 # FastAPI 主程式
├── requirements.txt          # Python 依賴
├── setup.ps1                 # Windows 一鍵安裝腳本
├── start_fast.bat            # 啟動（僅 TorchXRayVision）
├── start_full.bat            # 啟動（XRV + CheXagent）
├── start_tunnel.bat          # 啟動 Cloudflare Tunnel
├── templates/
│   └── index.html            # 前端上傳介面
├── uploads/                  # 上傳的 CXR 影像（自動建立）
└── antigravity-skill/        # Antigravity IDE Skill
    ├── SKILL.md
    └── scripts/
        ├── classify_xrv.py   # TorchXRayVision 獨立腳本
        └── report_chexagent.py  # CheXagent 獨立腳本
```

## 快速開始

### 1. 安裝（PowerShell）

```powershell
cd cxr-server
.\setup.ps1
```

或手動安裝：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# PyTorch（有 NVIDIA GPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# PyTorch（無 GPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 其他依賴
pip install fastapi uvicorn python-multipart jinja2 Pillow numpy scikit-image torchxrayvision

# CheXagent（可選，需要 GPU）
pip install transformers==4.39.0 accelerate opencv-python albumentations
```

### 2. 啟動 Server

**快速模式**（僅 TorchXRayVision，任何電腦都能跑）：
```powershell
.\start_fast.bat
# 或
python server.py
```

**完整模式**（加上 CheXagent，需要 NVIDIA GPU 8GB+）：
```powershell
.\start_full.bat
# 或
$env:ENABLE_CHEXAGENT="true"; python server.py
```

Server 啟動後開啟 http://localhost:8765

### 3. 建立 Cloudflare Tunnel（讓醫院能連）

**安裝 cloudflared：**
```powershell
winget install Cloudflare.cloudflared
```

**啟動 Quick Tunnel（最簡單，免帳號）：**
```powershell
.\start_tunnel.bat
# 或
cloudflared tunnel --url http://localhost:8765
```

Terminal 會顯示一個隨機的 `https://xxxx-xxxx.trycloudflare.com` 網址。
在醫院瀏覽器開啟這個網址就能使用。

> Quick Tunnel 每次重啟網址會變。如果需要固定網址，見下方進階設定。

### 4. 在醫院使用

1. 開啟醫院電腦的瀏覽器
2. 輸入 Cloudflare Tunnel 提供的 HTTPS 網址
3. 上傳 CXR 影像
4. 選擇分析模式（快速 / 完整）
5. 等待結果顯示

## 進階設定

### 固定 Tunnel 網址（需要免費 Cloudflare 帳號）

```powershell
# 登入
cloudflared tunnel login

# 建立 tunnel
cloudflared tunnel create cxr-server

# 設定 DNS（假設你有 example.com 網域）
cloudflared tunnel route dns cxr-server cxr.example.com

# 建立設定檔 ~/.cloudflared/config.yml
```

config.yml：
```yaml
tunnel: <your-tunnel-id>
credentials-file: C:\Users\<you>\.cloudflared\<tunnel-id>.json

ingress:
  - hostname: cxr.example.com
    service: http://localhost:8765
  - service: http_status:404
```

啟動：
```powershell
cloudflared tunnel run cxr-server
```

### 安全性設定

1. **設定 Token 認證**：
   ```powershell
   $env:CXR_SECRET_TOKEN="your-strong-random-token"
   python server.py
   ```

2. **Cloudflare Access**（推薦）：
   在 Cloudflare Zero Trust 後台設定 Access Policy，
   只允許你的 Email 登入，比 token 更安全。

### 在 Antigravity IDE 中使用

將 `antigravity-skill/` 目錄複製到：
- 全域：`%USERPROFILE%\.gemini\antigravity\skills\cxr-analyzer\`
- 專案：`<project>\.agent\skills\cxr-analyzer\`

之後在 Antigravity 中對 agent 說「分析這張 CXR」就會自動觸發。

## 硬體需求

| 模式 | CPU | RAM | GPU | 速度 |
|------|-----|-----|-----|------|
| 快速（XRV only） | 任何現代 CPU | 4GB+ | 不需要 | ~2秒 |
| 完整（XRV + CheXagent） | 任何 | 16GB+ | RTX 3060+ 8GB | ~5-10秒 |
| 完整（CPU fallback） | 8核+ | 32GB+ | 不需要 | ~2-3分鐘 |

## 注意事項

- 本系統僅供研究與輔助參考，不能作為臨床診斷依據
- Nodule/Mass 偽陰性率約 25-30%，不能僅依賴 AI 排除腫瘤
- 所有分析結果需要合格醫師最終判讀
- 上傳的影像僅存在本地電腦，不會傳送到任何雲端服務

## License

本專案使用的模型授權：
- TorchXRayVision: Apache 2.0
- CheXagent-2-3b: CC-BY-NC-4.0（僅限非商業使用）
