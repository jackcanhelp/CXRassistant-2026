@echo off
cd /d "%~dp0"

REM ================================================
REM  選擇 Tunnel 模式：
REM    named = 固定網址（需先完成 cloudflared 設定）
REM    quick = 每次隨機網址（免帳號，立即可用）
REM ================================================
set TUNNEL_MODE=named

if "%TUNNEL_MODE%"=="named" (
    echo Starting Named Tunnel (fixed URL)...
    echo URL: 請看 Cloudflare Dashboard 或 config.yml 設定的 hostname
    echo.
    cloudflared tunnel run cxr-server
) else (
    echo Starting Quick Tunnel (random URL)...
    echo.
    cloudflared tunnel --url http://localhost:8765
)
