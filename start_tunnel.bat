@echo off
REM ==========================================
REM  Cloudflare Tunnel - Quick Tunnel
REM  讓醫院瀏覽器能連到家中 CXR Server
REM ==========================================
REM
REM  首次使用請先安裝 cloudflared:
REM    winget install Cloudflare.cloudflared
REM  或從 https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
REM
REM  此腳本會建立一個臨時的 Quick Tunnel（免費，無需帳號）
REM  每次重啟會產生新的隨機網址
REM  如果需要固定網址，請參考 README.md 的進階設定
REM ==========================================

echo ============================================
echo   Starting Cloudflare Tunnel...
echo   Tunneling localhost:8765 to public HTTPS
echo ============================================
echo.
echo   等待 tunnel 建立後，會顯示一個 https://xxxx.trycloudflare.com 網址
echo   在醫院的瀏覽器開啟該網址即可使用
echo.

cloudflared tunnel --url http://localhost:8765
