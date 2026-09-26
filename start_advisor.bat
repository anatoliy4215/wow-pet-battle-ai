@echo off
rem Phone advisor: server for https://<your-tunnel-host> (cloudflared tunnel -> this PC :5173).
rem The phone link with the key is ADVISOR_TOKEN in .env:  https://<your-tunnel-host>/?k=<token>
cd /d "%~dp0"
echo Advisor server on port 5173. Close this window to stop it.
.venv\Scripts\python.exe -m bp.advisor.server
