@echo off
rem PvP training is in stage 2 now (look-ahead). This keeps the old name working after a reboot.
rem Stage 1 (self-play PPO) can still be run by hand:
rem   .venv\Scripts\python.exe -m bp.pvp.train --resume --out checkpoints/pvp/main.pt
cd /d "%~dp0"
call start_pvp_stage2.bat
