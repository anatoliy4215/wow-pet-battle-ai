@echo off
rem Continue training from the last checkpoint (safe to run after a reboot).
rem Usage: start_training.bat [total steps, default 10000]
cd /d "%~dp0"
set STEPS=%1
if "%STEPS%"=="" set STEPS=10000
if exist logs\ppo_ach.log copy /y logs\ppo_ach.log logs\ppo_ach_prev.log >nul
echo Training up to step %STEPS%. Progress: logs\dashboard.html
.venv\Scripts\python.exe -m bp.rl.ppo --achievements --workers 14 --envs 24 --steps 64 --updates %STEPS% --resume checkpoints/ppo_ach.pt --out checkpoints/ppo_ach.pt > logs\ppo_ach.log 2>&1
