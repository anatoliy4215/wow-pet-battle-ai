@echo off
rem PvP stage 2 (look-ahead training): start or continue after a reboot.
rem   learner  - trains the network on look-ahead games, writes checkpoints/pvp/stage2.pt
rem   actor    - plays look-ahead games on this PC
rem   fleet    - keeps rented vast.ai boxes fed and collects their games (stops them below $0.30 credit)
rem Progress: logs\pvp_dashboard.html
cd /d "%~dp0"
start "pvp learner" /min cmd /c ".venv\Scripts\python.exe -m bp.pvp.learner --resume --out checkpoints/pvp/stage2.pt > logs\pvp_learner.log 2>&1"
start "pvp actor" /min cmd /c ".venv\Scripts\python.exe -m bp.pvp.actor --weights checkpoints/pvp/actor.pt --out data/pvp/shards/local --workers 10 > logs\pvp_actor.log 2>&1"
start "pvp fleet" /min cmd /c ".venv\Scripts\python.exe -m bp.cloud.fleet run > logs\pvp_fleet.log 2>&1"
echo Stage 2 started. Progress: logs\pvp_dashboard.html
