@echo off
rem Short training session: learner + actor on this PC overnight, 9 hours (change the number below),
rem then the model is saved, the phone advisor picks it up, and the ratings on the dashboard are refreshed.
cd /d "%~dp0"
set MIN=540
start "pvp actor" /min cmd /c ".venv\Scripts\python.exe -m bp.pvp.actor --weights checkpoints/pvp/actor.pt --out data/pvp/shards/local --workers 14 > logs\pvp_actor.log 2>&1"
.venv\Scripts\python.exe -m bp.pvp.learner --resume --out checkpoints/pvp/stage2.pt --buffer 1500000 --minutes %MIN% > logs\pvp_learner.log 2>&1
echo Done.
