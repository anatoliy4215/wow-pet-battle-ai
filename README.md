# БП — нейросеть для битв питомцев WoW

## Структура
- `bp/data/fetch.py` — скачивает игровые таблицы (wago.tools), данные тренеров (wow-petsim), породы (BattlePetBreedID) в `data/raw/`
- `bp/data/build.py` — собирает `data/processed/*.json` (species, abilities, tamers, …)
- `bp/sim/` — симулятор боя
  - `battle.py` — раунды, питомцы, команды, пассивки
  - `ops.py` — интерпретатор эффектов способностей (все 89 кодов, см. `docs/opcodes.md`)
  - `targets.py` — определение целей эффектов (один враг / вся команда / площадка)
  - `ai.py` — ИИ тренеров по их скриптам (синтаксис tdBattlePetScript)
- `bp/rl/` — обучение (PPO): `features.py` кодирование состояния, `model.py` трансформер, `env.py`, `ppo.py`
- `bp/pvp/` — PvP-модель (отдельная от PvE)
  - `pool.py` — все питомцы, доступные игрокам (2039 видов, редкое качество, 25 ур., все породы), мета и бан-лист
  - `features.py` — что видит модель: только то, что видно в игре (скрытые способности соперника — «одна из двух»)
  - `env.py` — бой двух игроков: выбор первого питомца, одновременные ходы, замены
  - `model.py` — трансформер 11.8 млн параметров: ход, оценка позиции, прогноз хода соперника, прогноз его скрытых способностей
  - `train.py` — самообучение с лигой прошлых версий (PFSP) + боты-якоря; `dashboard.py` → `logs/pvp_dashboard.html`
  - `tactics.py` — экзамен по тактическим приёмам, `search.py` — просчёт вперёд, `ladder.py` — рейтинг Эло версий
  - `teambuilder.py` — подбор команды против меты (учитывает `data/pvp/disabled.json`)
  - `observed.py` — собирает команды реальных соперников из логов в `data/pvp/observed.json`
- `addon/BPLogger/` — аддон WoW для записи боёв и выгрузки коллекции
- `tests/` — проверки механик (часть цифр взята из настоящих логов)

## Команды
```
.venv\Scripts\python.exe -m bp.data.fetch
.venv\Scripts\python.exe -m bp.data.build
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m bp.rl.ppo --workers 14 --envs 24 --updates 3000 --out checkpoints/ppo.pt
start_pvp_training.bat                              (PvP: старт / продолжение после перезагрузки)
.venv\Scripts\python.exe -m bp.pvp.tactics           (экзамен текущей PvP-модели)
.venv\Scripts\python.exe -m bp.pvp.teambuilder       (команда из своей коллекции против меты)
.venv\Scripts\python.exe -m bp.pvp.observed          (учесть соперников из своих PvP-логов)
```

## Проверенные по логам механики
- характеристики: health = (base+breed)·5·lvl·q + 100, power/speed = (base+breed)·lvl·q
- урон = Points·(1+Power/20) · семейство · модификаторы + плоские (×AttackBonus наложившего), округление вниз
- крит = floor(урон · 1.5)
- длительность аур: раунд наложения не считается; Duration 0 = до конца текущего раунда; −1 = бессрочно
- гуманоид +4% max hp в конце раунда, если нанёс урон
