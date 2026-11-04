# WoW Pet Battle AI

A from-scratch **World of Warcraft pet battle simulator** (Python + a fast Rust port) and neural networks trained on it by self-play:

- **Simulator** — every ability effect opcode of the game (89 of them, documented in [docs/opcodes.md](docs/opcodes.md)), passives, weather, breeds; damage, health and aura-duration formulas **checked against real in-game battle logs** (see below and `tests/`).
- **PvE model** — PPO agent that beats tamer NPCs (their scripts in tdBattlePetScript syntax) and writes step-by-step plans.
- **PvP model** — 11.8M-parameter transformer trained with league self-play (PFSP) + one-round look-ahead search; sees only what a player sees (hidden enemy abilities are "one of two"). Wins ~81–83% against a greedy bot.
- **Team builder** — picks a team from your collection against the current meta.
- **Addons** — `BPLogger` records battles (full state every round) and exports your collection; `BPAdvisor` shows the battle state as a QR code that a phone page scans to get the model's advice.
- **Rust engine** (`rust/bpsim`) — the simulator, features and search as a Python extension for fast self-play.

The project is archived (October 2026). Trained models and battle logs are attached to the
[`archive-2026-10-11` release](https://github.com/anatoliy4215/wow-pet-battle-ai/releases/tag/archive-2026-10-11);
how to restore everything is in [docs/ARCHIVE.md](docs/ARCHIVE.md) (Russian).

**Verified mechanics** (from logs):
`health = (base+breed)·5·lvl·q + 100`, `power/speed = (base+breed)·lvl·q`;
`damage = Points·(1+Power/20) · family · modifiers + flat`, rounded down; `crit = floor(damage·1.5)`;
aura duration does not count the round it was applied; Humanoid heals 4% max hp at round end if it dealt damage.

Quick start (Windows, Python 3.13, CUDA build of PyTorch):
```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m bp.data.fetch
.venv\Scripts\python.exe -m bp.data.build
.venv\Scripts\python.exe -m pytest tests
```

Notes: this is a research / hobby project. The addons only read the game state and never automate anything.
`bp/poker` and `rust/poker` are an unrelated side experiment (heads-up hold'em engine and bot) that shares the advisor server.
Code is MIT-licensed; game data and third-party references are not (see [LICENSE](LICENSE)).

---

# По-русски: БП — нейросеть для битв питомцев WoW

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

## Архив
Проект в архиве: как восстановить и где лежат модели — [docs/ARCHIVE.md](docs/ARCHIVE.md).

## Лицензия
Код — MIT ([LICENSE](LICENSE)). Игровые данные и сторонние файлы в `data/raw/` под неё не подпадают.
