# Архив проекта (2026-10-11)

Проект остановлен и сохранён. Код — в этом репозитории. Всё, что git не хранит (модели, логи боёв,
служебные логи), лежит в GitHub Release **`archive-2026-10-11`** этого же репозитория.

## Где что лежит

| Что | Где | Куда положить при восстановлении |
|---|---|---|
| Код, аддоны, игровые данные, мета, бан-лист, наблюдения соперников | репозиторий (`main`) | — |
| Текущая PvP-модель (шаг 5716) | release: `pvp_actor.pt` | `checkpoints/pvp/actor.pt` |
| Полное состояние обучения (модель + оптимизатор + история + лига) | release: `pvp_stage2.pt` | `checkpoints/pvp/stage2.pt` |
| Модель конца первого этапа | release: `pvp_stage1_final.pt` | `checkpoints/pvp/stage1_final.pt` (и копией в `main.pt`) |
| Якорь для «против ранней версии» | release: `pvp_league_snap_000020.pt` | `checkpoints/pvp/league/snap_000020.pt` |
| Все оставшиеся снимки лиги (22 шт., для `bp.pvp.ladder`) | release: `pvp_league_snapshots.zip` | распаковать в корень проекта |
| PvE-модель (тренеры) | release: `pve_ppo_ach.pt` | `checkpoints/ppo_ach.pt` |
| Логи реальных боёв (BPLogger, нужны тестам `tests/test_replay_logs.py`) | release: `battle_logs.zip` | распаковать в корень проекта (`data/logs/`) |
| Служебные логи: dashboard, рейтинги, запросы советника, ход обучения | release: `logs.zip` | распаковать в корень (`logs/`); токен в `pvp_dashboard.html` заменён на `<redacted>` |
| Контрольные суммы | release: `SHA256SUMS.txt` | `sha256sum -c SHA256SUMS.txt` |

**Не сохранено (секреты, восстановить вручную в `.env`):** `VAST_AI_KEY` (ключ vast.ai), `ADVISOR_TOKEN`
(ключ телефона для советника — сервер сам создаст новый, если строки нет), `GITHUB_TOKEN`.
SSH-ключи (`~/.ssh/vastai_ed25519`, `~/.ssh/bp_node_ed25519`) лежат вне папки проекта.

## Восстановление окружения

```
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
:: Rust-симулятор (Rust через rustup, MSVC Build Tools); maturin develop падает на кириллице в пути:
cd rust\bpsim && ..\..\.venv\Scripts\python.exe -m maturin build --release && ..\..\.venv\Scripts\python.exe -m pip install target\wheels\bpsim-*.whl
.venv\Scripts\python.exe -m bp.sim.export_rust
.venv\Scripts\python.exe -m pytest tests
```

## Состояние на момент архивации

- PvP-модель: шаг 5716, трансформер 11.8 млн параметров, входы с подсказками симулятора (`bp/pvp/probe.py`).
  Против жадного бота ~81–83% контрольных боёв. Экзамен: добивание 0.78, не бьёт в недосягаемого +0.37,
  не оглушает зверька +0.35, «бессмертная» нежить +0.21, сильное семейство +0.40.
- Симулятор сверен с логами боёв (Python и бит-в-бит Rust-копия `rust/bpsim`).
- Советник: аддон `addon/BPAdvisor` (QR) → страница на телефоне → `bp/advisor/server.py` (`start_advisor.bat`).
- Обучение: `bp/pvp/actor_rs.py` (Rust-актор) + `bp/pvp/learner.py`; аренда — `bp/cloud/fleet.py`,
  удалённый узел — `bp/cloud/node.py` (репозиторий `bp-train-node`; на узле друга данных уже нет).
- Покер: `rust/poker` (движок HU NLHE, бот v0) и страница `/poker`.

## Что делать дальше, если вернёмся

1. Модель 5716 обучалась частично до исправления Consume Corpse (op144) — переоценивает Spectral Raven и т.п.;
   нужен цикл дообучения.
2. Честная проверка подсказок: обучение с подсказками против копии без них (одинаковые данные и время).
3. Экзамен-ловушка: «цель в воздухе, но соперник, скорее всего, сменит питомца» — бьёт ли модель.
4. Поиск глубже (2–3 раунда) в обучении; вспомогательная задача «угадай эффект способностей соперника».
5. Бессмертный раунд нежити почти не выучен (экзамен, возможно, сам спорный: Сокрушение лечит через Recovery).
