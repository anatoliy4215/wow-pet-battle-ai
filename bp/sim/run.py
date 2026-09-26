"""Helpers to set up and play battles."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .ai import ScriptController
from .battle import Battle
from .data import PetSpec, game_data

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def script_overrides():
    path = ROOT / "data" / "tamer_scripts.json"
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def tamer_specs(tamer_id):
    t = game_data().tamers[tamer_id]
    override = script_overrides().get(t["npc"], {})
    specs, scripts = [], []
    for p in t["pets"]:
        if p["species"] is None:
            continue
        specs.append(PetSpec(species=p["species"], abilities=p["abilities"], level=p["level"] or 25,
                             quality=p["quality"] if p["quality"] is not None else 3, breed=p["breed"]))
        scripts.append(override.get(str(p["slot"]), p["behaviour"]))
    return specs, ScriptController(scripts, swap_when_cc=bool(override.get("swap_when_cc")))


def play(battle: Battle, ctrl_a, ctrl_b, max_steps=400):
    steps = 0
    while not battle.over and steps < max_steps:
        a = ctrl_a.choose(battle, 0)
        b = ctrl_b.choose(battle, 1)
        battle.step(a, b)
        steps += 1
    if not battle.over:
        battle.winner = -1
    return battle.winner
