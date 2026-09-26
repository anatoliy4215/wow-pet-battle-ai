"""Single-agent environment: our team (side 0) vs a scripted NPC tamer (side 1)."""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..sim.battle import Battle
from ..sim.data import PetSpec, game_data
from ..sim.run import tamer_specs
from .features import action_mask, encode

ROOT = Path(__file__).resolve().parents[2]


def load_collection(level=25):
    path = ROOT / "data" / "processed" / "collection.json"
    if not path.exists():
        return None
    pets = json.loads(path.read_text(encoding="utf-8"))["pets"]
    gd = game_data()
    return [p for p in pets if p["level"] >= level and p["battle"] and p["species"] in gd.species
            and all(gd.species[p["species"]]["abilities"])]


def achievement_tasks():
    path = ROOT / "data" / "processed" / "achievements.json"
    tasks = set()
    for a in json.loads(path.read_text(encoding="utf-8")):
        if a["kind"] not in ("family_tamer", "tamer_list"):
            continue
        for t in a["tamers"]:
            if t.get("tamer_id") is None:
                continue
            fam = t.get("required_family_override", a["required_family"])
            tasks.add((fam, int(t["tamer_id"])))
    return sorted(tasks, key=lambda x: (x[1], -1 if x[0] is None else x[0]))


class ScenarioSampler:
    """Picks (tamer, our team).

    Teams come from the player's collection (exact stats) when data/processed/collection.json exists,
    otherwise random level-25 rare pets. Optionally restricted to one family.
    """

    def __init__(self, tamer_ids=None, family=None, seed=None, use_collection=True, achievements=False):
        gd = game_data()
        self.rng = random.Random(seed)
        self.tamers = tamer_ids or [t for t, v in gd.tamers.items() if v["active"] and not v["wild"] and v["pets"]]
        col = load_collection() if use_collection else None
        if col:
            self.pool = [("own", p) for p in col]
        else:
            self.pool = [("any", {"species": s, "family": v["family"]}) for s, v in gd.species.items() if v["battle"]]
        self.family = family
        # (family or None, tamer_id) tasks from the player's unfinished achievements
        self.tasks = achievement_tasks() if achievements else None

    def _pool(self, family):
        return [x for x in self.pool if family is None or x[1]["family"] == family]

    def sample(self):
        if self.tasks:
            family, tid = self.rng.choice(self.tasks)
        else:
            family, tid = self.family, self.rng.choice(self.tamers)
        self.last_family = family
        return tid, self.team(family)

    def team(self, family=None):
        team = []
        for kind, x in self.rng.sample(self._pool(family), 3):
            choices = "".join(self.rng.choice("12") for _ in range(3))
            if kind == "own":
                spec = PetSpec.from_choices(x["species"], choices, level=x["level"], quality=x["quality"],
                                            breed=x["breed"] or 3, stats=tuple(x["stats"]))
            else:
                sp = game_data().species[x["species"]]
                breed = self.rng.choice(sp["breeds"]) if sp["breeds"] else 3
                spec = PetSpec.from_choices(x["species"], choices, breed=breed)
            team.append(spec)
        return team


class PetBattleEnv:
    def __init__(self, sampler: ScenarioSampler, seed=None, max_rounds=100):
        self.sampler = sampler
        self.rng = random.Random(seed)
        self.max_rounds = max_rounds
        self.battle = None

    def reset(self, scenario=None):
        self.family = None
        if scenario is None:
            scenario = self.sampler.sample()
            self.family = getattr(self.sampler, "last_family", None)
        tid, team = scenario
        specs_b, self.opponent = tamer_specs(tid)
        self.tamer = tid
        self.battle = Battle(team, specs_b, seed=self.rng.getrandbits(32))
        self._advance_opponent_only()
        return self.observe()

    def observe(self):
        return encode(self.battle, 0), action_mask(self.battle, 0)

    def _advance_opponent_only(self):
        # if only the opponent must replace a dead pet, do it without asking us
        b = self.battle
        while not b.over and b.needs_replacement(1) and not b.needs_replacement(0):
            b.step(6, self.opponent.choose(b, 1))

    def step(self, action):
        b = self.battle
        if action not in b.legal_actions(0):
            action = b.legal_actions(0)[0]
        opp = self.opponent.choose(b, 1) if not b.needs_replacement(0) or b.needs_replacement(1) else 6
        b.step(action, opp)
        self._advance_opponent_only()
        done = b.over or b.round >= self.max_rounds
        reward = 0.0
        if done:
            reward = 1.0 if b.winner == 0 else -1.0
        return self.observe(), reward, done
