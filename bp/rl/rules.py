"""Check a human-readable plan: a simple rule controller played in the simulator.

A plan is {pet name: [(ability name, condition), ...]} tried in order; conditions get (battle, pet).
Replacement order when a pet dies: list `order`.
"""
from __future__ import annotations

import numpy as np

from ..sim import constants as C
from ..sim.data import game_data
from .env import PetBattleEnv, ScenarioSampler


class RuleController:
    def __init__(self, rules, order):
        self.rules = rules
        self.order = order

    def choose(self, battle, side):
        t = battle.teams[side]
        acts = battle.legal_actions(side)
        if battle.needs_replacement(side):
            for name in self.order:
                for i, p in enumerate(t.pets):
                    if p.name == name and not p.dead:
                        return 3 + i
            return acts[0]
        if acts == [C.A_PASS]:
            return C.A_PASS
        pet = t.active_pet
        gd = game_data()
        for ability, cond in self.rules.get(pet.name, []):
            for slot, aid in enumerate(pet.abilities):
                if aid and gd.abilities[aid]["name"] == ability and slot in acts and (cond is None or cond(battle, pet)):
                    return slot
        abil = [a for a in acts if a in C.A_ABILITY]
        return abil[0] if abil else C.A_PASS


def winrate(tamer, team, rules, order, games=300):
    wins, rounds = [], []
    for g in range(games):
        e = PetBattleEnv(ScenarioSampler(seed=0), seed=50_000 + g)
        e.reset((tamer, team))
        ctrl = RuleController(rules, order)
        done = False
        while not done:
            _, _, done = e.step(ctrl.choose(e.battle, 0))
        wins.append(e.battle.winner == 0)
        rounds.append(e.battle.round)
    return float(np.mean(wins)), float(np.mean(rounds))
