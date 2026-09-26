"""Simple scripted PvP opponents: sparring partners early in training and fixed rating anchors later."""
from __future__ import annotations

from ..sim import constants as C
from ..sim.data import game_data
from .abilityinfo import descriptor
from .features import PHASE_LEAD, effectiveness


def _pet_score(pet, enemy):
    """How good `pet` looks against `enemy` (family matchup of its attacks, health left)."""
    gd = game_data()
    best = max((effectiveness(gd.abilities[a]["family"], enemy.family) for a in pet.abilities if a), default=0)
    return best - effectiveness(enemy.family, pet.family) + pet.hp / max(1, pet.max_hp)


class RandomBot:
    def act(self, battle, side, phase, rng):
        if phase == PHASE_LEAD:
            return 3 + rng.randrange(len(battle.teams[side].pets))
        return rng.choice(battle.legal_actions(side))


class GreedyBot:
    """Highest expected damage now, heal when low, set up an aura if it is missing, sensible swaps."""

    def act(self, battle, side, phase, rng):
        me, en = battle.teams[side], battle.teams[1 - side]
        if phase == PHASE_LEAD:
            scores = [sum(_pet_score(p, e) for e in en.pets) for p in me.pets]
            return 3 + max(range(len(scores)), key=lambda i: scores[i] + rng.random() * 0.1)
        legal = battle.legal_actions(side)
        if battle.needs_replacement(side):
            return max(legal, key=lambda a: _pet_score(me.pets[a - 3], en.active_pet) + rng.random() * 0.1)
        pet, foe = me.active_pet, en.active_pet
        gd = game_data()
        best, best_v = C.A_PASS, -1e9
        for a in legal:
            if a not in C.A_ABILITY:
                continue
            aid = pet.abilities[a]
            d = descriptor(aid)
            ab = gd.abilities[aid]
            eff = effectiveness(ab["family"], foe.family)
            mult = 1.5 if eff > 0 else (C.WEAK_MULT if eff < 0 else 1.0)
            v = (d[13] * 50 + d[23] * 50 * 2) * (1 + pet.power / 20) * mult * d[15]
            v += d[20] * foe.max_hp
            if d[16] > 0:
                v += d[16] * 50 * (1 + pet.power / 20) * (2.0 if pet.hp < 0.4 * pet.max_hp else 0.3)
            if d[17] or d[18] or d[19]:
                v += 40 * (1 + pet.power / 20) * 0.5
            if foe.state(C.S_UNTARGETTABLE, battle) > 0 or foe.state(C.S_DODGE, battle) >= 100:
                v *= 0.1
            v *= 1 + 0.1 * rng.random()
            if v > best_v:
                best, best_v = a, v
        swaps = [a for a in legal if a in C.A_SWAP]
        if swaps and pet.hp < 0.25 * pet.max_hp and rng.random() < 0.3:
            return max(swaps, key=lambda a: _pet_score(me.pets[a - 3], foe))
        return best
