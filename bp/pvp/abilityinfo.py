"""Static per-ability descriptor vectors built from the DB2 effect rows.

The network learns an embedding per ability id, but rare abilities are seen seldom; these hand-made
numbers (damage points, accuracy, cooldown, duration, what states the ability / its aura sets, ...)
let it generalise. Everything here is also valid for auras (auras are abilities in DB2), and it is a
plain lookup table, so it can later be exported for an in-game (Lua) version of the model.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..sim.data import game_data
from ..sim.targets import KIND

# ops whose `aura` lands on the enemy (target) vs on the caster / own pad
ENEMY_AURA_OPS = {50, 52, 54, 77, 86, 168, 172, 178, 230}
SELF_AURA_OPS = {26, 28, 63, 75, 76, 137, 150, 223, 248}
PCT_DAMAGE_OPS = {62, 68, 363, 197}
PERCENT_STATES = {23, 24, 25, 40, 41, 65, 66, 73, 87, 88, 99, 314}


@lru_cache(maxsize=1)
def state_index():
    """State ids that any ability sets, in a fixed order (index into the state part of the vector)."""
    gd = game_data()
    ids = sorted({s for a in gd.abilities.values() for s in a["states"]})
    return {s: i for i, s in enumerate(ids)}


def _state_vec(states):
    idx = state_index()
    v = np.zeros(len(idx), dtype=np.float32)
    for s, val in states.items():
        if s in idx:
            v[idx[s]] = np.clip(val / 100 if s in PERCENT_STATES else val, -3, 3)
    return v


def n_features():
    return 24 + 2 * len(state_index())


@lru_cache(maxsize=None)
def descriptor(aid):
    gd = game_data()
    v = np.zeros(n_features(), dtype=np.float32)
    ab = gd.abilities.get(aid)
    if ab is None:
        return v
    fam = ab.get("family", -1)
    if 0 <= fam < 10:
        v[fam] = 1
    v[10] = ab.get("cooldown", 0) / 5
    main = [t for t in ab["turns"] if t["type"] == 0]
    trig = [t for t in ab["turns"] if t["type"] == 1]
    v[11] = len([t for t in main if t["effects"]]) / 3
    v[12] = float(bool(trig))
    dmg = heal = hits = 0
    acc = []
    enemy_aura = self_aura = None
    dur = 0
    for t in ab["turns"]:
        for e in t["effects"]:
            k = KIND.get(e["op"])
            p = e["params"]
            if k == "damage" and not p.get("IsPeriodic"):
                if e["op"] in PCT_DAMAGE_OPS:
                    v[20] += p.get("Points", 0) / 100
                else:
                    dmg += p.get("Points", 0)
                hits += 1
                if p.get("Accuracy"):
                    acc.append(min(p["Accuracy"], 100))
            elif k == "heal":
                heal += p.get("Points", 0) or p.get("Percentage", 0)
            if e["aura"]:
                if e["op"] in ENEMY_AURA_OPS and enemy_aura is None:
                    enemy_aura = e["aura"]
                elif e["op"] in SELF_AURA_OPS and self_aura is None:
                    self_aura = e["aura"]
                dur = max(dur, p.get("Duration", 0))
            if e["op"] == 80:
                v[19] = 1
            if e["range"] == 2:
                v[21] = 1
    v[13] = dmg / 50
    v[14] = hits / 3
    v[15] = (min(acc) if acc else 100) / 100
    v[16] = heal / 50
    v[17] = float(enemy_aura is not None)
    v[18] = float(self_aura is not None)
    v[22] = dur / 5
    # periodic damage done by the applied aura (DoTs), one level deep
    au = enemy_aura or self_aura
    if au and au in gd.abilities:
        tick = 0
        for t in gd.abilities[au]["turns"]:
            for e in t["effects"]:
                if KIND.get(e["op"]) == "damage":
                    tick += e["params"].get("Points", 0)
        v[23] = tick / 50
    ns = len(state_index())
    v[24:24 + ns] = _state_vec(ab["states"])
    if au and au in gd.abilities:
        v[24 + ns:] = _state_vec(gd.abilities[au]["states"])
    return v
