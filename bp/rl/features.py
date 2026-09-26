"""Battle state -> fixed-size tensors for the network (always from the point of view of `side`).

Tokens:
  6 pet tokens   (own 3 then enemy 3)
  MAX_AURAS aura tokens (auras on pets, team auras, weather)
Integer features go through embeddings, float features are concatenated.
"""
from __future__ import annotations

import numpy as np

from ..sim import constants as C
from ..sim.data import game_data

MAX_AURAS = 16
N_PET_FLOAT = 24
N_AURA_FLOAT = 6

KEY_STATES = [C.S_STUNNED, C.S_UNTARGETTABLE, C.S_UNDERGROUND, C.S_FLYING, C.S_TURN_LOCK, C.S_SWAP_OUT_LOCK,
              C.S_DODGE, C.S_BLOCK_COUNT, C.S_UNKILLABLE, C.S_DMG_DEALT_PCT, C.S_DMG_TAKEN_PCT, C.S_RESILIENT]


def vocab_sizes():
    gd = game_data()
    return {"species": max(gd.species) + 1, "ability": max(gd.abilities) + 1}


def encode(battle, side):
    """Returns dict of numpy arrays: pet_int [6,5], pet_float [6,N_PET_FLOAT],
    aura_int [MAX_AURAS,2], aura_float [MAX_AURAS,N_AURA_FLOAT], aura_mask [MAX_AURAS], glob [4]."""
    pet_int = np.zeros((6, 5), dtype=np.int64)  # species, family+1, ability x3
    pet_float = np.zeros((6, N_PET_FLOAT), dtype=np.float32)
    aura_int = np.zeros((MAX_AURAS, 2), dtype=np.int64)  # aura ability id, holder slot+1 (0 = weather)
    aura_float = np.zeros((MAX_AURAS, N_AURA_FLOAT), dtype=np.float32)
    aura_mask = np.zeros(MAX_AURAS, dtype=np.float32)
    teams = [battle.teams[side], battle.teams[1 - side]]
    auras = []
    for ti, team in enumerate(teams):
        for pi in range(3):
            slot = ti * 3 + pi
            if pi >= len(team.pets):
                continue
            pet = team.pets[pi]
            pet_int[slot, 0] = pet.spec.species
            pet_int[slot, 1] = pet.family + 1
            for k in range(3):
                pet_int[slot, 2 + k] = pet.abilities[k] or 0
            f = pet_float[slot]
            f[0] = 1.0
            f[1] = pet.hp / max(1, pet.max_hp)
            f[2] = pet.max_hp / 2000
            f[3] = pet.power / 400
            f[4] = pet.speed / 400
            f[5] = pet.spec.level / 25
            f[6] = float(pet.dead)
            f[7] = float(team.active == pi)
            f[8] = float(pet.undead_used)
            f[9] = float(pet.mech_revived)
            f[10] = float(pet.channel is not None)
            for k in range(3):
                f[11 + k] = pet.cooldowns[k] / 5
            if team.active == pi and not pet.dead:
                for j, s in enumerate(KEY_STATES[:10]):
                    f[14 + j] = np.clip(pet.state(s) / (100 if s in (C.S_DODGE, C.S_DMG_DEALT_PCT, C.S_DMG_TAKEN_PCT) else 1), -2, 2)
            for a in pet.auras:
                auras.append((a, slot + 1))
        for a in team.auras:
            auras.append((a, 7 + ti))
    if battle.weather is not None:
        auras.append((battle.weather, 9))
    for i, (a, holder) in enumerate(auras[:MAX_AURAS]):
        aura_int[i, 0] = a.aid
        aura_int[i, 1] = holder
        aura_float[i, 0] = a.duration / 10 if a.duration >= 0 else -1
        aura_float[i, 1] = a.stacks / 3
        aura_float[i, 2] = float(a.caster is not None and a.caster.team.side == side)
        aura_float[i, 3] = getattr(a, "age", 0) / 10
        aura_mask[i] = 1
    glob = np.array([battle.round / 50, float(battle.needs_replacement(side)), 0, 0], dtype=np.float32)
    return {"pet_int": pet_int, "pet_float": pet_float, "aura_int": aura_int, "aura_float": aura_float,
            "aura_mask": aura_mask, "glob": glob}


def action_mask(battle, side):
    m = np.zeros(C.N_ACTIONS, dtype=np.float32)
    for a in battle.legal_actions(side):
        m[a] = 1
    return m
