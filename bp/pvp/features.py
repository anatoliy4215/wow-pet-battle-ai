"""PvP observation: battle state -> fixed-size arrays, from one side's point of view.

Only what a player can see in the game is encoded (so the model can later run in-game, and it can
never "cheat"):
  * own pets: everything (abilities, cooldowns, stats, auras)
  * enemy pets: family, health, power, speed, auras; abilities only once they were used - for an
    unrevealed slot both possible abilities are given, flagged as "unknown"
  * weather, team-pad objects, round number, phase (lead pick / normal round / replacement)

Tokens: 1 global + 6 pets + 27 ability slots (own 3x3 chosen, enemy 3x3x2 candidates) + MAX_AURAS.
Static ability descriptors (abilityinfo.py) are looked up inside the model by ability id.

Hints (appended 2026-09-25; older networks read only the first 40 pet / 10 ability columns):
  pet   40 has a second life left (Undead return / Mechanical Failsafe not used yet)
        41 effective health: (health + the Failsafe's 20% if still unused) / max health
        42 immune to control now (Critter, or Resilient after a stun)
  own active pet's abilities 10-17: what the ability does this round, measured by the simulator
        (bp/pvp/probe.py: damage, kill, own heal, immune, unreachable, blocked, hit chance)
"""
from __future__ import annotations

import numpy as np

from ..sim import constants as C
from ..sim.data import game_data
from .abilityinfo import descriptor
from .probe import N_PROBE, probe

MAX_AURAS = 20
N_ABIL = 27
N_GLOB = 12
N_PET = 43
N_ABF = 10 + N_PROBE
N_AUF = 8
PHASE_LEAD, PHASE_ROUND, PHASE_REPLACE = 0, 1, 2

PET_STATES = [C.S_TURN_LOCK, C.S_STUNNED, C.S_DODGE, C.S_UNTARGETTABLE, C.S_UNDERGROUND, C.S_FLYING,
              C.S_BLOCK_COUNT, C.S_UNKILLABLE, C.S_DMG_DEALT_PCT, C.S_DMG_TAKEN_PCT, C.S_SPEED_PCT, C.S_ACCURACY,
              C.S_CRIT, C.S_HEAL_TAKEN_PCT, C.S_FLAT_DMG_TAKEN, C.S_FLAT_DMG_DEALT, C.S_SWAP_OUT_LOCK,
              C.S_SWAP_IN_LOCK, C.S_POISONED, C.S_BURNING, C.S_CHILLED, C.S_WEBBED, C.S_BLEEDING, C.S_BLIND,
              C.S_RESILIENT]
PCT = {C.S_DODGE, C.S_DMG_DEALT_PCT, C.S_DMG_TAKEN_PCT, C.S_SPEED_PCT, C.S_ACCURACY, C.S_CRIT, C.S_HEAL_TAKEN_PCT}
assert 15 + len(PET_STATES) + 3 == N_PET


def all_states(pet, battle):
    """Every state value of `pet` at once (same rules as Pet.state, one pass instead of one per state)."""
    v = dict(pet.states)
    for s, x in game_data().species[pet.spec.species]["states"].items():
        if s not in (C.S_POWER, C.S_SPEED, C.S_STAMINA):
            v[s] = v.get(s, 0) + x
    for a in pet.auras:
        for s, x in a.ability["states"].items():
            v[s] = v.get(s, 0) + x * a.stacks
    if pet.team.active_pet is pet:
        for a in pet.team.auras:
            for s, x in a.ability["states"].items():
                v[s] = v.get(s, 0) + x
    w = battle.weather
    if w is not None:
        for s, x in w.ability["states"].items():
            if not (pet.family == C.ELEMENTAL and x < 0 and s not in C.WEATHER_STATES):
                v[s] = v.get(s, 0) + x
    if pet.family == C.BEAST and pet.hp * 2 < pet.max_hp:
        v[C.S_DMG_DEALT_PCT] = v.get(C.S_DMG_DEALT_PCT, 0) + C.BEAST_BONUS
    if pet.family == C.FLYING and pet.hp * 2 > pet.max_hp:
        v[C.S_SPEED_PCT] = v.get(C.S_SPEED_PCT, 0) + C.FLYING_SPEED_PCT
    return v


def _clamp(x, lo=-3.0, hi=3.0):
    return lo if x < lo else hi if x > hi else x


def effectiveness(ab_family, target_family):
    if ab_family is None or ab_family < 0 or target_family is None:
        return 0.0
    if C.STRONG.get(ab_family) == target_family:
        return 1.0
    if C.WEAK.get(ab_family) == target_family:
        return -1.0
    return 0.0


def _ability_float(f, pet, power, slot, aid, known, opp, legal_now):
    """opp: (family, max hp, hp) of the pet this ability would hit, or None."""
    ab = game_data().abilities.get(aid, {})
    d = descriptor(aid)
    f[0] = 1.0
    f[1] = float(known)
    f[2] = pet.cooldowns[slot] / 5 if known else 0.0
    f[3] = float(legal_now)
    if opp is not None:
        eff = effectiveness(ab.get("family"), opp[0])
        f[4] = eff
        mult = 1.5 if eff > 0 else (C.WEAK_MULT if eff < 0 else 1.0)
        raw = d[13] * 50 * (1 + power / 20) * mult  # direct damage points, one use
        f[5] = min(3.0, raw / max(1, opp[1]))
        f[6] = min(3.0, raw / max(1, opp[2]))
    f[7] = float(pet.channel is not None and pet.channel[0] == slot)
    f[8] = float(pet.last_ability == aid)
    f[9] = float(slot in pet.revealed)


def encode(battle, side, phase=PHASE_ROUND, hide=True):
    glob = np.zeros(N_GLOB, dtype=np.float32)
    pet_int = np.zeros(6, dtype=np.int64)          # family + 1 (0 = empty)
    pet_float = np.zeros((6, N_PET), dtype=np.float32)
    ab_int = np.zeros((N_ABIL, 3), dtype=np.int64)  # ability id, owner slot 0-5, ability slot 0-2
    ab_float = np.zeros((N_ABIL, N_ABF), dtype=np.float32)
    ab_mask = np.zeros(N_ABIL, dtype=np.float32)
    aura_int = np.zeros((MAX_AURAS, 2), dtype=np.int64)  # aura id, holder (0-5 pets, 6/7 own/enemy pad, 8 weather)
    aura_float = np.zeros((MAX_AURAS, N_AUF), dtype=np.float32)
    aura_mask = np.zeros(MAX_AURAS, dtype=np.float32)
    gd = game_data()
    me, en = battle.teams[side], battle.teams[1 - side]
    lead = phase == PHASE_LEAD
    my_act = None if lead else me.active_pet
    en_act = None if lead else en.active_pet
    legal = set(battle.legal_actions(side)) if phase == PHASE_ROUND else set()

    # derived stats once per pet (each property walks all auras)
    stats = {id(p): (p.max_hp, p.power, p.speed) for t in (me, en) for p in t.pets}
    auras = []
    for ti, team in enumerate((me, en)):
        for pi, pet in enumerate(team.pets):
            s = ti * 3 + pi
            mhp, pw, spd = stats[id(pet)]
            pet_int[s] = pet.family + 1
            f = pet_float[s]
            f[0] = 1.0
            f[1] = pet.hp / max(1, mhp)
            f[2] = mhp / 2000
            f[3] = pw / 400
            f[4] = spd / 400
            f[5] = float(pet.dead)
            f[6] = float(not lead and team.active == pi)
            f[7] = float(pet.undead_used)
            f[8] = float(pet.mech_revived)
            f[9] = float(pet.channel is not None)
            f[10] = pet.hp / 2000
            f[11] = min(10, battle.round - pet.swapped_in) / 10 if not lead and team.active == pi else 0.0
            opp = en_act if ti == 0 else my_act
            if opp is not None:
                ospd = stats[id(opp)][2]
                f[12] = float(spd > ospd) - float(spd < ospd)
                f[13] = effectiveness(pet.family, opp.family)   # my family's attacks vs their active
                f[14] = effectiveness(opp.family, pet.family)   # their family's attacks vs me
            if not pet.dead:
                sv = all_states(pet, battle)
                for j, st in enumerate(PET_STATES):
                    v = sv.get(st, 0)
                    f[15 + j] = _clamp(v / 100 if st in PCT else v)
                natural = sv.get(C.S_TYPE_OVERRIDE, 0) == 0  # a polymorphed pet has no racial
                undead_life = pet.family == C.UNDEAD and not pet.undead_used and natural
                mech_life = pet.family == C.MECHANICAL and not pet.mech_revived and natural
                f[40] = float(undead_life or mech_life)
                f[41] = min(2.0, (pet.hp + (0.2 * mhp if mech_life else 0)) / max(1, mhp))
                f[42] = float(pet.family == C.CRITTER or sv.get(C.S_RESILIENT, 0) > 0)
            for a in pet.auras:
                auras.append((a, s))
        for a in team.auras:
            auras.append((a, 6 + ti))
    if battle.weather is not None:
        auras.append((battle.weather, 8))

    # ability slots: own chosen 3 per pet (0-8), enemy 2 candidates per slot (9-26)
    def target(p):
        return None if p is None or p.dead else (p.family, stats[id(p)][0], p.hp)

    t_en, t_me = target(en_act), target(my_act)
    hints = probe(battle, side, sorted(legal)) if phase == PHASE_ROUND and my_act is not None else None
    k = 0
    for pi, pet in enumerate(me.pets):
        pw = stats[id(pet)][1]
        for sl in range(3):
            aid = pet.abilities[sl]
            if aid:
                ab_int[k] = (aid, pi, sl)
                act_now = pet is my_act and sl in legal
                _ability_float(ab_float[k], pet, pw, sl, aid, True, t_en, act_now)
                if hints is not None and pet is my_act:
                    ab_float[k, 10:] = hints[sl]
                ab_mask[k] = 1
            k += 1
    for pi, pet in enumerate(en.pets):
        opts = gd.species[pet.spec.species]["abilities"]
        pw = stats[id(pet)][1]
        for sl in range(3):
            aid = pet.abilities[sl]
            if not aid:
                k += 2
                continue
            if not hide or sl in pet.revealed:
                ab_int[k] = (aid, 3 + pi, sl)
                _ability_float(ab_float[k], pet, pw, sl, aid, True, t_me, False)
                ab_mask[k] = 1
            else:
                for j, cand in enumerate(opts[sl][:2]):
                    ab_int[k + j] = (cand, 3 + pi, sl)
                    _ability_float(ab_float[k + j], pet, pw, sl, cand, False, t_me, False)
                    ab_mask[k + j] = 1
            k += 2

    for i, (a, holder) in enumerate(auras[:MAX_AURAS]):
        aura_int[i] = (a.aid, holder)
        f = aura_float[i]
        f[0] = a.duration / 5 if a.duration >= 0 else 0.0
        f[1] = float(a.duration < 0)
        f[2] = a.stacks / 3
        f[3] = float(a.caster is not None and a.caster.team.side == side)
        f[4] = min(10, a.age) / 10
        f[5] = a.blocks_left / 3
        f[6] = float(a.next_attack)
        f[7] = 1.0
        aura_mask[i] = 1

    glob[phase] = 1.0
    glob[3] = battle.round / 30
    glob[4] = len(me.alive) / 3
    glob[5] = len(en.alive) / 3
    if my_act is not None and en_act is not None:
        glob[6] = float(stats[id(my_act)][2] > stats[id(en_act)][2])
        glob[7] = float(stats[id(my_act)][2] == stats[id(en_act)][2])
        glob[8] = float(en_act.channel is not None)
        glob[9] = float(my_act.state(C.S_TURN_LOCK, battle) > 0)
    glob[10] = float(battle.weather is not None)
    glob[11] = float(battle.needs_replacement(1 - side))
    return {"glob": glob, "pet_int": pet_int, "pet_float": pet_float, "ab_int": ab_int, "ab_float": ab_float,
            "ab_mask": ab_mask, "aura_int": aura_int, "aura_float": aura_float, "aura_mask": aura_mask}


KEYS = ["glob", "pet_int", "pet_float", "ab_int", "ab_float", "ab_mask", "aura_int", "aura_float", "aura_mask"]


def action_mask(battle, side, phase=PHASE_ROUND):
    m = np.zeros(C.N_ACTIONS, dtype=np.float32)
    if phase == PHASE_LEAD:
        m[3:3 + len(battle.teams[side].pets)] = 1
        return m
    for a in battle.legal_actions(side):
        m[a] = 1
    return m


def loadout_truth(battle, side):
    """Per enemy (pet, slot): 1 if the enemy picked the second option, and whether it is still hidden."""
    gd = game_data()
    y = np.zeros(9, dtype=np.float32)
    hidden = np.zeros(9, dtype=np.float32)
    for pi, pet in enumerate(battle.teams[1 - side].pets):
        opts = gd.species[pet.spec.species]["abilities"]
        for sl in range(3):
            aid = pet.abilities[sl]
            if aid and len(opts[sl]) >= 2:
                y[pi * 3 + sl] = float(aid == opts[sl][1])
                hidden[pi * 3 + sl] = float(sl not in pet.revealed)
    return y, hidden
