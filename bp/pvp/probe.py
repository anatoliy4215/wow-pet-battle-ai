"""Exact "what does this button do right now" hints for the network, computed by the simulator itself.

For every ability the active pet can use this round, two copies of the battle play one round with
average dice (no crit, middle damage roll; every accuracy roll lands but its damage is scaled by its
chance, so a 50% extra hit counts half): once with the ability, once passing.
The opponent passes in both, so only the ability's own effect is measured; the difference is what
the ability does this round:

  0  probed            1 = the row below is filled (0: not usable now / not a normal round)
  1  enemy damage      expected health the enemy active loses, / its max health (a lost life counts
                       as all its current health: death, Undead's return, Mechanical's Failsafe)
  2  takes a life      1 if the enemy active dies or uses up its Undead / Failsafe life when the
                       ability's hits land (for abilities with accuracy rolls: a copy where all hit)
  3  own health        own active's health change / max health (heals +, self-damage -)
  4  immune            the enemy was immune to something this ability does (stun on a Critter,
                       control after Resilient, ...)
  5  unreachable       an attack could not reach the target (underground, flying, ...)
  6  blocked           a hit was absorbed by a shield / block
  7  hit chance        accuracy against this target (own accuracy, its dodge); 1 = sure hit

Because the numbers come from the same code that is checked against the combat logs, the hints are
as right as the simulator. Nothing hidden is used: the opponent only passes (its unknown abilities
never run), its health / stats / auras are visible in the game too. Checks: tests/test_probe.py.
"""
from __future__ import annotations

import numpy as np

from ..sim import constants as C
from ..sim.clone import fast_clone
from ..sim.data import game_data

N_PROBE = 8
PASS = C.A_PASS


def _life(p):
    return p.hp, p.dead, p.undead_used, p.mech_revived


def _loss(before, pet):
    hp, dead, und, mech = before
    took = (pet.dead and not dead) or (pet.undead_used and not und) or (pet.mech_revived and not mech)
    return (hp if took else hp - pet.hp), took


def _run(battle, side, action, expect=True):
    c = fast_clone(battle, 0)
    c.logging = False
    c.force_hit, c.force_crit, c.force_variance = True, False, 0.0
    c.probe_expect = expect
    c.probe_counts = {}
    me, en = c.teams[side].active_pet, c.teams[1 - side].active_pet
    b_me, b_en = _life(me), _life(en)
    acts = [PASS, PASS]
    acts[side] = action
    c.step(*acts)
    loss, took = _loss(b_en, en)
    own = (-b_me[0] if me.dead else me.hp - b_me[0])
    return loss, took, own, c.probe_counts


def hit_chance(battle, pet, target, aid):
    """Accuracy of the ability's first rolled effect against `target` (same formula as ops.roll_hit)."""
    ab = game_data().abilities.get(aid, {})
    turns = [t for t in ab.get("turns", []) if t["type"] == 0]
    acc = None
    for e in turns[0]["effects"] if turns else []:
        if "Accuracy" in e["params"]:
            acc = e["params"]["Accuracy"]
            break
    if acc is None:
        return 1.0
    acc = acc + pet.state(C.S_ACCURACY) - target.state(C.S_DODGE)
    lvl = target.spec.level - pet.spec.level
    if lvl > 0:
        acc -= lvl * C.LEVEL_HIT_PENALTY
    return float(min(1.0, max(0.0, acc / 100)))


def probe(battle, side, legal=None):
    """(3, N_PROBE) rows for the active pet's ability slots of `side` (zeros where not usable)."""
    out = np.zeros((3, N_PROBE), dtype=np.float32)
    me, en = battle.teams[side].active_pet, battle.teams[1 - side].active_pet
    if battle.over or me.dead or en.dead or me.channel is not None:
        return out
    if legal is None:
        legal = battle.legal_actions(side)
    slots = [a for a in legal if a in C.A_ABILITY]
    if not slots:
        return out
    base_loss, base_took, base_own, base_c = _run(battle, side, PASS)
    mhp_en, mhp_me = max(1, en.max_hp), max(1, me.max_hp)
    en_side = 1 - side
    base_forced = None
    for sl in slots:
        loss, took, own, cnt = _run(battle, side, sl)
        f = out[sl]
        f[0] = 1.0
        f[1] = min(3.0, max(-1.0, (loss - base_loss) / mhp_en))
        if cnt.get(("rolled", en_side)):  # some hit may miss: would it kill if they all land?
            if base_forced is None:
                base_forced = _run(battle, side, PASS, expect=False)[1]
            took = _run(battle, side, sl, expect=False)[1]
            f[2] = float(took and not base_forced)
        else:
            f[2] = float(took and not base_took)
        f[3] = min(1.0, max(-1.0, (own - base_own) / mhp_me))
        f[4] = float(cnt.get(("immune", en_side), 0) > base_c.get(("immune", en_side), 0))
        f[5] = float(cnt.get(("unreachable", en_side), 0) > base_c.get(("unreachable", en_side), 0))
        f[6] = float(cnt.get(("blocked", en_side), 0) > base_c.get(("blocked", en_side), 0))
        f[7] = hit_chance(battle, me, en, me.abilities[sl])
    return out
