"""Practice positions: part of the training games start right at a moment the network rarely meets.

  undead      an Undead pet that is about to die / in its "cannot die" round
  mech        a Mechanical pet just before / after its Failsafe
  critter_cc  a Critter facing a pet that can stun / put to sleep (the control does nothing)
  cc          a stunning pet against anything: after a stun the target is Resilient for a while

The teams are sampled as usual, then one pet is replaced so the situation can happen; both sides are
played by the scripted greedy bot (with some random moves) until it does, and from there the game
goes on as a normal training game. Nothing about the right move is told: the network plays the
position out and learns from the result like everywhere else.
"""
from __future__ import annotations

from functools import lru_cache

from ..sim import constants as C
from ..sim.data import PetSpec, game_data
from .bots import GreedyBot
from .features import PHASE_ROUND
from .pool import breeds_of, classes

TARGET_AURA_OPS = {178, 50, 52, 54, 168, 229, 230, 131, 172}
KINDS = ["undead", "mech", "critter_cc", "cc"]
_bot = GreedyBot()


@lru_cache(maxsize=1)
def _pools():
    gd = game_data()
    by_family = {}
    for c in classes():
        by_family.setdefault(gd.species[c[0]]["family"], []).append(c[0])
    cc = []  # (species, slot, option) whose ability stuns / puts the target to sleep
    for s in [c[0] for c in classes()]:
        for k, opts in enumerate(gd.species[s]["abilities"]):
            for j, aid in enumerate(opts[:2]):
                if _is_cc(aid):
                    cc.append((s, k, j))
    return by_family, cc


def _is_cc(aid):
    gd = game_data()
    for t in gd.abilities.get(aid, {}).get("turns", []):
        if t["type"] != 0:
            continue
        for e in t["effects"]:
            if e["op"] in TARGET_AURA_OPS:
                st = gd.abilities.get(e["aura"] or C.STUNNED_AURA, {}).get("states", {})
                if st.get(C.S_STUNNED) or st.get(C.S_TURN_LOCK):
                    return True
    return False


def _spec(rng, species, fixed=None):
    ch = [rng.choice("12") for _ in range(3)]
    if fixed is not None:
        ch[fixed[0]] = str(fixed[1] + 1)
    return PetSpec.from_choices(species, "".join(ch), breed=rng.choice(breeds_of(species)))


def _cc_ready(battle, side):
    pet = battle.teams[side].active_pet
    return any(a in C.A_ABILITY and _is_cc(pet.abilities[a]) for a in battle.legal_actions(side))


def _trigger(kind, b, rng):
    for side in (0, 1):
        p, foe = b.teams[side].active_pet, b.teams[1 - side].active_pet
        if p.dead or foe.dead:
            continue
        frac = p.hp / max(1, p.max_hp)
        if kind == "undead" and p.family == C.UNDEAD:
            if p.state(C.S_UNKILLABLE) > 0 or (not p.undead_used and frac < 0.3 and rng.random() < 0.5):
                return True
        if kind == "mech" and p.family == C.MECHANICAL:
            if (not p.mech_revived and frac < 0.35) or (p.mech_revived and rng.random() < 0.5):
                return True
        if kind == "critter_cc" and foe.family == C.CRITTER and _cc_ready(b, side):
            return True
        if kind == "cc" and _cc_ready(b, side) and rng.random() < 0.35:
            return True
    return False


def start(env, sampler, rng, kind=None, max_rounds=25):
    """Reset `env` into a practice position (falls back to a normal game start if it never happens)."""
    by_family, cc = _pools()
    kind = kind or rng.choice(KINDS)
    teams = [sampler.team(), sampler.team()]
    x = rng.randrange(2)
    lead = [rng.randrange(3), rng.randrange(3)]
    if kind == "undead":
        teams[x][0], lead[x] = _spec(rng, rng.choice(by_family[C.UNDEAD])), 0
    elif kind == "mech":
        teams[x][0], lead[x] = _spec(rng, rng.choice(by_family[C.MECHANICAL])), 0
    else:
        s, k, j = rng.choice(cc)
        teams[1 - x][0], lead[1 - x] = _spec(rng, s, (k, j)), 0
        if kind == "critter_cc":
            teams[x][0], lead[x] = _spec(rng, rng.choice(by_family[C.CRITTER])), 0
    for t in teams:  # a replaced pet may repeat a species already in the team
        seen = set()
        for i, p in enumerate(t):
            while p.species in seen:
                p = sampler.random_pet()
            t[i] = p
            seen.add(p.species)
    env.reset(teams)
    env.act({0: 3 + lead[0], 1: 3 + lead[1]})
    for _ in range(max_rounds):
        if env.done:
            env.reset()
            return
        if env.phase == PHASE_ROUND and _trigger(kind, env.battle, rng):
            return
        acts = {}
        for s in env.pending():
            legal = env.battle.legal_actions(s)
            acts[s] = rng.choice(legal) if rng.random() < 0.2 else _bot.act(env.battle, s, env.phase, rng)
        env.act(acts)
