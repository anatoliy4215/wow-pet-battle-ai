"""Tactical exam: does the network know the basic tricks?  Positions are generated procedurally from
random battles, mostly as counterfactual pairs (same position with and without the key detail), so
the score measures the reaction to that detail and not general preferences.

  finish      enemy at 1 hp, a sure hit is available            -> P(sure damaging hit), want ~1
  untarget    enemy is underground / in the air / phased and we are faster: our attack would hit
              nothing -> drop in P(damage) vs the same position without the state, want > 0
  immortal    enemy undead in its last "cannot die" round: damage is wasted -> drop in P(damage)
  family      the same position twice, only the enemy family differs so that one attack is strong in the
              first and weak in the second -> rise in P(that attack), want > 0
  cc_critter  we have an ability that only stuns / puts to sleep; the same position with the enemy turned
              into a Critter (immune to it) -> drop in P(that ability), want > 0

python -m bp.pvp.tactics [--ckpt checkpoints/pvp/main.pt]
"""
from __future__ import annotations

import random
from functools import lru_cache

import numpy as np
import torch

from ..sim import constants as C
from ..sim import ops as OPS
from ..sim.data import game_data
from ..sim.targets import KIND
from .abilityinfo import descriptor
from .env import PvPEnv
from .features import KEYS, PHASE_ROUND, action_mask, effectiveness, encode
from .model import masked
from .pool import TeamSampler

UNTARGET_AURAS = {340: "Underground (Burrow)", 341: "Flying (Lift-Off)", 830: "Underwater (Dive)", 839: "Leaping", 261: "Invisible (Fading)"}


@lru_cache(maxsize=None)
def first_turn_damage(aid):
    ab = game_data().abilities[aid]
    main = [t for t in ab["turns"] if t["type"] == 0]
    if not main:
        return 0
    return sum(e["params"].get("Points", 0) for e in main[0]["effects"]
               if KIND.get(e["op"]) == "damage" and not e["params"].get("IsPeriodic"))


def _damaging(pet, legal):
    return [a for a in legal if a in C.A_ABILITY and descriptor(pet.abilities[a])[13] > 0]


def _random_positions(n, seed):
    """Mid-battle positions from greedy-vs-greedy games (both pets alive, a normal round)."""
    from .bots import GreedyBot
    rng = random.Random(seed)
    env = PvPEnv(TeamSampler(seed=seed), seed=seed + 1)
    bot = GreedyBot()
    made = 0
    while made < n:
        env.reset()
        env.act({0: 3 + rng.randrange(3), 1: 3 + rng.randrange(3)})
        stop = rng.randrange(0, 6)
        for _ in range(stop):
            if env.done:
                break
            env.act({s: bot.act(env.battle, s, env.phase, rng) for s in env.pending()})
        if env.done or env.phase != PHASE_ROUND:
            continue
        made += 1
        yield env.battle


def build_suite(n=300, seed=7):
    from .search import clone
    gd = game_data()
    from .probe import probe
    suite = {"finish": [], "untarget": [], "immortal": [], "family2": [], "cc_critter": []}
    for b in _random_positions(n * 40, seed):
        me, en = b.teams[0].active_pet, b.teams[1].active_pet
        legal = b.legal_actions(0)
        dmg = _damaging(me, legal)
        if not dmg:
            continue
        # finish: enemy at 1 hp, at least one sure first-turn hit
        sure = [a for a in dmg if first_turn_damage(me.abilities[a]) > 0 and descriptor(me.abilities[a])[15] >= 1]
        if sure and len(suite["finish"]) < n and en.family != C.UNDEAD and not b.teams[1].auras:
            c = clone(b, 0)
            c.teams[1].active_pet.hp = 1
            suite["finish"].append((encode(c, 0), action_mask(c, 0), sure))
        # untarget / immortal: counterfactual pairs
        if len(suite["untarget"]) < n and me.speed > en.speed:
            c = clone(b, 0)
            aid = random.Random(len(suite["untarget"])).choice(list(UNTARGET_AURAS))
            if aid in gd.abilities:
                OPS.add_aura(c, c.teams[1].active_pet, aid, c.teams[1].active_pet, 1)
                if c.teams[1].active_pet.state(C.S_UNTARGETTABLE, c) > 0:
                    suite["untarget"].append(((encode(b, 0), action_mask(b, 0)), (encode(c, 0), action_mask(c, 0)), dmg))
        if len(suite["immortal"]) < n and en.family == C.UNDEAD:
            c = clone(b, 0)
            OPS.add_aura(c, c.teams[1].active_pet, C.UNDEAD_AURA, c.teams[1].active_pet, 1)
            c.teams[1].active_pet.undead_used = True
            suite["immortal"].append(((encode(b, 0), action_mask(b, 0)), (encode(c, 0), action_mask(c, 0)), dmg))
        # family: flip the enemy family so that one attack goes from strong to weak, everything else neutral
        if len(suite["family2"]) < n and len(dmg) >= 2:
            a = dmg[0]
            fam_a = gd.abilities[me.abilities[a]]["family"]
            if fam_a >= 0:
                strong_vs, weak_vs = C.STRONG[fam_a], C.WEAK[fam_a]
                others = [gd.abilities[me.abilities[x]]["family"] for x in dmg[1:]]
                clash = {strong_vs, weak_vs}
                if not any(o in clash or C.STRONG.get(o) in clash or C.WEAK.get(o) in clash for o in others):
                    c1, c2 = clone(b, 0), clone(b, 0)
                    c1.teams[1].active_pet.family = strong_vs
                    c2.teams[1].active_pet.family = weak_vs
                    suite["family2"].append(((encode(c1, 0), action_mask(c1, 0)), (encode(c2, 0), action_mask(c2, 0)), [a]))
        # cc_critter: an ability that does nothing to a Critter (the simulator says: immune, no damage)
        if len(suite["cc_critter"]) < n and en.family != C.CRITTER:
            c = clone(b, 0)
            c.teams[1].active_pet.family = C.CRITTER
            h0, h1 = probe(b, 0, legal), probe(c, 0, legal)
            cc = [a for a in legal if a in C.A_ABILITY and h1[a, 4] and h1[a, 1] <= 0 and h1[a, 3] <= 0 and not h0[a, 4]]
            if cc:
                suite["cc_critter"].append(((encode(b, 0), action_mask(b, 0)), (encode(c, 0), action_mask(c, 0)), cc))
        if all(len(v) >= n for v in suite.values()):
            break
    return suite


def _probs(net, pairs, device="cuda"):
    obs = [p[0] for p in pairs]
    masks = [p[1] for p in pairs]
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(obs), 1024):
            x = {k: torch.as_tensor(np.stack([o[k] for o in obs[i:i + 1024]]), device=device) for k in KEYS}
            m = torch.as_tensor(np.stack(masks[i:i + 1024]), device=device)
            out.append(torch.softmax(masked(net(x)[0], m), -1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, C.N_ACTIONS))


def evaluate(net, suite):
    res = {}
    if suite["finish"]:
        p = _probs(net, [(o, m) for o, m, _ in suite["finish"]])
        res["finish"] = float(np.mean([p[i, g].sum() for i, (_, _, g) in enumerate(suite["finish"])]))
    for key in ("untarget", "immortal", "family2", "cc_critter"):
        items = suite[key]
        if items:
            p0 = _probs(net, [x[0] for x in items])
            p1 = _probs(net, [x[1] for x in items])
            res[key] = float(np.mean([p0[i, d].sum() - p1[i, d].sum() for i, (_, _, d) in enumerate(items)]))
    res["n"] = {k: len(v) for k, v in suite.items()}
    return res


def main():
    import argparse
    from pathlib import Path
    from .teambuilder import load_net
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "pvp" / "main.pt"))
    args = ap.parse_args()
    net = load_net(args.ckpt)
    print(evaluate(net, build_suite()))


if __name__ == "__main__":
    main()
