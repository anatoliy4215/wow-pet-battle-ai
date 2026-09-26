"""Fixed benchmark for the dashboard: the same team pairs every time (both seatings), so the numbers
are comparable across the whole run - unlike the win rates inside training, where teams, sides and
opponents keep changing.

  vs greedy   current network vs the greedy bot
  vs early    current network vs a frozen early version (league snapshot of update 20)
One game is recorded move by move ("how does it play right now") for the dashboard.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from ..sim.data import game_data
from .bots import GreedyBot
from .env import PvPEnv
from .features import KEYS, PHASE_LEAD, PHASE_REPLACE
from .model import masked
from .pool import BREED_NAMES, TeamSampler

ACTION_NAMES = {3: "выход/смена на 1-го", 4: "выход/смена на 2-го", 5: "выход/смена на 3-го", 6: "пропуск хода"}


def make_pairs(n=200, seed=4242):
    ts = TeamSampler(seed=seed, meta_pet_frac=0.5, meta_team_frac=0.3)
    pairs = [(ts.team(), ts.team()) for _ in range(n)]
    return pairs + [(b, a) for a, b in pairs]


def _describe_action(battle, side, phase, a):
    gd = game_data()
    team = battle.teams[side]
    if a < 3 and phase not in (PHASE_LEAD, PHASE_REPLACE):
        pet = team.active_pet
        return f"{pet.name}: {gd.abilities[pet.abilities[a]]['name']}"
    if 3 <= a <= 5:
        who = team.pets[a - 3].name
        return f"выходит {who}" if phase in (PHASE_LEAD, PHASE_REPLACE) else f"смена на {who}"
    return "пропуск хода"


def _team_info(team):
    gd = game_data()
    return [{"name": p.name, "breed": BREED_NAMES.get(p.spec.breed, "?"),
             "abilities": [gd.abilities[a]["name"] for a in p.abilities if a]} for p in team.pets]


def play(net, opponent, pairs, seed=0, record=0, device="cuda"):
    """Our network plays side 0 of every pair. opponent: 'greedy' or a network.
    -> (score in [0, 1], recorded game of pair `record` or None)."""
    rng = random.Random(seed)
    bot = GreedyBot()
    envs = []
    for i, (ta, tb) in enumerate(pairs):
        e = PvPEnv(None, seed=seed * 7919 + i)
        e.reset((ta, tb))
        envs.append(e)
    rec = None
    if record is not None:
        b = envs[record].battle
        rec = {"teams": [_team_info(b.teams[0]), _team_info(b.teams[1])], "rounds": []}
    players = [net, opponent]
    while True:
        live = [(i, e, s) for i, e in enumerate(envs) if not e.done for s in e.pending()]
        if not live:
            break
        acts, values = {}, {}
        for side, pl in enumerate(players):
            mine = [(i, e, s) for i, e, s in live if s == side]
            if not mine:
                continue
            if pl == "greedy":
                for i, e, s in mine:
                    acts.setdefault(i, {})[s] = bot.act(e.battle, s, e.phase, rng)
                continue
            obs, masks = zip(*[e.observe(s) for _, e, s in mine])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = {k: torch.as_tensor(np.stack([o[k] for o in obs]), device=device) for k in KEYS}
                logits, v, _, _ = pl(x)
                a = torch.distributions.Categorical(
                    logits=masked(logits, torch.as_tensor(np.stack(masks), device=device))).sample().cpu().numpy()
                v = v.float().cpu().numpy()
            for (i, e, s), ai, vi in zip(mine, a, v):
                acts.setdefault(i, {})[s] = int(ai)
                if s == 0:
                    values[i] = float(vi)
        for i, act in acts.items():
            e = envs[i]
            if rec is not None and i == record:
                b = e.battle
                rec["rounds"].append({
                    "round": b.round + (0 if e.phase in (PHASE_LEAD, PHASE_REPLACE) else 1),
                    "phase": int(e.phase),
                    "me": _describe_action(b, 0, e.phase, act[0]) if 0 in act else None,
                    "opp": _describe_action(b, 1, e.phase, act[1]) if 1 in act else None,
                    "win": (values[i] + 1) / 2 if i in values else None,
                })
            e.act(act)
            if rec is not None and i == record:
                b = e.battle
                rec["rounds"][-1]["hp"] = [[round(p.hp / max(1, p.max_hp), 3) for p in t.pets] for t in b.teams]
    if rec is not None:
        rec["winner"] = envs[record].winner
    score = float(np.mean([1.0 if e.winner == 0 else 0.5 if e.winner == -1 else 0.0 for e in envs]))
    return score, rec
