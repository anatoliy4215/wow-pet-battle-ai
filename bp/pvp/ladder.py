"""Elo ladder of saved league versions: every pair plays the same random team pairs (both seatings),
ratings are fitted Bradley-Terry style.  The greedy bot is included as the fixed anchor (Elo 1000).

  python -m bp.pvp.ladder [--every 100] [--games 200]   -> logs/pvp_ladder.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .bots import GreedyBot
from .env import PvPEnv
from .features import KEYS
from .model import masked
from .pool import TeamSampler
from .teambuilder import load_net

ROOT = Path(__file__).resolve().parents[2]


def play(players, teams, seed=0, device="cuda"):
    """players: (a, b) each a net or 'greedy'; teams: list of (team_a, team_b). -> score of a in [0, 1]."""
    rng = random.Random(seed)
    envs = []
    for i, (ta, tb) in enumerate(teams):
        e = PvPEnv(None, seed=seed * 104729 + i)
        e.reset((ta, tb))
        envs.append(e)
    bot = GreedyBot()
    while True:
        live = [(e, s) for e in envs if not e.done for s in e.pending()]
        if not live:
            break
        acts = {}
        for pi, pl in enumerate(players):
            mine = [(e, s) for e, s in live if s == pi]
            if not mine:
                continue
            if pl == "greedy":
                for e, s in mine:
                    acts.setdefault(id(e), (e, {}))[1][s] = bot.act(e.battle, s, e.phase, rng)
                continue
            obs, masks = zip(*[e.observe(s) for e, s in mine])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = {k: torch.as_tensor(np.stack([o[k] for o in obs]), device=device) for k in KEYS}
                logits = pl(x)[0]
                a = torch.distributions.Categorical(
                    logits=masked(logits, torch.as_tensor(np.stack(masks), device=device))).sample().cpu().numpy()
            for (e, s), ai in zip(mine, a):
                acts.setdefault(id(e), (e, {}))[1][s] = int(ai)
        for e, act in acts.values():
            e.act(act)
    return float(np.mean([1.0 if e.winner == 0 else 0.5 if e.winner == -1 else 0.0 for e in envs]))


def fit_elo(names, results, anchor="greedy", iters=2000):
    """results: {(i, j): (score_i, n)} -> Elo per name, anchor fixed at 1000."""
    r = np.zeros(len(names))
    for _ in range(iters):
        grad = np.zeros(len(names))
        for (i, j), (s, n) in results.items():
            p = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))
            grad[i] += n * (s - p)
            grad[j] -= n * (s - p)
        r += 0.5 * grad / max(1, sum(n for _, n in results.values())) * 400
        r -= r[names.index(anchor)] - 1000
    return {n: float(x) for n, x in zip(names, r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "checkpoints" / "pvp" / "league"))
    ap.add_argument("--every", type=int, default=100, help="use every N-th update's snapshot")
    ap.add_argument("--games", type=int, default=200)
    args = ap.parse_args()
    snaps = sorted(Path(args.dir).glob("snap_*.pt"))
    snaps = [p for p in snaps if int(p.stem[5:]) % args.every == 0] or snaps[-1:]
    names = ["greedy"] + [f"v{int(p.stem[5:])}" for p in snaps]
    players = ["greedy"] + [load_net(p) for p in snaps]
    ts = TeamSampler(seed=99)
    pairs = [(ts.team(), ts.team()) for _ in range(args.games // 2)]
    teams = pairs + [(b, a) for a, b in pairs]
    results = {}
    for i, j in itertools.combinations(range(len(names)), 2):
        s = play((players[i], players[j]), teams, seed=i * 1000 + j)
        results[(i, j)] = (s, len(teams))
        print(f"{names[i]} vs {names[j]}: {s:.2f}", flush=True)
    elo = fit_elo(names, results)
    out = {"elo": elo, "pairs": {f"{names[i]}|{names[j]}": s for (i, j), (s, _) in results.items()}}
    (ROOT / "logs" / "pvp_ladder.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for n in names:
        print(f"{n:>8s} {elo[n]:7.0f}")
    _ = math


if __name__ == "__main__":
    main()
