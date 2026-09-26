"""Find the best teams from the player's collection against a tamer, using the trained policy.

Usage:  python -m bp.rl.evaluate --ckpt checkpoints/ppo_ach.pt --tamer 524 [--family 1] [--teams 300] [--games 20]
        python -m bp.rl.evaluate --ckpt ... --list            (list achievement tamers)
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from ..sim import constants as C
from ..sim.data import game_data
from .env import PetBattleEnv, ScenarioSampler, achievement_tasks
from .features import vocab_sizes
from .model import PetNet
from .ppo import KEYS


def load_policy(path, device):
    vs = vocab_sizes()
    net = PetNet(vs["species"], vs["ability"]).to(device)
    net.load_state_dict(torch.load(path, map_location=device)["model"])
    net.eval()
    return net


def play_batch(net, envs, device, greedy=True):
    """Play all envs to the end; returns list of wins."""
    obs = [e.observe() for e in envs]
    done = [False] * len(envs)
    wins = [False] * len(envs)
    while not all(done):
        idx = [i for i, d in enumerate(done) if not d]
        o = {k: torch.as_tensor(np.stack([obs[i][0][k] for i in idx]), device=device) for k in KEYS}
        m = torch.as_tensor(np.stack([obs[i][1] for i in idx]), device=device)
        with torch.no_grad():
            logits, _ = net(o, m)
        acts = logits.argmax(1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        for j, i in enumerate(idx):
            obs[i], r, d = envs[i].step(int(acts[j]))
            if d:
                done[i] = True
                wins[i] = envs[i].battle.winner == 0
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ppo_ach.pt")
    ap.add_argument("--tamer", type=int)
    ap.add_argument("--family", type=int, default=None)
    ap.add_argument("--teams", type=int, default=200)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    gd = game_data()
    if args.list:
        for fam, tid in achievement_tasks():
            t = gd.tamers[tid]
            print(tid, t["name"], "|", t["zone"], "|", C.FAMILY_NAMES[fam] if fam is not None else "any")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_policy(args.ckpt, device)
    sampler = ScenarioSampler(seed=0)
    pool = sampler._pool(args.family)
    rng = random.Random(0)
    results = []
    for _ in range(args.teams):
        members = rng.sample(pool, 3)
        choices = ["".join(rng.choice("12") for _ in range(3)) for _ in members]
        from ..sim.data import PetSpec
        team = [PetSpec.from_choices(x["species"], c, level=x["level"], quality=x["quality"], breed=x["breed"] or 3,
                                     stats=tuple(x["stats"])) for (_, x), c in zip(members, choices)]
        envs = []
        for g in range(args.games):
            e = PetBattleEnv(sampler, seed=g)
            e.reset((args.tamer, team))
            envs.append(e)
        wins = play_batch(net, envs, device)
        results.append((np.mean(wins), team))
    results.sort(key=lambda r: -r[0])
    t = gd.tamers[args.tamer]
    print(f"Tamer: {t['name']} ({t['zone']}), family: {C.FAMILY_NAMES[args.family] if args.family is not None else 'any'}")
    for wr, team in results[:args.top]:
        desc = []
        for s in team:
            abil = "/".join(gd.abilities[a]["name"] for a in s.abilities if a)
            desc.append(f"{gd.species[s.species]['name']} [{abil}]")
        print(f"  {wr:5.0%}  " + "  +  ".join(desc))


if __name__ == "__main__":
    main()
