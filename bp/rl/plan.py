"""Turn the trained policy into a step-by-step battle plan (like a Xu-Fu strategy).

Plays many battles with the greedy policy, records (our pet, move, enemy pet) per round and
follows the most common line; reports how often each step is taken and the final win rate.

Usage (from Python):  plan(tamer_id, team_specs, games=200) -> dict
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from ..sim import constants as C
from ..sim.data import game_data
from .env import PetBattleEnv, ScenarioSampler
from .ppo import KEYS


def _label(env, action):
    b = env.battle
    pet = b.teams[0].active_pet
    if action in C.A_ABILITY:
        return f"{pet.name}: {game_data().abilities[pet.abilities[action]]['name']}"
    if action in C.A_SWAP:
        return f"swap -> {b.teams[0].pets[action - 3].name}"
    return f"{pet.name}: (pass / continues)"


def record(net, tamer, team, games=200, device="cpu"):
    sampler = ScenarioSampler(seed=0)
    envs = []
    for g in range(games):
        e = PetBattleEnv(sampler, seed=10_000 + g)
        e.reset((tamer, team))
        envs.append(e)
    traj = [[] for _ in envs]
    obs = [e.observe() for e in envs]
    done = [False] * len(envs)
    wins = [False] * len(envs)
    while not all(done):
        idx = [i for i, d in enumerate(done) if not d]
        o = {k: torch.as_tensor(np.stack([obs[i][0][k] for i in idx]), device=device) for k in KEYS}
        m = torch.as_tensor(np.stack([obs[i][1] for i in idx]), device=device)
        with torch.no_grad():
            logits, _ = net(o, m)
        acts = logits.argmax(1).tolist()
        for j, i in enumerate(idx):
            e = envs[i]
            b = e.battle
            enemy = b.teams[1].active_pet.name
            forced = b.legal_actions(0) == [C.A_PASS]
            traj[i].append((b.round + 1, _label(e, acts[j]), enemy, forced))
            obs[i], _, d = e.step(acts[j])
            if d:
                done[i] = True
                wins[i] = b.winner == 0
    return traj, wins


def main_line(traj, wins, min_share=0.5):
    """Follow the most common move while at least min_share of the battles agree."""
    alive = list(range(len(traj)))
    steps = []
    k = 0
    while alive:
        c = Counter(traj[i][k][1:3] for i in alive if k < len(traj[i]))
        if not c:
            break
        (move, enemy), n = c.most_common(1)[0]
        share = n / len(traj)
        if share < min_share:
            steps.append({"step": k, "move": None, "note": "дальше бои расходятся"})
            break
        forced = any(traj[i][k][3] for i in alive if k < len(traj[i]) and traj[i][k][1:3] == (move, enemy))
        steps.append({"step": k, "move": move, "enemy": enemy, "share": share, "forced": forced})
        alive = [i for i in alive if k < len(traj[i]) and traj[i][k][1:3] == (move, enemy)]
        k += 1
    followers = alive
    return steps, (np.mean([wins[i] for i in followers]) if followers else float("nan")), len(followers) / len(traj)
