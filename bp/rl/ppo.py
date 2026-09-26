"""PPO training against scripted NPC tamers.

Usage:  python -m bp.rl.ppo --workers 16 --envs 16 --updates 2000 [--family 1] [--tamers 1,2,3]

Environments run in worker processes; the policy runs batched on the GPU in the main process.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict, deque

from . import dashboard
from .env import PetBattleEnv, ScenarioSampler
from .features import vocab_sizes
from .model import PetNet

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["pet_int", "pet_float", "aura_int", "aura_float", "aura_mask", "glob"]


def worker(remote, n_envs, seed, sampler_kw):
    sampler = ScenarioSampler(seed=seed, **sampler_kw)
    envs = [PetBattleEnv(sampler, seed=seed * 1000 + i) for i in range(n_envs)]
    obs = [e.reset() for e in envs]
    remote.send(obs)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            out = []
            for e, a in zip(envs, data):
                (o, m), r, d = e.step(int(a))
                info = None
                if d:
                    info = (f"{e.family}:{e.tamer}", e.battle.winner == 0, e.battle.round)
                    o, m = e.reset()
                out.append((o, m, r, d, info))
            remote.send(out)
        elif cmd == "close":
            break


def stack(obs_list, device):
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), device=device) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=64, help="rollout length per env")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--family", type=int, default=None)
    ap.add_argument("--tamers", type=str, default=None)
    ap.add_argument("--achievements", action="store_true", help="train on tamers of unfinished achievements")
    ap.add_argument("--out", type=str, default="checkpoints/ppo.pt")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler_kw = {"family": args.family, "tamer_ids": [int(x) for x in args.tamers.split(",")] if args.tamers else None,
                  "achievements": args.achievements}
    ctx = mp.get_context("spawn")
    remotes, procs = [], []
    for w in range(args.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, args.envs, w + 1, sampler_kw), daemon=True)
        p.start()
        remotes.append(a)
        procs.append(p)
    obs = []
    for r in remotes:
        obs.extend(r.recv())
    n = len(obs)

    vs = vocab_sizes()
    net = PetNet(vs["species"], vs["ability"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    start_upd = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["model"])
        if Path(args.resume).name == Path(args.out).name:
            start_upd = ck.get("update", 0)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    gamma, lam, clip, epochs, mb = 0.99, 0.95, 0.2, 4, 2048
    recent = []
    per_task = defaultdict(lambda: deque(maxlen=200))
    run_name = Path(args.out).stem
    stats = {"name": run_name, "start": time.time(), "start_upd": start_upd, "total_updates": args.updates,
             "history": [], "per_task": {}}
    prev = ROOT / "logs" / "dashboard.json"
    if start_upd and prev.exists():
        old = json.loads(prev.read_text(encoding="utf-8"))
        if old.get("name") == run_name:
            stats["history"] = [h for h in old["history"] if h["upd"] <= start_upd]
    t0 = time.time()
    for upd in range(start_upd + 1, args.updates + 1):
        buf = {k: [] for k in KEYS}
        masks, acts, logps, vals, rews, dones = [], [], [], [], [], []
        for t in range(args.steps):
            o = stack([x[0] for x in obs], device)
            m = torch.as_tensor(np.stack([x[1] for x in obs]), device=device)
            with torch.no_grad():
                logits, v = net(o, m)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            for k in KEYS:
                buf[k].append(o[k])
            masks.append(m)
            acts.append(a)
            logps.append(dist.log_prob(a))
            vals.append(v)
            a_np = a.cpu().numpy()
            for i, r in enumerate(remotes):
                r.send(("step", a_np[i * args.envs:(i + 1) * args.envs]))
            res = []
            for r in remotes:
                res.extend(r.recv())
            obs = [(x[0], x[1]) for x in res]
            rews.append(torch.tensor([x[2] for x in res], device=device))
            dones.append(torch.tensor([float(x[3]) for x in res], device=device))
            for x in res:
                if x[4] is not None:
                    recent.append(x[4])
                    per_task[x[4][0]].append(x[4][1])
        with torch.no_grad():
            o = stack([x[0] for x in obs], device)
            m = torch.as_tensor(np.stack([x[1] for x in obs]), device=device)
            _, last_v = net(o, m)
        # GAE
        T = args.steps
        adv = torch.zeros(T, n, device=device)
        last = torch.zeros(n, device=device)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else vals[t + 1]
            nonterm = 1 - dones[t]
            delta = rews[t] + gamma * nv * nonterm - vals[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + torch.stack(vals)
        flat = {k: torch.cat(buf[k]) for k in KEYS}
        f_mask, f_act, f_logp = torch.cat(masks), torch.cat(acts), torch.cat(logps)
        f_adv, f_ret = adv.flatten(), ret.flatten()
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        N = f_act.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N, device=device)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                logits, v = net({k: flat[k][idx] for k in KEYS}, f_mask[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(f_act[idx])
                ratio = (lp - f_logp[idx]).exp()
                a_ = f_adv[idx]
                pg = -torch.min(ratio * a_, ratio.clamp(1 - clip, 1 + clip) * a_).mean()
                vl = F.mse_loss(v, f_ret[idx])
                ent = dist.entropy().mean()
                loss = pg + 0.5 * vl - 0.01 * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if upd % 10 == 0 or upd == 1:
            last_k = recent[-2000:]
            wr = np.mean([x[1] for x in last_k]) if last_k else float("nan")
            rl = np.mean([x[2] for x in last_k]) if last_k else float("nan")
            sps = (upd - start_upd) * T * n / (time.time() - t0)
            print(f"upd {upd} games {len(recent)} winrate {wr:.3f} rounds {rl:.1f} ent {ent.item():.2f} "
                  f"vloss {vl.item():.3f} {sps:.0f} steps/s", flush=True)
            stats["history"].append({"upd": upd, "games": len(recent), "winrate": float(wr), "rounds": float(rl),
                                     "sps": sps, "t": time.time()})
            stats["per_task"] = {k: {"wr": float(np.mean(v)), "n": len(v)} for k, v in per_task.items() if len(v) >= 20}
            try:
                dashboard.render(ROOT / "logs" / "dashboard.html", stats)
            except Exception as ex:  # never kill training because of the dashboard
                print("dashboard error", ex, flush=True)
        if upd % 10 == 0 or upd == args.updates:
            torch.save({"model": net.state_dict(), "args": vars(args), "update": upd}, out)
    for r in remotes:
        r.send(("close", None))


if __name__ == "__main__":
    main()
