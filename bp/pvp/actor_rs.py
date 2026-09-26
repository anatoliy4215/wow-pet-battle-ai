"""Stage 2 self-play actor on the Rust engine (bpsim.ActorPool): one process, all cores as threads.

Same job and output as bp/pvp/actor.py (look-ahead self-play -> .npz shards for bp.pvp.learner),
but the battles, features, hints and search run in Rust. Two pools take turns so the GPU evaluates
one while the CPU works on the other. Playout-cap randomisation: a share of the decisions (--deep)
is searched deep (--k-deep worlds, two rounds); only those train the policy (shard key "pw").

  python -m bp.pvp.actor_rs --weights checkpoints/pvp/actor.pt --out data/pvp/shards/local --threads 12
"""
from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .actor import DTYPES, KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--envs", type=int, default=256, help="battles per pool (two pools)")
    ap.add_argument("--k-deep", type=int, default=8)
    ap.add_argument("--k-cheap", type=int, default=4)
    ap.add_argument("--depth-deep", type=int, default=2)
    ap.add_argument("--deep", type=float, default=0.25, help="share of decisions searched deep")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--practice", type=float, default=0.2)
    ap.add_argument("--shard", type=int, default=4096)
    ap.add_argument("--hours", type=float, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)

    import torch
    from ..sim.rust import export_state, module, spec_tuple
    from .env import PvPEnv
    from .model import load as load_net, masked
    from .pool import TeamSampler
    from .scenarios import start as scenario_start

    bpsim = module()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = args.seed or int(time.time()) % 100000
    rng = random.Random(seed)
    sampler = TeamSampler(seed=seed, meta_pet_frac=0.35, meta_team_frac=0.15)
    practice_env = PvPEnv(sampler, seed=seed + 1)
    net, mtime = None, 0.0

    def load():
        nonlocal net, mtime
        m = os.path.getmtime(args.weights)
        if m == mtime:
            return
        ck = torch.load(args.weights, map_location=device, weights_only=False)
        n = load_net(ck, device)
        n.eval()
        net, mtime = n, m
        print(f"weights loaded (update {ck.get('update')})", flush=True)

    while not os.path.exists(args.weights):
        print("waiting for weights...", flush=True)
        time.sleep(10)
    load()

    def new_starts(n):
        res = []
        for _ in range(n):
            if rng.random() < args.practice:
                scenario_start(practice_env, sampler, rng)
                if not practice_env.done and practice_env.phase != 0:
                    b = practice_env.battle
                    res.append(([spec_tuple(p.spec) for p in b.teams[0].pets],
                                [spec_tuple(p.spec) for p in b.teams[1].pets], export_state(b)))
                    continue
            res.append(([spec_tuple(s) for s in sampler.team()], [spec_tuple(s) for s in sampler.team()], None))
        return res

    def evaluate(obs, mask):
        pols, vals, los = [], [], []
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            for s in range(0, len(mask), 4096):
                x = {k: torch.from_numpy(obs[k][s:s + 4096]).to(device, non_blocking=True) for k in KEYS}
                m = torch.from_numpy(mask[s:s + 4096]).to(device, non_blocking=True)
                logits, v, _, lo = net(x)
                pols.append(torch.softmax(masked(logits, m), -1).float())
                vals.append(v.float())
                los.append(torch.sigmoid(lo.float()))
        return (torch.cat(pols).cpu().numpy(), torch.cat(vals).cpu().numpy(), torch.cat(los).cpu().numpy())

    pools = [bpsim.ActorPool(args.envs, seed=seed * 10 + i, k_deep=args.k_deep, k_cheap=args.k_cheap,
                             depth_deep=args.depth_deep, deep_frac=args.deep, tau=args.tau) for i in range(2)]
    ex = ThreadPoolExecutor(max_workers=2)
    futs = []
    for p in pools:
        p.add_starts(new_starts(args.envs * 2))
        futs.append(ex.submit(p.kick))
    shard_n = 0
    buf = []
    n_games = n_samples = n_evals = 0
    t0 = last_log = last_check = time.time()
    stop_file = Path(__file__).resolve().parents[2] / "logs" / "STOP_ACTORS"

    def flush(force=False):
        nonlocal shard_n
        have = sum(len(b["z"]) for b in buf)
        if not have or (have < args.shard and not force):
            return
        d = {k: np.concatenate([b[k] for b in buf]) for k in buf[0]}
        buf.clear()
        data = {k: d[k].astype(DTYPES[k]) for k in KEYS}
        data["mask"] = d["mask"].astype(np.uint8)
        data["pi"] = d["pi"].astype(np.float16)
        data["q"] = d["q"].astype(np.float16)
        data["z"] = d["z"].astype(np.int8)
        data["zd"] = d["zd"].astype(np.float16)
        data["opp_act"] = d["opp_act"].astype(np.int8)
        data["lo_y"] = d["lo_y"].astype(np.uint8)
        data["pw"] = d["pw"].astype(np.uint8)
        name = f"rs_{seed}_{int(time.time())}_{shard_n:05d}"
        tmp = out / (name + ".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(out / (name + ".npz"))
        shard_n += 1

    while True:
        for i, p in enumerate(pools):
            futs[i].result()  # this pool's CPU work is done
            if p.n_starts() < args.envs:
                p.add_starts(new_starts(args.envs))
            if p.n_samples() >= 2048:
                s = p.take_samples()
                n_samples += len(s["z"])
                buf.append(s)
            n_games += len(p.take_games())
            obs, mask = p.requests()
            if len(mask) == 0:
                futs[i] = ex.submit(p.kick)
                continue
            n_evals += len(mask)
            pol, val, lo = evaluate(obs, mask)  # the other pool keeps working meanwhile
            futs[i] = ex.submit(p.reply, pol, val, lo)
        flush()
        now = time.time()
        if now - last_check > 30:
            last_check = now
            try:
                load()
            except Exception as e:  # the learner may be writing the file
                print("reload failed:", e, flush=True)
        if now - last_log > 60:
            last_log = now
            m = (now - t0) / 60
            print(f"{m:.0f} min: {n_games} games, {n_samples} samples, {n_evals / (now - t0):,.0f} evals/s, "
                  f"{len(list(out.glob('*.npz')))} shards waiting", flush=True)
        if (args.hours and now - t0 > args.hours * 3600) or (stop_file.exists() and stop_file.stat().st_mtime > t0):
            for f in futs:
                f.result()
            for p in pools:
                if p.n_samples():
                    buf.append(p.take_samples())
            flush(force=True)
            break


if __name__ == "__main__":
    main()
