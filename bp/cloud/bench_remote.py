"""Hardware benchmark for our workload (runs on a rented box, or locally for reference).

  python -m bp.cloud.bench_remote [--seconds 20]  -> one JSON line
CPU: battle simulator + observation encoder (what self-play workers do), 1 process and all cores.
GPU: PvP network inference (batch 1024) and training (fwd+bwd, batch 1024, bf16), the real model size.
Net: download speed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import random
import time


def cpu_info():
    name = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    # a docker container may see all host cores but be limited by a CPU quota (cgroup)
    quota = None
    try:
        q, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota:
        n = max(1, min(n, int(quota + 0.5)))
    return name, n


def sim_worker(seconds, seed, q):
    from bp.pvp.bots import RandomBot
    from bp.pvp.env import PvPEnv
    from bp.pvp.pool import TeamSampler
    rng = random.Random(seed)
    env = PvPEnv(TeamSampler(seed=seed), seed=seed)
    bot = RandomBot()
    env.reset()
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        if env.done:
            env.reset()
        acts = {}
        for s in env.pending():
            env.observe(s)
            n += 1
            acts[s] = bot.act(env.battle, s, env.phase, rng)
        env.act(acts)
    q.put(n / (time.time() - t0))


def cpu_bench(procs, seconds):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=sim_worker, args=(seconds, 100 + i, q)) for i in range(procs)]
    for p in ps:
        p.start()
    rates = [q.get() for _ in ps]
    for p in ps:
        p.join()
    return sum(rates)


def gpu_bench():
    import numpy as np
    import torch
    from bp.pvp.env import PvPEnv
    from bp.pvp.features import KEYS
    from bp.pvp.model import PvPNet
    from bp.pvp.pool import TeamSampler
    if not torch.cuda.is_available():
        return {"gpu": None}
    env = PvPEnv(TeamSampler(seed=1), seed=2)
    env.reset()
    env.act({0: 3, 1: 3})
    o = env.observe(0)[0]
    dev = "cuda"
    net = PvPNet(d=320, layers=8).to(dev)
    B = 1024
    batch = {k: torch.as_tensor(np.stack([o[k]] * B)).to(dev) for k in KEYS}
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            net(batch)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            net(batch)
        torch.cuda.synchronize()
        inf = B * 20 / (time.time() - t)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    for i in range(12):
        if i == 2:
            torch.cuda.synchronize()
            t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            lg, v, op, lo = net(batch)
            loss = lg.float().mean() + v.float().mean() + op.float().mean() + lo.float().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    tr = B * 10 / (time.time() - t)
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "infer_obs_s": round(inf),
            "train_samples_s": round(tr), "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}


def net_bench():
    import urllib.request
    try:
        t = time.time()
        with urllib.request.urlopen("https://speed.cloudflare.com/__down?bytes=50000000", timeout=60) as r:
            n = len(r.read())
        return round(n * 8 / 1e6 / (time.time() - t))
    except Exception as ex:  # noqa: BLE001
        return str(ex)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--procs", type=int, default=0)
    args = ap.parse_args()
    name, n = cpu_info()
    out = {"cpu": name, "cores": n}
    t0 = time.time()
    out["sim_1core"] = round(cpu_bench(1, args.seconds))
    out["sim_all"] = round(cpu_bench(args.procs or n, args.seconds))
    try:
        out.update(gpu_bench())
    except Exception as ex:  # noqa: BLE001 - e.g. no bf16 on an old card: keep the CPU numbers
        out["gpu_error"] = str(ex)[:120]
    out["net_mbps"] = net_bench()
    out["bench_s"] = round(time.time() - t0)
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
