"""League self-play PPO for PvP pet battles.

  python -m bp.pvp.train --out checkpoints/pvp/main.pt            (fresh start)
  python -m bp.pvp.train --out checkpoints/pvp/main.pt --resume   (continue after a restart)

Workers run the battles (both sides), bots act inside the workers, the networks run batched on the
GPU in the main process.  Opponents of the learner:
  self    - the current network plays both sides (both sides' decisions are training data)
  snap:N  - a frozen earlier version (league); chosen by "prioritized fictitious self-play": versions
            we still lose to are picked more often, so old weaknesses do not come back
  greedy / random - scripted anchors (fixed strength -> the rating curve on the dashboard means something)
Workers keep full trajectories and compute advantages themselves; the main process only collects
finished samples when it trains.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------------------------- worker
def worker(remote, wid, n_envs, seed, cfg):
    torch.set_num_threads(1)
    from .bots import GreedyBot, RandomBot
    from .env import PvPEnv
    from .features import KEYS
    from .pool import TeamSampler

    rng = random.Random(seed)
    sampler = TeamSampler(seed=seed, meta_pet_frac=cfg["meta_pet_frac"], meta_team_frac=cfg["meta_team_frac"])
    envs = [PvPEnv(sampler, seed=seed * 1000 + i, lead_select=cfg["lead"], hide=cfg["hide"]) for i in range(n_envs)]
    bots = {"greedy": GreedyBot(), "random": RandomBot()}
    mix = cfg["mix"]  # [(label, prob)]; labels: self, snap:<slot>, greedy, random
    gamma, lam = cfg["gamma"], cfg["lam"]

    ctrl = [None] * n_envs      # per env: [controller of side 0, side 1]; "L" = learner, int = snapshot slot, str = bot
    label = [None] * n_envs
    traj = [None] * n_envs      # per env, per side: list of transitions (learner sides only)
    done_samples = []           # finished transitions, ready to ship
    ended = []

    def start(i):
        envs[i].reset()
        lab = rng.choices([m[0] for m in mix], [m[1] for m in mix])[0]
        label[i] = lab
        if lab == "self":
            c = ["L", "L"]
        else:
            other = int(lab[5:].split("@")[0]) if lab.startswith("snap:") else lab
            c = ["L", other] if rng.random() < 0.5 else [other, "L"]
        ctrl[i] = c
        traj[i] = [[], []]

    def finish(i):
        env = envs[i]
        w = env.winner
        for side in (0, 1):
            if ctrl[i][side] != "L":
                continue
            tr = traj[i][side]
            if not tr:
                continue
            z = 0.0 if w == -1 else (1.0 if w == side else -1.0)
            y, _ = env.loadout_truth(side)
            adv = 0.0
            nxt = 0.0
            for t in range(len(tr) - 1, -1, -1):
                r = z if t == len(tr) - 1 else 0.0
                v = tr[t]["val"]
                delta = r + gamma * nxt - v
                adv = delta + gamma * lam * adv
                tr[t]["adv"] = adv
                tr[t]["ret"] = adv + v
                tr[t]["lo_y"] = y
                nxt = v
            done_samples.extend(tr)
        learner = [s for s in (0, 1) if ctrl[i][s] == "L"]
        ended.append((label[i], learner, w, env.battle.round))
        start(i)

    for i in range(n_envs):
        start(i)

    bot_act = [dict() for _ in range(n_envs)]
    while True:
        # gather decisions that need a network; let bots act; step envs that only wait for bots
        reqs, obs, masks = [], [], []
        for i, env in enumerate(envs):
            while True:
                if env.done:
                    finish(i)
                pend = env.pending()
                bot_act[i] = {}
                nn_sides = []
                for s in pend:
                    c = ctrl[i][s]
                    if isinstance(c, str) and c != "L":
                        bot_act[i][s] = bots[c].act(env.battle, s, env.phase, rng)
                    else:
                        nn_sides.append(s)
                if nn_sides:
                    for s in nn_sides:
                        o, m = env.observe(s)
                        reqs.append((i, s, 0 if ctrl[i][s] == "L" else ctrl[i][s]))
                        obs.append(o)
                        masks.append(m)
                    break
                env.act(bot_act[i])
        batch = {k: np.stack([o[k] for o in obs]) for k in KEYS}
        remote.send(("req", batch, np.stack(masks), [r[2] for r in reqs], ended, len(done_samples)))
        ended = []
        cmd, data = remote.recv()
        while cmd != "step":
            if cmd == "collect":
                if done_samples:
                    out = {k: np.stack([s["obs"][k] for s in done_samples]) for k in KEYS}
                    for k in ("mask", "lo_y"):
                        out[k] = np.stack([s[k] for s in done_samples])
                    for k in ("act", "logp", "val", "adv", "ret", "opp_act"):
                        out[k] = np.array([s[k] for s in done_samples])
                else:
                    out = None
                done_samples.clear()
                remote.send(("samples", out))
            elif cmd == "mix":
                mix = data
            elif cmd == "close":
                return
            cmd, data = remote.recv()
        acts, logps, vals = data
        # apply: combine network actions with bot actions per env
        per_env = defaultdict(dict)
        for j, (i, s, pid) in enumerate(reqs):
            per_env[i][s] = int(acts[j])
            if pid == 0:
                traj[i][s].append({"obs": obs[j], "mask": masks[j], "act": int(acts[j]), "logp": float(logps[j]),
                                   "val": float(vals[j]), "opp_act": -1})
        for i, a in per_env.items():
            a.update(bot_act[i])
            for s in a:
                if ctrl[i][s] == "L" and traj[i][s]:
                    last = traj[i][s][-1]
                    if last["opp_act"] == -1 and (1 - s) in a and env_ok(envs[i]):
                        last["opp_act"] = a[1 - s]
            envs[i].act(a)


def env_ok(env):
    # opponent-action targets only for real rounds (in the lead / replacement phase it is a pet pick)
    return env.phase == 1


# ---------------------------------------------------------------------------------------------- main
def to_dev(batch, keys, device, idx=None):
    out = {}
    for k in keys:
        x = batch[k] if idx is None else batch[k][idx]
        out[k] = torch.as_tensor(x, device=device)
    return out


class League:
    """Frozen snapshots + results against every opponent label."""

    def __init__(self, dirpath, max_active=3):
        self.dir = Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snaps = []  # [{"upd": n, "path": str}]
        self.results = defaultdict(lambda: deque(maxlen=400))  # label -> recent learner scores (1/0.5/0)
        self.max_active = max_active
        self.active = {}  # slot -> snapshot upd

    def state(self):
        return {"snaps": self.snaps, "results": {k: list(v) for k, v in self.results.items()},
                "active": self.active}

    def load_state(self, st):
        self.snaps = st.get("snaps", [])
        for k, v in st.get("results", {}).items():
            self.results[k].extend(v)
        self.active = {int(k): v for k, v in st.get("active", {}).items()}

    def add(self, net, upd):
        path = self.dir / f"snap_{upd:06d}.pt"
        torch.save({"model": net.state_dict(), "cfg": net.cfg, "update": upd}, path)
        self.snaps.append({"upd": upd, "path": str(path)})

    def winrate(self, label):
        r = self.results.get(label)
        return float(np.mean(r)) if r else None

    def pick(self, rng):
        """PFSP: weight (1 - winrate)^2, recent versions always in the running."""
        if not self.snaps:
            return {}
        cand = self.snaps[-40:]
        w = []
        for s in cand:
            wr = self.winrate(f"snap@{s['upd']}")
            w.append((1 - (wr if wr is not None else 0.5)) ** 2 + 0.02)
        chosen = {}
        pool = list(range(len(cand)))
        latest = len(cand) - 1
        picks = [latest]
        while len(picks) < min(self.max_active, len(cand)):
            k = rng.choices(pool, w)[0]
            if k not in picks:
                picks.append(k)
        for slot, k in enumerate(picks, start=1):
            chosen[slot] = cand[k]["upd"]
        self.active = chosen
        return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--samples", type=int, default=65536, help="samples per update")
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=4096, help="samples per optimizer step")
    ap.add_argument("--micro", type=int, default=1024, help="samples per forward pass (GPU memory)")
    ap.add_argument("--d", type=int, default=320)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/pvp/main.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-lead", action="store_true", help="first pet always starts (no lead pick phase)")
    ap.add_argument("--show-abilities", action="store_true", help="enemy abilities visible from the start")
    args = ap.parse_args()

    from . import dashboard
    from .features import KEYS
    from .model import PvPNet, build, masked

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    league = League(out.parent / "league")
    rng = random.Random(12345)

    net = PvPNet(d=args.d, layers=args.layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    start_upd, games_total, samples_total = 0, 0, 0
    history = []
    if args.resume and out.exists():
        ck = torch.load(out, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_upd = ck["update"]
        games_total = ck.get("games", 0)
        samples_total = ck.get("samples", 0)
        league.load_state(ck.get("league", {}))
        history = ck.get("history", [])
        print(f"resumed at update {start_upd}", flush=True)
    n_params = sum(p.numel() for p in net.parameters())

    snap_nets = {}

    def load_active(active):
        for slot, upd in active.items():
            s = next(x for x in league.snaps if x["upd"] == upd)
            m = snap_nets.get(slot)
            if m is None or getattr(m, "_upd", None) != upd:
                ck = torch.load(s["path"], map_location=device, weights_only=False)
                m = build(ck["cfg"]).to(device)
                m.load_state_dict(ck["model"])
                m.eval()
                m._upd = upd
                snap_nets[slot] = m

    def make_mix():
        active = league.pick(rng)
        load_active(active)
        if not active:
            return [("self", 0.80), ("greedy", 0.15), ("random", 0.05)], active
        mix = [("self", 0.55), ("greedy", 0.08), ("random", 0.02)]
        per = 0.35 / len(active)
        mix += [(f"snap:{slot}@{u}", per) for slot, u in active.items()]
        return mix, active

    from . import bench, tactics
    suite = tactics.build_suite(n=300)
    bench_pairs = bench.make_pairs()
    early = {}

    def run_bench(row):
        """Fixed benchmark (same 400 games every time) + one recorded game for the dashboard."""
        try:
            row["bench_greedy"], game = bench.play(net, "greedy", bench_pairs, seed=1, record=0)
            game["upd"] = row["upd"]
            (ROOT / "logs" / "pvp_sample_game.json").write_text(json.dumps(game, ensure_ascii=False), encoding="utf-8")
            anchor = out.parent / "league" / "snap_000020.pt"
            if anchor.exists():
                if "net" not in early:
                    ck = torch.load(anchor, map_location=device, weights_only=False)
                    m = build(ck["cfg"]).to(device)
                    m.load_state_dict(ck["model"])
                    m.eval()
                    early["net"] = m
                row["bench_early"], _ = bench.play(net, early["net"], bench_pairs, seed=2, record=None)
        except Exception as ex:
            print("bench error", ex, flush=True)

    mix, active = make_mix()
    cfg = {"lead": not args.no_lead, "hide": not args.show_abilities, "meta_pet_frac": 0.35, "meta_team_frac": 0.15,
           "mix": mix, "gamma": 1.0, "lam": 0.95}
    ctx = mp.get_context("spawn")
    remotes, procs = [], []
    for w in range(args.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, w, args.envs, 1000 * (start_upd + 1) + w, cfg), daemon=True)
        p.start()
        remotes.append(a)
        procs.append(p)

    recent_len = deque(maxlen=4000)
    t_start = time.time()
    upd = start_upd
    t_collect = time.time()

    def reply(msgs):
        """Run the networks on every worker's pending decisions and send the actions back."""
        sizes = [len(m[3]) for m in msgs]
        obs = {k: np.concatenate([m[1][k] for m in msgs]) for k in KEYS}
        mask = np.concatenate([m[2] for m in msgs])
        pids = np.concatenate([np.array(m[3], dtype=np.int64) for m in msgs])
        acts = np.zeros(len(pids), dtype=np.int64)
        logps = np.zeros(len(pids), dtype=np.float32)
        vals = np.zeros(len(pids), dtype=np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for pid in np.unique(pids):
                idx = np.nonzero(pids == pid)[0]
                model = net if pid == 0 else snap_nets.get(int(pid), net)
                logits, v, _, _ = model(to_dev(obs, KEYS, device, idx))
                dist = torch.distributions.Categorical(logits=masked(logits, torch.as_tensor(mask[idx], device=device)))
                a = dist.sample()
                acts[idx] = a.cpu().numpy()
                logps[idx] = dist.log_prob(a).float().cpu().numpy()
                vals[idx] = v.float().cpu().numpy()
        off = 0
        for r, n in zip(remotes, sizes):
            r.send(("step", (acts[off:off + n], logps[off:off + n], vals[off:off + n])))
            off += n

    def train_step(upd):
        nonlocal samples_total, mix, active
        for r in remotes:
            r.send(("collect", None))
        parts = [r.recv()[1] for r in remotes]
        parts = [p for p in parts if p is not None]
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        N = len(data["act"])
        t_train = time.time()
        collect_s = t_train - t_collect
        adv = data["adv"].astype(np.float32)
        data["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        for k in ("logp", "ret", "val"):
            data[k] = data[k].astype(np.float32)
        lr = args.lr * min(1.0, (upd + 1) / 20)  # warm-up
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        stats = defaultdict(list)
        stop = False
        for _ in range(args.epochs):
            if stop:
                break
            perm = np.random.permutation(N)
            step_kl = []
            for s0 in range(0, N, args.mb):
                big = perm[s0:s0 + args.mb]
                chunks = [big[c:c + args.micro] for c in range(0, len(big), args.micro)]
                opt.zero_grad(set_to_none=True)
                kls = []
                for idx in chunks:
                    o = to_dev(data, KEYS, device, idx)
                    m = torch.as_tensor(data["mask"][idx], device=device)
                    act = torch.as_tensor(data["act"][idx], device=device)
                    old_lp = torch.as_tensor(data["logp"][idx], device=device)
                    a_ = torch.as_tensor(data["adv"][idx], device=device)
                    ret = torch.as_tensor(data["ret"][idx], device=device)
                    opp_t = torch.as_tensor(data["opp_act"][idx], device=device)
                    lo_y = torch.as_tensor(data["lo_y"][idx], device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, v, opp, lo = net(o)
                    dist = torch.distributions.Categorical(logits=masked(logits, m))
                    lp = dist.log_prob(act)
                    ratio = (lp - old_lp).exp()
                    pg = -torch.min(ratio * a_, ratio.clamp(0.8, 1.2) * a_).mean()
                    vl = F.mse_loss(v.float(), ret)
                    ent = dist.entropy().mean()
                    ok = opp_t >= 0
                    opp_l = F.cross_entropy(opp.float()[ok], opp_t[ok]) if ok.any() else torch.zeros((), device=device)
                    hid = o["ab_mask"][:, 10::2]  # second candidate present = slot still hidden
                    lo_l = (F.binary_cross_entropy_with_logits(lo.float(), lo_y, reduction="none") * hid).sum() / hid.sum().clamp(min=1)
                    loss = pg + 0.5 * vl - 0.01 * ent + 0.1 * opp_l + 0.1 * lo_l
                    (loss * len(idx) / len(big)).backward()  # gradient accumulation over micro-batches
                    with torch.no_grad():
                        stats["pg"].append(pg.item())
                        stats["vl"].append(vl.item())
                        stats["ent"].append(ent.item())
                        logr = lp - old_lp
                        kls.append(((logr.exp() - 1) - logr).mean().item())  # low-variance KL estimate, >= 0
                        stats["clip"].append(((ratio - 1).abs() > 0.2).float().mean().item())
                        if ok.any():
                            stats["opp_acc"].append((opp.argmax(1)[ok] == opp_t[ok]).float().mean().item())
                        if hid.sum() > 0:
                            pred = (lo > 0).float()
                            stats["lo_acc"].append((((pred == lo_y).float() * hid).sum() / hid.sum()).item())
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                kl = float(np.mean(kls))
                stats["kl"].append(kl)
                step_kl.append(kl)
                stats["steps"].append(1)
                # the policy already moved far enough this update: stop early (running mean of 3 steps)
                if len(step_kl) >= 3 and np.mean(step_kl[-3:]) > args.target_kl:
                    stop = True
                    break
        net.eval()
        upd += 1
        samples_total += N
        train_s = time.time() - t_train
        row = {"upd": upd, "t": time.time(), "games": games_total, "samples": samples_total,
               "rounds": float(np.mean(recent_len)) if recent_len else 0.0,
               "collect_s": collect_s, "train_s": train_s, "sps": N / max(1e-6, collect_s + train_s), "n": N}
        for k, v in stats.items():
            row[k] = float(np.mean(v))
        row["steps"] = len(stats["steps"])  # optimizer steps done before the KL stop
        for lab in ("greedy", "random"):
            row[f"wr_{lab}"] = league.winrate(lab)
        snap_wr = [league.winrate(f"snap@{x['upd']}") for x in league.snaps[-10:]]
        snap_wr = [x for x in snap_wr if x is not None]
        row["wr_snaps"] = float(np.mean(snap_wr)) if snap_wr else None
        if upd % 25 == 0 or upd == start_upd + 1:
            run_bench(row)
        if upd % 10 == 0 or upd == start_upd + 1:
            try:
                for k, v in tactics.evaluate(net, suite).items():
                    if k != "n":
                        row[f"tac_{k}"] = v
            except Exception as ex:
                print("tactics error", ex, flush=True)
        history.append(row)
        print(f"upd {upd} games {games_total} samples {samples_total} rounds {row['rounds']:.1f} "
              f"ent {row['ent']:.2f} vl {row['vl']:.3f} kl {row['kl']:.4f} clip {row['clip']:.2f} steps {row['steps']} "
              f"opp_acc {row.get('opp_acc', 0):.2f} lo_acc {row.get('lo_acc', 0):.2f} "
              f"wr_greedy {row['wr_greedy'] or 0:.2f} wr_random {row['wr_random'] or 0:.2f} "
              f"wr_snaps {row['wr_snaps'] or 0:.2f} | collect {collect_s:.1f}s train {train_s:.1f}s", flush=True)
        if upd % 20 == 0:
            league.add(net, upd)
        if upd % 5 == 0 or upd == 1:
            mix, active = make_mix()
            for r in remotes:
                r.send(("mix", mix))
        if upd % 5 == 0:
            ck = {"model": net.state_dict(), "opt": opt.state_dict(), "update": upd, "cfg": net.cfg,
                  "games": games_total, "samples": samples_total, "league": league.state(), "history": history,
                  "args": vars(args)}
            tmp = out.with_suffix(".tmp")
            torch.save(ck, tmp)
            tmp.replace(out)
        try:
            dashboard.render(ROOT / "logs" / "pvp_dashboard.html", history, league, n_params, t_start, start_upd)
        except Exception as ex:  # never kill training because of the dashboard
            print("dashboard error", ex, flush=True)
        return upd

    while upd < args.updates:
        msgs = [r.recv() for r in remotes]
        ready = 0
        for _, _, _, _, ended, n_done in msgs:
            ready += n_done
            for lab, learner, w, rounds in ended:
                games_total += 1
                recent_len.append(rounds)
                if lab == "self":
                    continue
                key = lab if not lab.startswith("snap:") else "snap@" + lab.split("@")[1]
                for side in learner:
                    league.results[key].append(0.5 if w == -1 else float(w == side))
        if ready >= args.samples:
            upd = train_step(upd)
            t_collect = time.time()
        reply(msgs)
    for r in remotes:
        r.recv()
        r.send(("close", None))


if __name__ == "__main__":
    main()
