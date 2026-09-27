"""Stage 2 learner: trains the network on look-ahead samples written by bp.pvp.actor (local and rented
boxes), publishes fresh weights for the actors, keeps the league / benchmark / dashboard going.

  python -m bp.pvp.learner --init checkpoints/pvp/main.pt --out checkpoints/pvp/stage2.pt [--resume]

Losses: policy = cross-entropy to the improved policy pi_t (only where there was a real choice),
value = discounted final result (zd, see actor.py; older shards: plain result), plus the
opponent-move and hidden-ability heads as before.
Samples live in a replay buffer of the newest --buffer samples; each is used about --reuse times.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .actor import KEYS

ROOT = Path(__file__).resolve().parents[2]


class Buffer:
    def __init__(self, cap):
        self.cap, self.parts, self.n, self.seen = cap, deque(), 0, 0

    def add(self, d):
        self.parts.append(d)
        n = len(d["z"])
        self.n += n
        self.seen += n
        while self.n - len(self.parts[0]["z"]) >= self.cap and len(self.parts) > 1:
            self.n -= len(self.parts.popleft()["z"])

    def sample(self, bs, rng):
        sizes = np.array([len(p["z"]) for p in self.parts])
        which = rng.choice(len(self.parts), size=bs, p=sizes / sizes.sum())
        out = defaultdict(list)
        for pi in np.unique(which):
            k = int((which == pi).sum())
            p = self.parts[pi]
            idx = rng.integers(0, len(p["z"]), size=k)
            for key, arr in p.items():
                out[key].append(arr[idx])
        return {k: np.concatenate(v) for k, v in out.items()}


INT_KEYS = ("pet_int", "ab_int", "aura_int", "opp_act")


def load_shard(path):
    """A shard kept in its compact on-disk types (float16 / int8 / uint8): the replay buffer holds
    millions of samples, float32 / int64 copies needed ~4x the memory (the PC swapped). batch()
    converts only the sampled rows."""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    from .features import N_ABF, N_PET
    for k, n in (("pet_float", N_PET), ("ab_float", N_ABF)):  # shards from before the hint features
        if d[k].shape[-1] < n:
            d[k] = np.concatenate([d[k], np.zeros(d[k].shape[:-1] + (n - d[k].shape[-1],), d[k].dtype)], -1)
    if "zd" not in d:
        d["zd"] = d["z"]
    # playout-cap randomisation (actor_rs): only deeply searched decisions train the policy
    if "pw" not in d:
        d["pw"] = np.ones(len(d["z"]), np.uint8)
    return d


def batch_types(b):
    """Sampled rows -> the types the network / losses use."""
    return {k: (v.astype(np.int64) if k in INT_KEYS else v.astype(np.float32)) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/pvp/main.pt")
    ap.add_argument("--out", default="checkpoints/pvp/stage2.pt")
    ap.add_argument("--shards", default="data/pvp/shards", help="read every */*.npz below this directory")
    ap.add_argument("--weights", default="checkpoints/pvp/actor.pt", help="published for the actors")
    ap.add_argument("--buffer", type=int, default=3_000_000)
    ap.add_argument("--reuse", type=float, default=4.0)
    ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--micro", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="no dashboard / benchmark (tests)")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long: save, publish, refresh ratings")
    ap.add_argument("--steps", type=int, default=0, help="stop after this many updates (same finish as --minutes)")
    ap.add_argument("--duty", type=float, default=1.0, help="share of time the GPU trains (0.7: pauses so the PC stays quieter)")
    args = ap.parse_args()

    from . import bench, dashboard, tactics
    from .model import load as load_net, masked
    from .train import League

    device = "cuda"
    out = ROOT / args.out
    shard_root = ROOT / args.shards
    used_dir = shard_root / "_used"
    used_dir.mkdir(parents=True, exist_ok=True)
    src = out if (args.resume and out.exists()) else ROOT / args.init
    ck = torch.load(src, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    net = load_net(ck, device, upgrade=True)
    upgraded = net.cfg != {**cfg, "n_pet": cfg.get("n_pet", 40), "n_abf": cfg.get("n_abf", 10)}
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    if src == out and "opt" in ck and not upgraded:  # widened inputs: the old optimizer state no longer fits
        opt.load_state_dict(ck["opt"])
    upd = ck.get("update", 0)
    history = ck.get("history", [])
    games_total = ck.get("games", 0)
    league = League(out.parent / "league")
    league.load_state(ck.get("league", {}))
    stage2_start = ck.get("stage2_start", upd)
    n_params = sum(p.numel() for p in net.parameters())

    def publish():
        tmp = (ROOT / args.weights).with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "cfg": net.cfg, "update": upd}, tmp)
        tmp.replace(ROOT / args.weights)

    publish()
    suite = tactics.build_suite(n=300)
    pairs = bench.make_pairs()
    early = {}
    buf = Buffer(args.buffer)
    rng = np.random.default_rng(0)
    trained = 0.0
    t_start = time.time()
    last_pub = last_ck = time.time()
    stats = defaultdict(list)
    first_upd = upd
    print(f"stage 2 learner from update {upd}", flush=True)
    bad_shards = set()
    while True:
        # ingest new shards
        new = 0
        for p in sorted(shard_root.glob("*/*.npz")):
            if p.parent == used_dir or p.name.endswith(".tmp.npz") or p in bad_shards:
                continue
            try:
                d = load_shard(p)
            except Exception as ex:  # e.g. cut short by a power loss
                print("bad shard", p.name, str(ex)[:100], flush=True)
                try:
                    p.unlink(missing_ok=True)
                except OSError:  # Windows may still hold the failed reader's handle
                    bad_shards.add(p)
                continue
            buf.add(d)
            new += len(d["z"])
            games_total += int((d["z"] != 0).sum() // 20)  # rough; exact counts come from actor logs
            shutil.move(str(p), str(used_dir / f"{p.parent.name}_{p.name}"))
        # keep only the newest used shards on disk
        old = sorted(used_dir.glob("*.npz"), key=lambda x: x.stat().st_mtime)
        for p in old[:-2000]:
            p.unlink(missing_ok=True)
        budget = buf.seen * args.reuse / args.mb
        if buf.n < 20000 or trained >= budget:
            time.sleep(2)
            continue
        # one optimizer step
        t_step = time.time()
        batch = batch_types(buf.sample(args.mb, rng))
        net.train()
        opt.zero_grad(set_to_none=True)
        for s0 in range(0, args.mb, args.micro):
            sl = slice(s0, s0 + args.micro)
            o = {k: torch.as_tensor(batch[k][sl], device=device) for k in KEYS}
            m = torch.as_tensor(batch["mask"][sl], device=device)
            pi_t = torch.as_tensor(batch["pi"][sl], device=device)
            z = torch.as_tensor(batch["zd"][sl], device=device)
            opp_t = torch.as_tensor(batch["opp_act"][sl], device=device)
            lo_y = torch.as_tensor(batch["lo_y"][sl], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, v, opp, lo = net(o)
            logp = torch.log_softmax(masked(logits, m), -1)
            pw = torch.as_tensor(batch["pw"][sl], device=device)
            choice = (m.sum(1) > 1).float() * pw  # a real decision, searched deep enough to teach the policy
            pl = -((pi_t * logp).sum(1) * choice).sum() / choice.sum().clamp(min=1)
            ent = -(logp.exp() * logp).sum(1).mean()
            vl = F.mse_loss(v.float(), z)
            ok = opp_t >= 0
            opp_l = F.cross_entropy(opp.float()[ok], opp_t[ok]) if ok.any() else torch.zeros((), device=device)
            hid = o["ab_mask"][:, 10::2]
            lo_l = (F.binary_cross_entropy_with_logits(lo.float(), lo_y, reduction="none") * hid).sum() / hid.sum().clamp(min=1)
            loss = pl + 0.5 * vl - 0.003 * ent + 0.1 * opp_l + 0.1 * lo_l
            (loss * args.micro / args.mb).backward()
            stats["pl"].append(pl.item())
            stats["vl"].append(vl.item())
            stats["ent"].append(ent.item())
            if ok.any():
                stats["opp_acc"].append((opp.argmax(1)[ok] == opp_t[ok]).float().mean().item())
            if hid.sum() > 0:
                stats["lo_acc"].append(((((lo > 0).float() == lo_y).float() * hid).sum() / hid.sum()).item())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        net.eval()
        trained += 1
        if args.duty < 1:
            torch.cuda.synchronize()
            time.sleep((time.time() - t_step) * (1 - args.duty) / args.duty)
        # every 16 optimizer steps = one "update" (65k samples) on the same scale as stage 1
        if int(trained) % 16 == 0:
            upd += 1
            row = {"upd": upd, "t": time.time(), "games": games_total, "samples": buf.seen, "stage": 2, "n": 65536,
                   "run": {"from": first_upd, "steps": args.steps, "minutes": args.minutes, "started": t_start},
                   "buffer": buf.n}
            for k, v in stats.items():
                row[k] = float(np.mean(v))
            stats.clear()
            if upd % 25 == 0 and not args.quiet:
                try:
                    row["bench_greedy"], game = bench.play(net, "greedy", pairs, seed=1, record=0)
                    game["upd"] = upd
                    (ROOT / "logs" / "pvp_sample_game.json").write_text(json.dumps(game, ensure_ascii=False), encoding="utf-8")
                    anchor = out.parent / "league" / "snap_000020.pt"
                    if anchor.exists():
                        if "net" not in early:
                            a = torch.load(anchor, map_location=device, weights_only=False)
                            e = load_net(a, device)
                            e.eval()
                            early["net"] = e
                        row["bench_early"], _ = bench.play(net, early["net"], pairs, seed=2, record=None)
                except Exception as ex:
                    print("bench error", ex, flush=True)
            if upd % 10 == 0:
                torch.cuda.empty_cache()  # bench / exam batches reserve GPU memory; on Windows that also
                # counts as committed RAM (the PC swapped with the actor on the same GPU)
                for k, v in tactics.evaluate(net, suite).items():
                    if k != "n":
                        row[f"tac_{k}"] = v
            history.append(row)
            print(f"upd {upd} buffer {buf.n} seen {buf.seen} pl {row.get('pl', 0):.3f} vl {row.get('vl', 0):.3f} "
                  f"ent {row.get('ent', 0):.2f} opp_acc {row.get('opp_acc', 0):.2f} lo_acc {row.get('lo_acc', 0):.2f} "
                  f"bench {row.get('bench_greedy', '')}", flush=True)
            if upd % 20 == 0:
                league.add(net, upd)
            if upd % 300 == 0 and not args.quiet:  # refresh the pet / team ratings in the background
                import subprocess, sys
                subprocess.Popen([sys.executable, "-m", "bp.pvp.rankings"], cwd=str(ROOT),
                                 stdout=open(ROOT / "logs" / "pvp_rankings.log", "w"), stderr=subprocess.STDOUT)
            if not args.quiet:
                try:
                    dashboard.render(ROOT / "logs" / "pvp_dashboard.html", history, league, n_params, t_start, upd - 1)
                except Exception as ex:
                    print("dashboard error", ex, flush=True)
        now = time.time()
        if now - last_pub > 60:
            publish()
            last_pub = now
        finish = (args.minutes and now - t_start > args.minutes * 60) or (args.steps and upd >= first_upd + args.steps)
        if now - last_ck > 300 or finish:
            tmp = out.with_suffix(".tmp")
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "update": upd, "cfg": net.cfg,
                        "games": games_total, "samples": buf.seen, "league": league.state(), "history": history,
                        "stage2_start": stage2_start}, tmp)
            tmp.replace(out)
            last_ck = now
        if finish:
            publish()
            (ROOT / "logs" / "STOP_ACTORS").write_text("1")  # local actors stop when they see this
            if not args.quiet:
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "bp.pvp.rankings", "--ckpt", str(out)], cwd=str(ROOT),
                               stdout=open(ROOT / "logs" / "pvp_rankings.log", "w"), stderr=subprocess.STDOUT)
                subprocess.run([sys.executable, "-m", "bp.pvp.dashboard"], cwd=str(ROOT))
            print("finished: checkpoint saved, ratings refreshed", flush=True)
            return
        _ = random


if __name__ == "__main__":
    main()
