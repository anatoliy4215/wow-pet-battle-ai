"""Stage 2 self-play actors: every decision is improved by a one-round look-ahead in the simulator.

For the side s that has to decide:
  worlds  K copies of the battle in which the opponent's still-hidden abilities are sampled
          (from what the network believes about them - the real ones are never looked at)
  reply   in every world the opponent's move is sampled from the network (from its own point of view)
  Q(a)    mean network value for s after playing a against that reply, with fresh dice, in all worlds
  target  pi_t(a) ~ pi(a) * exp(Q(a) / tau)   - a better policy that stays close to the network
The move actually played is drawn from pi_t.  Samples (observation, pi_t, Q, final result, auxiliary
targets) are written as .npz shards; bp.pvp.learner trains the network on them.
Value target zd = result * GAMMA ** (rounds until the end): a win now is worth more than a win in 15
rounds, so in a won (or lost) position the moves still differ in value instead of all being "100%".
Part of the games start from practice positions (bp.pvp.scenarios): Undead / Mechanical second
lives, control against a Critter or a Resilient pet.

  python -m bp.pvp.actor --weights checkpoints/pvp/actor.pt --out data/pvp/shards/local --workers 14
It runs the same way on a rented box, whose own GPU then does the network work.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np

KEYS = ["glob", "pet_int", "pet_float", "ab_int", "ab_float", "ab_mask", "aura_int", "aura_float", "aura_mask"]
GAMMA = 0.98
DTYPES = {"glob": np.float16, "pet_int": np.int8, "pet_float": np.float16, "ab_int": np.int16, "ab_float": np.float16,
          "ab_mask": np.uint8, "aura_int": np.int16, "aura_float": np.float16, "aura_mask": np.uint8}


# ------------------------------------------------------------------------------------------ worker side
class Search:
    """Look-ahead for one side of one env; advanced by the worker in two network round trips."""

    def __init__(self, env, side, k, rng, lo_prev):
        from .features import action_mask, encode
        from .search import clone, determinize
        b = env.battle
        self.side, self.phase = side, env.phase
        self.obs = encode(b, side, env.phase)
        self.mask = action_mask(b, side, env.phase)
        self.legal = [a for a in range(7) if self.mask[a] > 0]
        self.worlds, self.opp_obs = [], []
        self.opp_acts = env.phase != 2 or b.needs_replacement(1 - side)  # does the opponent move too
        if len(self.legal) > 1:
            lo = lo_prev if lo_prev is not None else np.full(9, 0.5)
            for _ in range(k):
                w = clone(b, rng.getrandbits(32))
                determinize(w, side, lo, rng)
                self.worlds.append(w)
                if self.opp_acts:
                    self.opp_obs.append((encode(w, 1 - side, env.phase), action_mask(w, 1 - side, env.phase)))

    def root_requests(self):
        return [(self.obs, self.mask)] + self.opp_obs

    def expand(self, replies, rng):
        """replies: network outputs for root_requests(). Plays every legal action in every world."""
        from .features import action_mask, encode
        from .search import _apply, _phase, clone
        self.prior, _, self.lo = replies[0]
        self.states, self.tags = [], []
        if len(self.legal) <= 1:
            return []
        for j, w in enumerate(self.worlds):
            b_j = None
            if self.opp_acts:
                p_opp = replies[1 + j][0]
                b_j = int(rng.choices(range(7), p_opp)[0])
            # common random numbers: every action of a world gets the same reply and the same dice,
            # so Q differences between actions are not drowned in hit/crit luck
            seed_j = rng.getrandbits(32)
            for a in self.legal:
                c = clone(w, seed_j)
                acts = {self.side: a}
                if b_j is not None:
                    acts[1 - self.side] = b_j
                _apply(c, self.phase, acts)
                self.states.append(c)
                self.tags.append(a)
        self.value_req = []
        self.fixed = {}
        for i, c in enumerate(self.states):
            if c.over:
                self.fixed[i] = 0.0 if c.winner not in (0, 1) else (1.0 if c.winner == self.side else -1.0)
            else:
                self.value_req.append(i)
        return [(encode(self.states[i], self.side, _phase(self.states[i])),
                 action_mask(self.states[i], self.side, _phase(self.states[i]))) for i in self.value_req]

    def finish(self, replies, tau, rng):
        q = np.zeros(7, dtype=np.float32)
        if len(self.legal) <= 1:
            pi = np.zeros(7, dtype=np.float32)
            pi[self.legal[0]] = 1.0
            self.pi, self.q = pi, q
            return self.legal[0]
        vals = np.zeros(len(self.states), dtype=np.float32)
        for i, v in self.fixed.items():
            vals[i] = v
        for i, r in zip(self.value_req, replies):
            vals[i] = r[1]
        tags = np.array(self.tags)
        for a in self.legal:
            q[a] = vals[tags == a].mean()
        logits = np.full(7, -1e9)
        lp = np.log(np.maximum(self.prior, 1e-8))
        for a in self.legal:
            logits[a] = lp[a] + q[a] / tau
        pi = np.exp(logits - logits.max())
        pi /= pi.sum()
        self.pi, self.q = pi.astype(np.float32), q
        return int(rng.choices(range(7), pi)[0])


def worker(remote, wid, n_envs, seed, cfg):
    from .env import PvPEnv
    from .pool import TeamSampler
    rng = random.Random(seed)
    sampler = TeamSampler(seed=seed, meta_pet_frac=cfg["meta_pet_frac"], meta_team_frac=cfg["meta_team_frac"])
    envs = [PvPEnv(sampler, seed=seed * 1000 + i) for i in range(n_envs)]
    from .scenarios import start as scenario_start

    def new_game(e):
        if rng.random() < cfg.get("practice", 0.0):
            scenario_start(e, sampler, rng)
        else:
            e.reset()

    for e in envs:
        new_game(e)
    lo_prev = [[None, None] for _ in envs]
    traj = [[[], []] for _ in envs]
    searches = [None] * n_envs      # per env: {side: Search}
    stage = ["root"] * n_envs
    done_samples = []
    games = []
    out_dir = Path(cfg["out"])
    shard_n = [0]

    def flush(force=False):
        if not done_samples or (len(done_samples) < cfg["shard"] and not force):
            return
        data = {k: np.stack([s["obs"][k] for s in done_samples]).astype(DTYPES[k]) for k in KEYS}
        data["mask"] = np.stack([s["mask"] for s in done_samples]).astype(np.uint8)
        data["pi"] = np.stack([s["pi"] for s in done_samples]).astype(np.float16)
        data["q"] = np.stack([s["q"] for s in done_samples]).astype(np.float16)
        data["z"] = np.array([s["z"] for s in done_samples], dtype=np.int8)
        data["zd"] = np.array([s["zd"] for s in done_samples], dtype=np.float16)
        data["opp_act"] = np.array([s["opp_act"] for s in done_samples], dtype=np.int8)
        data["lo_y"] = np.stack([s["lo_y"] for s in done_samples]).astype(np.uint8)
        name = f"w{wid:02d}_{int(time.time())}_{shard_n[0]:05d}"
        tmp = out_dir / (name + ".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(out_dir / (name + ".npz"))
        shard_n[0] += 1
        done_samples.clear()

    def end_game(i):
        e = envs[i]
        w = e.winner
        for side in (0, 1):
            y, _ = e.loadout_truth(side)
            z = 0 if w == -1 else (1 if w == side else -1)
            end = e.battle.round
            for s in traj[i][side]:
                s["z"], s["lo_y"] = z, y
                s["zd"] = z * GAMMA ** max(0, end - s["rnd"])
                done_samples.append(s)
        games.append((w, e.battle.round))
        new_game(e)
        traj[i] = [[], []]
        lo_prev[i] = [None, None]
        flush()

    while True:
        reqs = []  # (env, side, kind)
        obs, masks = [], []
        for i, e in enumerate(envs):
            if stage[i] == "root":
                if e.done:
                    end_game(i)
                searches[i] = {s: Search(e, s, cfg["k"], rng, lo_prev[i][s]) for s in e.pending()}
                for s, sr in searches[i].items():
                    for o, m in sr.root_requests():
                        reqs.append((i, s))
                        obs.append(o)
                        masks.append(m)
            else:
                for s, sr in searches[i].items():
                    for o, m in sr.pending_values:
                        reqs.append((i, s))
                        obs.append(o)
                        masks.append(m)
        batch = {k: np.stack([o[k] for o in obs]) for k in KEYS} if obs else None
        remote.send(("req", batch, np.stack(masks) if masks else None, len(done_samples), games))
        games = []
        cmd, data = remote.recv()
        if cmd == "close":
            flush(force=True)
            return
        pol, val, lo = data
        # hand the replies back to their searches, in request order
        replies = {}
        for j, key in enumerate(reqs):
            replies.setdefault(key, []).append((pol[j], float(val[j]), lo[j]))
        for i, e in enumerate(envs):
            if stage[i] == "root":
                for s, sr in searches[i].items():
                    sr.pending_values = sr.expand(replies.get((i, s), []), rng)
                    lo_prev[i][s] = sr.lo
                stage[i] = "values"
            else:
                acts = {}
                for s, sr in searches[i].items():
                    acts[s] = sr.finish(replies.get((i, s), []), cfg["tau"], rng)
                for s, sr in searches[i].items():
                    traj[i][s].append({"obs": sr.obs, "mask": sr.mask, "pi": sr.pi, "q": sr.q, "rnd": e.battle.round,
                                       "opp_act": acts.get(1 - s, -1) if e.phase == 1 else -1})
                e.act(acts)
                stage[i] = "root"


# ------------------------------------------------------------------------------------------ main side
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="network file, reloaded whenever it changes")
    ap.add_argument("--out", required=True, help="directory for .npz sample shards")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--k", type=int, default=8, help="worlds per decision")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--shard", type=int, default=4096)
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = never)")
    ap.add_argument("--practice", type=float, default=0.2, help="share of games started from practice positions")
    args = ap.parse_args()

    import torch
    from .model import load as load_net, masked
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, mtime = None, 0.0

    def load():
        nonlocal net, mtime
        m = os.path.getmtime(args.weights)
        if m == mtime:
            return
        ck = torch.load(args.weights, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        n = load_net(ck, device)
        n.eval()
        net, mtime = n, m
        print(f"weights loaded (update {ck.get('update')})", flush=True)

    while not os.path.exists(args.weights):  # the learner publishes the first weights
        print("waiting for weights...", flush=True)
        time.sleep(10)
    load()
    cfg = {"k": args.k, "tau": args.tau, "shard": args.shard, "out": str(out), "meta_pet_frac": 0.35, "meta_team_frac": 0.15,
           "practice": args.practice}
    ctx = mp.get_context("spawn")
    remotes, procs = [], []
    for w in range(args.workers):
        a, b = ctx.Pipe()
        seed = int(time.time()) % 100000 * 100 + w
        p = ctx.Process(target=worker, args=(b, w, args.envs, seed, cfg), daemon=True)
        p.start()
        remotes.append(a)
        procs.append(p)
    t0 = last_check = last_log = time.time()
    n_games = 0
    from multiprocessing.connection import wait as mp_wait
    # asynchronous: run the network for whichever workers are ready (at least half, or after a short
    # wait) and let the others keep simulating - the GPU and the CPUs no longer take turns
    pending = {}
    first_t = None
    while True:
        for r in mp_wait([r for r in remotes if r not in pending], timeout=0.005):
            pending[r] = r.recv()
            if first_t is None:
                first_t = time.time()
        if not pending or (len(pending) < max(1, len(remotes) // 2) and time.time() - first_t < 0.02):
            continue
        batch_r = list(pending)
        msgs = [pending.pop(r) for r in batch_r]
        first_t = None
        for m in msgs:
            n_games += len(m[4])
        sizes = [0 if m[1] is None else len(m[2]) for m in msgs]
        parts = [m for m in msgs if m[1] is not None]
        if parts:
            obs = {k: np.concatenate([m[1][k] for m in parts]) for k in KEYS}
            mask = np.concatenate([m[2] for m in parts])
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                pol_all, val_all, lo_all = [], [], []
                for s in range(0, len(mask), 4096):
                    x = {k: torch.as_tensor(obs[k][s:s + 4096], device=device) for k in KEYS}
                    m_ = torch.as_tensor(mask[s:s + 4096], device=device)
                    logits, v, _, lo = net(x)
                    pol_all.append(torch.softmax(masked(logits, m_), -1).float().cpu().numpy())
                    val_all.append(v.float().cpu().numpy())
                    lo_all.append(torch.sigmoid(lo.float()).cpu().numpy())
            pol, val, lo = np.concatenate(pol_all), np.concatenate(val_all), np.concatenate(lo_all)
        off = 0
        for r, n in zip(batch_r, sizes):
            r.send(("reply", (pol[off:off + n], val[off:off + n], lo[off:off + n]) if n else (None, None, None)))
            off += n
        now = time.time()
        if now - last_check > 30:
            last_check = now
            try:
                load()
            except Exception as ex:  # the learner may be writing the file right now
                print("reload failed:", ex, flush=True)
        if now - last_log > 60:
            last_log = now
            shards = len(list(out.glob("*.npz")))
            print(f"{(now - t0) / 60:.0f} min: {n_games} games, {shards} shards waiting", flush=True)
        stop_file = Path(__file__).resolve().parents[2] / "logs" / "STOP_ACTORS"
        if (args.hours and now - t0 > args.hours * 3600) or (stop_file.exists() and stop_file.stat().st_mtime > t0):
            for r in remotes:
                r.send(("close", None))
            for p in procs:  # let every worker write its last partial shard
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
            break


if __name__ == "__main__":
    main()
