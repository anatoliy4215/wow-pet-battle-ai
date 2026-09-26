"""PvP team builder: which 3 pets (and ability choices) to queue with, against the meta.

  python -m bp.pvp.teambuilder                 # from your collection (data/processed/collection.json)
  python -m bp.pvp.teambuilder --all           # from every pet in the game (rare, any breed)
  python -m bp.pvp.teambuilder --top 5 --battles 400

Pets listed in data/pvp/disabled.json are never suggested ({"species": ["Anomalus", 1234, ...]}).

1. candidate teams are scored by the network's value head at the lead pick against a fixed sample
   of opponent teams drawn like the training opponents (meta teams and meta pets weighted up)
2. an evolutionary search (mutate a pet / an ability choice / a breed) keeps the best teams
3. the best few are verified by real simulated battles, network vs network
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from ..sim.battle import Battle
from ..sim.data import PetSpec, game_data
from .env import PvPEnv
from .features import KEYS, PHASE_LEAD, encode
from .model import PvPNet, build, masked
from .pool import BREED_NAMES, TeamSampler, breeds_of, classes, load_disabled

ROOT = Path(__file__).resolve().parents[2]


def load_net(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    net = build(cfg).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


def candidate_pool(use_collection=True):
    """[(species, breed, quality, stats or None)] the player could bring."""
    disabled = load_disabled()
    gd = game_data()
    if use_collection:
        pets = json.loads((ROOT / "data" / "processed" / "collection.json").read_text(encoding="utf-8"))["pets"]
        out = []
        for p in pets:
            s = p["species"]
            if p["level"] < 25 or not p.get("battle", True) or s not in gd.species or s in disabled:
                continue
            if not gd.species[s]["battle"]:
                continue
            out.append((s, p.get("breed") or 3, p.get("quality", 3), tuple(p["stats"]) if p.get("stats") else None))
        return out
    return [(s, b, 3, None) for c in classes() for s in c[:1] if s not in disabled for b in breeds_of(s)]


def make_spec(pet, choices):
    s, breed, q, stats = pet
    return PetSpec.from_choices(s, choices, breed=breed, quality=q, stats=stats)


class Scorer:
    def __init__(self, net, opponents, device="cuda"):
        self.net, self.opps, self.device = net, opponents, device

    def values(self, teams, batch=2048):
        """Mean network value (-1..1) of each team at the lead pick, over all opponent teams."""
        obs = []
        for t in teams:
            for o in self.opps:
                b = Battle(t, o, seed=0)
                obs.append(encode(b, 0, PHASE_LEAD, hide=True))
        vals = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(obs), batch):
                chunk = obs[i:i + batch]
                x = {k: torch.as_tensor(np.stack([c[k] for c in chunk]), device=self.device) for k in KEYS}
                vals.append(self.net(x)[1].float().cpu().numpy())
        v = np.concatenate(vals).reshape(len(teams), len(self.opps))
        return v.mean(1), v


def evolve(scorer, pool, rng, pop=192, gens=40, elite=32):
    def rand_team():
        pets = rng.sample(pool, 3)
        while len({p[0] for p in pets}) < 3:
            pets = rng.sample(pool, 3)
        return [(p, "".join(rng.choice("12") for _ in range(3))) for p in pets]

    def mutate(team):
        t = list(team)
        k = rng.randrange(3)
        r = rng.random()
        if r < 0.45:  # another pet
            used = {p[0][0] for p in t}
            p = rng.choice(pool)
            while p[0] in used:
                p = rng.choice(pool)
            t[k] = (p, "".join(rng.choice("12") for _ in range(3)))
        else:  # another ability in one slot
            p, ch = t[k]
            j = rng.randrange(3)
            ch = ch[:j] + ("1" if ch[j] == "2" else "2") + ch[j + 1:]
            t[k] = (p, ch)
        return t

    def specs(team):
        return [make_spec(p, ch) for p, ch in team]

    popu = [rand_team() for _ in range(pop)]
    seen = {}
    for g in range(gens):
        todo = [t for t in popu if key(t) not in seen]
        if todo:
            v, _ = scorer.values([specs(t) for t in todo])
            for t, x in zip(todo, v):
                seen[key(t)] = (float(x), t)
        ranked = sorted(seen.values(), key=lambda x: -x[0])
        best = [t for _, t in ranked[:elite]]
        popu = best + [mutate(rng.choice(best)) for _ in range(pop - elite)]
    return sorted(seen.values(), key=lambda x: -x[0])


def key(team):
    return tuple(sorted((p[0], p[1], ch) for p, ch in team))


def battle_winrate(net, team, opponents, battles=200, device="cuda", seed=0):
    """Network vs network battles: our team (side 0) vs each opponent team in turn."""
    rng = random.Random(seed)
    envs = []
    for i in range(battles):
        e = PvPEnv(None, seed=seed * 7919 + i)
        e.reset((team, opponents[i % len(opponents)]))
        envs.append(e)
    while True:
        live = [(e, s) for e in envs if not e.done for s in e.pending()]
        if not live:
            break
        obs, masks = zip(*[e.observe(s) for e, s in live])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            x = {k: torch.as_tensor(np.stack([o[k] for o in obs]), device=device) for k in KEYS}
            logits = net(x)[0]
            a = torch.distributions.Categorical(logits=masked(logits, torch.as_tensor(np.stack(masks), device=device))).sample()
        acts = {}
        for (e, s), ai in zip(live, a.cpu().numpy()):
            acts.setdefault(id(e), (e, {}))[1][s] = int(ai)
        for e, act in acts.values():
            e.act(act)
    wins = sum(1 for e in envs if e.winner == 0)
    draws = sum(1 for e in envs if e.winner == -1)
    _ = rng
    return (wins + 0.5 * draws) / len(envs)


def describe(team):
    gd = game_data()
    out = []
    for p, ch in team:
        s, breed, _, _ = p
        sp = gd.species[s]
        abil = [gd.abilities[sp["abilities"][k][int(c) - 1]]["name"] for k, c in enumerate(ch)]
        out.append(f"{sp['name']} ({BREED_NAMES.get(breed, '?')}): {' / '.join(abil)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "pvp" / "main.pt"))
    ap.add_argument("--all", action="store_true", help="any pet in the game, not only the collection")
    ap.add_argument("--opponents", type=int, default=96)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--battles", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    net = load_net(args.ckpt)
    ts = TeamSampler(seed=args.seed, meta_pet_frac=0.6, meta_team_frac=0.4)
    opps = [ts.team() for _ in range(args.opponents)]
    pool = candidate_pool(not args.all)
    print(f"{len(pool)} candidate pets, {len(opps)} opponent teams")
    ranked = evolve(Scorer(net, opps), pool, rng)
    results = []
    for v, team in ranked[:args.top * 3]:
        specs = [make_spec(p, ch) for p, ch in team]
        wr = battle_winrate(net, specs, opps, battles=args.battles, seed=args.seed)
        results.append((wr, v, team))
    results.sort(key=lambda x: -x[0])
    for wr, v, team in results[:args.top]:
        print(f"\nпобед в симуляции {wr:.0%} (оценка сети {(v + 1) / 2:.0%})")
        for line in describe(team):
            print("  " + line)


if __name__ == "__main__":
    main()
