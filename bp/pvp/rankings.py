"""Best pets and best teams according to the current network, against the PvP meta.

  python -m bp.pvp.rankings [--ckpt checkpoints/pvp/actor.pt]   -> logs/pvp_rankings.json (shown on the dashboard)
  python -m bp.pvp.rankings --teams-only    teams again from the saved pet scores (after the exclusion list changed)

data/pvp/rank_exclude.json (edited on the dashboard): pets / breeds the user can't get. The model still
knows and meets them; they are only left out of the suggested teams and the pet list.

Pets: every battle-distinct species with each of its 8 ability choices, placed into teams with random
teammates; score = network's win estimate at the lead pick against meta-weighted opponent teams,
averaged. The best choice (and breed) per species is reported.
Teams: evolutionary search over the top pets (value head), the best ones checked by real simulated
battles network vs network against the same opponents.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np

from ..sim.data import game_data
from .pool import BREED_NAMES, TeamSampler, breeds_of, classes, load_disabled
from .teambuilder import Scorer, battle_winrate, describe, evolve, load_net, make_spec

ROOT = Path(__file__).resolve().parents[2]


def pet_scores(scorer, rng, mates=4):
    gd = game_data()
    reps = [c[0] for c in classes()]
    filler = [(s, rng.choice(breeds_of(s)), 3, None) for s in rng.sample(reps, 400)]
    out = []
    for n, s in enumerate(reps):
        breeds = breeds_of(s)
        combos = ["".join(c) for c in itertools.product("12", repeat=3)]
        teams, tags = [], []
        for ch in combos:
            for _ in range(mates):
                b = rng.choice(breeds)
                m1, m2 = rng.sample(filler, 2)
                teams.append([make_spec((s, b, 3, None), ch),
                              make_spec(m1, "".join(rng.choice("12") for _ in range(3))),
                              make_spec(m2, "".join(rng.choice("12") for _ in range(3)))])
                tags.append((ch, b))
        v, _ = scorer.values(teams)
        by_ch, by_br = {}, {}
        for (ch, b), x in zip(tags, v):
            by_ch.setdefault(ch, []).append(x)
        best = max(by_ch, key=lambda c: np.mean(by_ch[c]))
        # best breed: the best ability choice again, once with every breed and fixed teammates
        if len(breeds) > 1:
            m1, m2 = rng.sample(filler, 2)
            fixed = [make_spec(m1, "111"), make_spec(m2, "111")]
            vb, _ = scorer.values([[make_spec((s, b, 3, None), best)] + fixed for b in breeds])
            by_br = {b: float(x) for b, x in zip(breeds, vb)}
        best_breed = max(by_br, key=by_br.get) if by_br else breeds[0]
        sp = gd.species[s]
        out.append({"species": s, "name": sp["name"], "family": sp["family"], "choices": best,
                    "abilities": [gd.abilities[sp["abilities"][k][int(c) - 1]]["name"] for k, c in enumerate(best)],
                    "breed": BREED_NAMES.get(best_breed, "?"),
                    "breeds": [BREED_NAMES.get(b, "?") for b in breeds],
                    "breed_scores": {BREED_NAMES.get(b, "?"): x for b, x in by_br.items()},
                    "score": float((np.mean(by_ch[best]) + 1) / 2), "avg": float((np.mean(v) + 1) / 2)})
        if n % 200 == 0:
            print(f"pets {n}/{len(reps)}", flush=True)
    out.sort(key=lambda r: -r["score"])
    return out


def load_exclude():
    """{species: None (all breeds) | {breed names}} from data/pvp/rank_exclude.json."""
    path = ROOT / "data" / "pvp" / "rank_exclude.json"
    out = {}
    if path.exists():
        for e in json.loads(path.read_text(encoding="utf-8")).get("items", []):
            s, b = int(e["species"]), e.get("breed")
            if b is None or out.get(s, set()) is None:
                out[s] = None
            else:
                out.setdefault(s, set()).add(b)
    return out


def allowed_breeds(species, excl):
    names = [BREED_NAMES.get(b, "?") for b in breeds_of(species)]
    if species not in excl:
        return names
    if excl[species] is None:
        return []
    return [b for b in names if b not in excl[species]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "pvp" / "actor.pt"))
    ap.add_argument("--opponents", type=int, default=32)
    ap.add_argument("--teams-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(1)
    net = load_net(args.ckpt)
    import torch
    upd = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("update")
    ts = TeamSampler(seed=7, meta_pet_frac=0.6, meta_team_frac=0.4)
    opps = [ts.team() for _ in range(args.opponents)]
    scorer = Scorer(net, opps)
    cache = ROOT / "logs" / "pvp_pet_scores.json"
    if args.teams_only and cache.exists() and json.loads(cache.read_text(encoding="utf-8")).get("update") == upd:
        pets = json.loads(cache.read_text(encoding="utf-8"))["pets"]
    else:
        pets = pet_scores(scorer, rng)
        cache.write_text(json.dumps({"update": upd, "pets": pets}, ensure_ascii=False), encoding="utf-8")
    disabled = load_disabled()
    excl = load_exclude()
    top = [p for p in pets if p["species"] not in disabled and allowed_breeds(p["species"], excl)][:150]
    pool = [(p["species"], b, 3, None) for p in top for b in breeds_of(p["species"])
            if BREED_NAMES.get(b, "?") in allowed_breeds(p["species"], excl)]
    ranked = evolve(scorer, pool, rng, pop=256, gens=40, elite=48)
    teams = []
    seen = set()
    for v, team in ranked:
        key = frozenset(p[0] for p, _ in team)
        if key in seen:
            continue
        seen.add(key)
        specs = [make_spec(p, ch) for p, ch in team]
        wr = battle_winrate(net, specs, opps, battles=192, seed=3)
        teams.append({"pets": describe(team), "value": float((v + 1) / 2), "winrate": wr,
                      "specs": [[p[0], BREED_NAMES.get(p[1], "?")] for p, _ in team]})
        if len(teams) >= 40:
            break
    teams.sort(key=lambda t: -t["winrate"])
    out = {"update": upd, "time": time.time(), "pets": pets[:400], "teams": teams,
           "opponents": "мета-команды и частые питомцы (как в обучении, с повышенным весом меты)"}
    (ROOT / "logs" / "pvp_rankings.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
