"""End-to-end battle preparation for (tamer, family): team search + rule search + written guide.

1. sample candidate teams from the collection (required family), score them with the policy
2. for the best candidates, search priority rules (all ability orders per pet, both bench orders)
   played by a simple rule controller - what a human can follow
3. write a Russian guide with exact cooldowns and an explicit fallback per pet

Usage:  python -m bp.rl.prepare --tamer 528 --families 0,1,2 [--candidates 400] [--top 5]
Output: data/processed/plans/<tamer>_<family>.json and .md
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "plans"


def _setup_maps():
    from ..sim.data import game_data
    gd = game_data()
    weather, debuff = {}, {}
    for aid, a in gd.abilities.items():
        for t in a["turns"]:
            for e in t["effects"]:
                if e["op"] == 80:
                    weather[aid] = e["aura"]
                if e["op"] in (50, 54) and e["aura"]:
                    debuff.setdefault(aid, e["aura"])
    return weather, debuff


def _cond(aid, weather, debuff):
    if aid in weather:
        w = weather[aid]
        return lambda b, p, w=w: b.weather is None or b.weather.aid != w
    if aid in debuff:
        au = debuff[aid]
        # the effect may sit on the enemy pet or on the enemy team pad (Flamethrower field)
        return lambda b, p, au=au: not (b.teams[1].active_pet.has_aura(au) or any(x.aid == au for x in b.teams[1].auras))
    return None


def _cond_text(aid, weather, debuff):
    from ..sim.data import game_data
    gd = game_data()
    if aid in weather:
        return f"сейчас нет погоды «{gd.abilities[weather[aid]]['name']}»"
    if aid in debuff:
        return f"на противнике нет эффекта «{gd.abilities[debuff[aid]]['name']}»"
    return None


def _eval_rules(args):
    tamer, members, choices, perms, order, games, seed = args
    from ..sim.data import game_data
    from .recommend import _team_from
    from .rules import winrate
    weather, debuff = _setup_maps()
    team = _team_from(members, choices)
    gd = game_data()
    rules = {}
    for spec, perm in zip(team, perms):
        abil = [spec.abilities[j] for j in perm]
        rules[gd.species[spec.species]["name"]] = [(gd.abilities[a]["name"], _cond(a, weather, debuff)) for a in abil]
    names = [gd.species[s.species]["name"] for s in team]
    wr, rounds = winrate(tamer, team, rules, [names[i] for i in order], games=games)
    return wr, rounds, perms, order


def _init_policy(ckpt):
    from . import recommend as R
    from .env import ScenarioSampler
    R._init(ckpt)
    R._SAMPLER = ScenarioSampler(seed=0)


def _score_team(args):
    tamer, members, choices, games, seed = args
    from . import recommend as R
    return R._winrate(tamer, R._team_from(members, choices), games, seed), members, choices


def prepare(tamer, family, pool, candidates=400, top=5, workers=14, ckpt=None):
    from ..sim.data import game_data
    from .env import ScenarioSampler
    gd = game_data()
    fam_pool = [x for _, x in ScenarioSampler(seed=0)._pool(family)]
    rng = random.Random(tamer * 100 + (family if family is not None else 11))
    jobs = []
    for i in range(candidates):
        m = rng.sample(fam_pool, 3)
        ch = ["".join(rng.choice("12") for _ in range(3)) for _ in m]
        jobs.append((tamer, m, ch, 6, i))
    scored = pool.map(_score_team, jobs, chunksize=4)
    scored.sort(key=lambda x: -x[0])
    best = None
    P = list(itertools.permutations(range(3)))
    for model_wr, members, choices in scored[:top]:
        rjobs = [(tamer, members, choices, (a, b, c), order, 100, 0)
                 for a in P for b in P for c in P for order in ((1, 2), (2, 1))]
        res = pool.map(_eval_rules, rjobs, chunksize=8)
        res.sort(key=lambda r: -r[0])
        final = pool.map(_eval_rules, [(tamer, members, choices, r[2], r[3], 600, 1) for r in res[:4]])
        final.sort(key=lambda r: -r[0])
        wr, rounds, perms, order = final[0]
        if best is None or wr > best["winrate"]:
            best = {"tamer": tamer, "family": family, "winrate": wr, "rounds": rounds, "model_wr": model_wr,
                    "members": members, "choices": choices, "perms": perms, "order": order}
    return best


def guide_md(plan):
    from ..sim import constants as C
    from ..sim.data import game_data
    from .recommend import _team_from
    gd = game_data()
    weather, debuff = _setup_maps()
    team = _team_from(plan["members"], plan["choices"])
    t = gd.tamers[plan["tamer"]]
    fam = C.FAMILY_NAMES[plan["family"]] if plan["family"] is not None else "любое"
    breeds = {3: "B/B", 4: "P/P", 5: "S/S", 6: "H/H", 7: "H/P", 8: "P/S", 9: "H/S", 10: "P/B", 11: "S/B", 12: "H/B"}
    lines = [f"## {t['name']}: {fam}", "",
             f"Победы в симуляторе: **{plan['winrate']:.0%}**, бой ~{plan['rounds']:.0f} раундов.", "",
             "| # | Питомец | Способности (1 / 2 / 3) |", "|---|---|---|"]
    for i, (s, m) in enumerate(zip(team, plan["members"]), 1):
        ab = " / ".join(gd.abilities[a]["name"] for a in s.abilities)
        lines.append(f"| {i} | **{gd.species[s.species]['name']}** ({breeds.get(m['breed'], '?')}) | {ab} |")
    names = [gd.species[s.species]["name"] for s in team]
    order = [0] + list(plan["order"])
    for k, idx in enumerate(order):
        s = team[idx]
        head = ["**Начинает бой:**", "**Когда он погибнет — выводи этого** (" + str(idx + 1) + "-й в команде):",
                "**Последним:**"][k]
        lines += ["", f"{head} **{names[idx]}**"]
        perm = plan["perms"][idx]
        for n, j in enumerate(perm, 1):
            aid = s.abilities[j]
            ab = gd.abilities[aid]
            conds = []
            if ab["cooldown"]:
                conds.append(f"если доступна (перезарядка {ab['cooldown']})")
            ct = _cond_text(aid, weather, debuff)
            if ct:
                conds.append(f"если {ct}")
            lines.append(f"{n}. **{ab['name']}**" + (" — " + ", ".join(conds) if conds else ""))
            if not conds:
                lines[-1] = f"- только **{ab['name']}**, каждый раунд" if n == 1 else f"{n}. иначе **{ab['name']}**"
                break  # always usable: anything below is never reached
        else:
            # every option is conditional: the rule controller then uses the first usable ability by slot
            free = [gd.abilities[a]["name"] for a in s.abilities if a and not gd.abilities[a]["cooldown"]]
            lines.append(f"Если ничего из этого нельзя — **{free[0]}**." if free
                         else "Если всё на перезарядке — пропусти ход (Pass).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamer", type=int, required=True)
    ap.add_argument("--families", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--candidates", type=int, default=400)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "ppo_ach.pt"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fams = [None if f == "any" else int(f) for f in args.families.split(",")]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_policy, initargs=(args.ckpt,)) as pool:
        for fam in fams:
            plan = prepare(args.tamer, fam, pool, args.candidates, args.top, args.workers, args.ckpt)
            base = OUT / f"{args.tamer}_{fam}"
            base.with_suffix(".json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
            md = guide_md(plan)
            base.with_suffix(".md").write_text(md, encoding="utf-8")
            print(f"family {fam}: {plan['winrate']:.0%}", flush=True)


if __name__ == "__main__":
    main()
