"""Best team from the player's collection for every (tamer, required family) achievement task.

For each task: sample candidate teams (3 pets of the required family, random ability choices),
play each a few times with the trained policy, then re-check the best candidates with more games.

Usage:  python -m bp.rl.recommend [--ckpt checkpoints/ppo_ach.pt] [--candidates 48] [--games 8]
Output: data/processed/recommendations.json and logs/recommendations.html
"""
from __future__ import annotations

import argparse
import html
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _init(ckpt):
    global NET
    import torch
    torch.set_num_threads(1)
    from .evaluate import load_policy
    NET = load_policy(ckpt, "cpu")


def _team_from(members, choices):
    from ..sim.data import PetSpec
    return [PetSpec.from_choices(x["species"], c, level=x["level"], quality=x["quality"], breed=x["breed"] or 3,
                                 stats=tuple(x["stats"])) for x, c in zip(members, choices)]


def _winrate(tid, team, games, seed):
    from .env import PetBattleEnv, ScenarioSampler
    from .evaluate import play_batch
    sampler = _SAMPLER
    envs = []
    for g in range(games):
        e = PetBattleEnv(sampler, seed=seed * 1000 + g)
        e.reset((tid, team))
        envs.append(e)
    return float(np.mean(play_batch(NET, envs, "cpu")))


def solve(task, candidates=48, games=8, final_games=40):
    global _SAMPLER
    from .env import ScenarioSampler
    fam, tid = task
    if "_SAMPLER" not in globals():
        _SAMPLER = ScenarioSampler(seed=0)
    pool = [x for _, x in _SAMPLER._pool(fam)]
    rng = random.Random(hash(task) & 0xFFFF)
    if len(pool) < 3:
        return {"family": fam, "tamer": tid, "error": "not enough level-25 pets of this family"}
    cands = []
    for i in range(candidates):
        members = rng.sample(pool, 3)
        choices = ["".join(rng.choice("12") for _ in range(3)) for _ in members]
        wr = _winrate(tid, _team_from(members, choices), games, i)
        cands.append((wr, members, choices))
    cands.sort(key=lambda c: -c[0])
    best = None
    for wr, members, choices in cands[:3]:
        final = _winrate(tid, _team_from(members, choices), final_games, 777)
        if best is None or final > best[0]:
            best = (final, members, choices)
    wr, members, choices = best
    team = _team_from(members, choices)
    from ..sim.data import game_data
    gd = game_data()
    return {"family": fam, "tamer": tid, "winrate": wr,
            "team": [{"species": s.species, "name": gd.species[s.species]["name"],
                      "abilities": [gd.abilities[a]["name"] for a in s.abilities if a],
                      "ability_ids": s.abilities, "choices": c, "breed": x["breed"]}
                     for s, c, x in zip(team, choices, members)]}


def _solve_star(args):
    return solve(*args)


def render(results, path):
    from ..sim import constants as C
    from ..sim.data import game_data
    gd = game_data()
    ach = json.loads((ROOT / "data" / "processed" / "achievements.json").read_text(encoding="utf-8"))
    by_task = {(r["family"], r["tamer"]): r for r in results}
    sections = []
    for a in ach:
        if a["kind"] not in ("family_tamer", "tamer_list"):
            continue
        rows = []
        for t in a["tamers"]:
            if t.get("tamer_id") is None:
                continue
            fam = t.get("required_family_override", a["required_family"])
            r = by_task.get((fam, int(t["tamer_id"])))
            if not r:
                continue
            name = gd.tamers[int(t["tamer_id"])]["name"]
            if "error" in r:
                rows.append(f"<tr><td>{html.escape(t['name'])}</td><td colspan=2 class=muted>{html.escape(r['error'])}</td></tr>")
                continue
            team = "<br>".join(f"<b>{html.escape(p['name'])}</b> <span class=muted>{html.escape(' / '.join(p['abilities']))}</span>"
                               for p in r["team"])
            cls = "good" if r["winrate"] >= 0.8 else ("mid" if r["winrate"] >= 0.5 else "bad")
            rows.append(f"<tr><td>{html.escape(t['name'])}<div class=muted>{html.escape(name)}</div></td>"
                        f"<td>{team}</td><td class='num {cls}'>{r['winrate']:.0%}</td></tr>")
        if rows:
            fam = a["required_family"]
            sections.append(f"<section class=card><h2>{html.escape(a['name_ru'] or a['name_en'])}</h2>"
                            f"<div class=muted>{html.escape(a['name_en'])} · семейство: "
                            f"{C.FAMILY_NAMES[fam] if fam is not None else 'любое'}</div>"
                            f"<table><tr><th>Тренер</th><th>Команда (способности)</th><th class=num>Победы</th></tr>"
                            f"{''.join(rows)}</table></section>")
    page = f"""<!doctype html><html lang=ru><head><meta charset=utf-8><title>Команды для достижений</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --card:#1f2024; --fg:#ececec; --muted:#9a9aa2; --line:#2e2f34; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,Segoe UI,sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:900px; margin:auto; }} h1 {{ font-size:22px; }} h2 {{ font-size:17px; margin:0; }}
.muted {{ color:var(--muted); font-size:13px; }} .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:14px 0; }}
table {{ width:100%; border-collapse:collapse; margin-top:8px; }} td,th {{ padding:6px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }} .good {{ color:#1f9d55; }} .mid {{ color:#c08a00; }} .bad {{ color:#d0453a; }}
</style></head><body><main><h1>Команды для незакрытых достижений</h1>
<p class=muted>Лучшая найденная команда из твоей коллекции против каждого тренера и доля побед в симуляторе (40 боёв).
Способности указаны в порядке ячеек 1/2/3. Сгенерировано {time.strftime('%Y-%m-%d %H:%M')}.</p>
{''.join(sections)}</main></body></html>"""
    path.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ppo_ach.pt")
    ap.add_argument("--candidates", type=int, default=48)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from .env import achievement_tasks
    tasks = achievement_tasks()[:args.limit]
    out = ROOT / "data" / "processed" / "recommendations.json"
    results = []
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(str(ROOT / args.ckpt),)) as pool:
        for i, r in enumerate(pool.imap_unordered(_solve_star, [(t, args.candidates, args.games) for t in tasks]), 1):
            results.append(r)
            if i % 25 == 0 or i == len(tasks):
                out.write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")
                render(results, ROOT / "logs" / "recommendations.html")
                print(f"{i}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    wr = [r["winrate"] for r in results if "winrate" in r]
    print(f"done: {len(results)} tasks, >=80%: {sum(w >= 0.8 for w in wr)}, >=50%: {sum(w >= 0.5 for w in wr)}")


if __name__ == "__main__":
    main()
