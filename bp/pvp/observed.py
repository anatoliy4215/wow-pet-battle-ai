"""What real PvP opponents bring: aggregates the user's BPLogger PvP battles into data/pvp/observed.json.

  python -m bp.pvp.observed [log files...]      (default: data/logs/archive/*.lua)

Per opponent pet: species, breed (inferred from its stats), ability choices that were seen.
TeamSampler adds these teams to the training opponents, so the model sees more of what the user
actually meets in the queue.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..data.collection import infer_breed
from ..data.lua import load_saved_variables
from ..sim.data import game_data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "pvp" / "observed.json"


def _as_list(x):
    if isinstance(x, dict):
        return [x[k] for k in sorted(x)]
    return list(x or [])


def pvp_battles(files):
    seen = set()
    for f in files:
        db = load_saved_variables(f)["BPLoggerDB"]
        for b in _as_list(db.get("battles")):
            key = (b.get("date"), b.get("zone"))
            if key in seen:
                continue
            seen.add(key)
            if b.get("pvp") and not b.get("wild") and b.get("start"):  # enemy is not an NPC and not wild
                yield b


LINK = re.compile(r"\|HbattlePetAbil:(\d+):(\d+):(\d+):(\d+)\|h")


def _used_abilities(b):
    """(ability id, caster max hp, power, speed) from every ability link in the combat log."""
    out = set()
    for r in _as_list(b.get("rounds")):
        for line in _as_list(r.get("log")):
            for m in LINK.finditer(line):
                out.add(tuple(int(x) for x in m.groups()))
    return out


def enemy_team(b):
    """Enemy pets with breed (from stats) and the ability choices revealed during the battle."""
    gd = game_data()
    start = _as_list(_as_list(b["start"]["pets"])[1])
    final = _as_list(_as_list(b.get("final", {}).get("pets", [[], []]))[1]) if b.get("final") else []
    used = _used_abilities(b)
    team = []
    for i, p in enumerate(start):
        s = p.get("species")
        if s not in gd.species:
            continue
        stats = (p.get("maxhp"), p.get("power"), p.get("speed"))
        breed = None
        for q in (p.get("quality"), (p.get("quality") or 4) - 1):
            breed = breed or infer_breed(s, p.get("level", 25), q, stats)
        abil = _as_list(p.get("abilities"))
        if i < len(final):
            abil = [a or f for a, f in zip(abil, _as_list(final[i].get("abilities")))]
        slots = gd.species[s]["abilities"]
        abil = [a if a in slots[k] else 0 for k, a in enumerate(abil[:3])]
        # hidden in PvP: take what the pet was seen using (link carries the caster's power and speed)
        for aid, _hp, power, speed in used:
            if (power, speed) != (p.get("power"), p.get("speed")):
                continue
            for k in range(3):
                if aid in slots[k] and not abil[k]:
                    abil[k] = aid
        choice = "".join(str(slots[k].index(a) + 1) if a else "?" for k, a in enumerate(abil))
        team.append({"species": s, "breed": breed, "choices": choice, "level": p.get("level")})
    return team


def main():
    files = [Path(x) for x in sys.argv[1:]] or sorted((ROOT / "data" / "logs" / "archive").glob("*.lua"))
    pets = defaultdict(lambda: {"count": 0, "breeds": Counter(), "choices": Counter()})
    teams = Counter()
    n = 0
    for b in pvp_battles(files):
        t = enemy_team(b)
        if len(t) != 3:
            continue
        n += 1
        for p in t:
            e = pets[p["species"]]
            e["count"] += 1
            if p["breed"]:
                e["breeds"][p["breed"]] += 1
            e["choices"][p["choices"]] += 1
        teams[tuple(sorted(p["species"] for p in t))] += 1
    out = {"battles": n,
           "pets": [{"species": s, "count": e["count"], "breeds": dict(e["breeds"]), "choices": dict(e["choices"])}
                    for s, e in sorted(pets.items(), key=lambda x: -x[1]["count"])],
           "teams": [{"species": list(k), "count": c} for k, c in teams.most_common()]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    gd = game_data()
    print(f"{n} PvP battles")
    for p in out["pets"][:20]:
        print(f"  {p['count']:4d}  {gd.species[p['species']]['name']}")


if __name__ == "__main__":
    main()
