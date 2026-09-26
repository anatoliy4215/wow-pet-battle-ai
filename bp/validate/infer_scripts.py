"""Infer NPC tamer move rotations from recorded battles.

NPC tamers follow fixed scripts. For pets whose wow-petsim script is missing or just "use(random)"
we extract the sequence of abilities each enemy pet started (multi-round abilities count once) and
find the shortest repeating cycle consistent with every recorded battle.

Output: data/processed/tamer_rotations.json  {npc: {pet_slot: {"rotation": [ability ids], "seen": n}}}
Usage:  python -m bp.validate.infer_scripts "data/logs/*.lua"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from ..data.lua import load_saved_variables
from ..sim.data import game_data
from . import replay as R

ROOT = Path(__file__).resolve().parents[2]


def n_turns(aid):
    ab = game_data().abilities.get(aid)
    return max(1, sum(1 for t in ab["turns"] if t["type"] == 0)) if ab else 1


def enemy_sequences(rec):
    """{enemy pet index: [ability id, ...]} of abilities started by each enemy pet, in order."""
    start = rec["start"]
    enemy = start["pets"][1]
    active = (start["active"][1] or 1) - 1
    seqs = {i: [] for i in range(len(enemy))}
    channel = 0
    for rnd in rec["rounds"]:
        lines = rnd.get("log", [])
        for raw in lines:
            m = R.re.match(r"^(.+) is now enemy active pet\.$", R.clean(raw))
            if m:
                for i, p in enumerate(enemy):
                    if p["name"] == m.group(1):
                        active = i
                        channel = 0
        if not any(l.startswith("Round") for l in lines):
            continue
        if channel > 0:
            channel -= 1
            continue
        pet = enemy[active]
        slots = game_data().species[pet["species"]]["abilities"]
        for raw in lines:
            m = R.LINK.search(raw)
            if not m:
                continue
            aid = int(m.group(1))
            if int(m.group(3)) == pet["power"] and any(aid in o for o in slots):
                seqs[active].append(aid)
                channel = n_turns(aid) - 1
                break
    return seqs


def shortest_cycle(seqs):
    """Shortest list c such that every observed sequence is a prefix of c repeated."""
    seqs = [s for s in seqs if s]
    if not seqs:
        return None
    longest = max(seqs, key=len)
    for k in range(1, len(longest) + 1):
        cyc = longest[:k]
        if all(all(s[i] == cyc[i % k] for i in range(len(s))) for s in seqs):
            return cyc
    return longest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    gd = game_data()
    by_npc = {}
    for f in [f for pat in args.files for f in glob.glob(pat)]:
        db = load_saved_variables(f)["BPLoggerDB"]
        for rec in db.get("battles", []):
            if not rec.get("start") or not rec.get("npc") or rec.get("pvp"):
                continue
            for i, seq in enemy_sequences(rec).items():
                by_npc.setdefault(int(rec["npc"]), {}).setdefault(i + 1, []).append(seq)
    out = {}
    for npc, pets in by_npc.items():
        for slot, seqs in pets.items():
            cyc = shortest_cycle(seqs)
            if cyc:
                out.setdefault(str(npc), {})[str(slot)] = {"rotation": cyc, "seen": sum(map(len, seqs))}
                names = " > ".join(gd.abilities[a]["name"] for a in cyc)
                print(f"npc {npc} pet {slot}: {names}   (moves seen: {sum(map(len, seqs))})")
    dest = ROOT / "data" / "processed" / "tamer_rotations.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
