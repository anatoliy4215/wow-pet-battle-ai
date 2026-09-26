"""Compare the scripted tamer AI with the enemy moves actually recorded in logs.

Usage:  python -m bp.validate.ai_check "data/logs/*.lua"
"""
import argparse
import glob
from collections import Counter

from ..data.lua import load_saved_variables
from ..sim import constants as C
from ..sim.data import game_data
from . import replay as R


def tamer_for(rec):
    gd = game_data()
    npc = int(rec["npc"]) if rec.get("npc") else None
    for tid, t in gd.tamers.items():
        if t["npc"] == npc:
            return tid
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    gd = game_data()
    from ..sim.run import tamer_specs
    for f in [f for pat in args.files for f in glob.glob(pat)]:
        db = load_saved_variables(f)["BPLoggerDB"]
        for i, rec in enumerate(db.get("battles", [])):
            tid = tamer_for(rec)
            if not rec.get("start") or tid is None:
                continue
            _, ctrl = tamer_specs(tid)
            b = R.build_battle(rec)
            prev = rec["start"]
            stats = Counter()
            seq = []
            for rnd in rec["rounds"]:
                lines = rnd.get("log", [])
                if not any(l.startswith("Round") for l in lines):
                    a = R.our_action(b, rnd)
                    e = R.enemy_swap_anywhere(b, lines)
                    na, ne = b.needs_replacement(0), b.needs_replacement(1)
                    if (na and a is not None) or (ne and e is not None):
                        b.step(a if na and a is not None else C.A_PASS, e if ne and e is not None else C.A_PASS)
                    prev = rnd.get("after") or prev
                    continue
                R.resync(b, prev)
                if (rnd.get("action") or [None])[0] == "forfeit":
                    break
                a = R.our_action(b, rnd)
                if a is None:
                    a = C.A_PASS
                real = R.enemy_action(b, lines)
                if b.needs_replacement(1) or b.needs_replacement(0):
                    real = R.enemy_swap_anywhere(b, lines) or C.A_PASS
                    b.step(a if b.needs_replacement(0) else C.A_PASS, real if b.needs_replacement(1) else C.A_PASS)
                    prev = rnd.get("after") or prev
                    continue
                if b.legal_actions(1) != [C.A_PASS]:
                    pred = ctrl.choose(b, 1)
                    pet = b.teams[1].active_pet
                    name = lambda x: gd.abilities[pet.abilities[x]]["name"] if x in C.A_ABILITY and pet.abilities[x] else str(x)
                    stats["match" if pred == real else "differ"] += 1
                    seq.append(f"{name(real)}{'' if pred == real else ' (AI: ' + name(pred) + ')'}")
                if a not in b.legal_actions(0):
                    a = C.A_PASS
                b.step(a, real if real in b.legal_actions(1) else C.A_PASS)
                prev = rnd.get("after") or prev
                if b.over:
                    break
            t = gd.tamers[tid]
            print(f"battle {i + 1} {t['name']}: {dict(stats)}")
            print("   real moves:", " > ".join(seq))


if __name__ == "__main__":
    main()
