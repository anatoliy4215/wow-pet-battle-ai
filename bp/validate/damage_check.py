"""Check every damage / heal line of recorded battles against the simulator formula.

Each combat-log hyperlink carries the caster's live stats: |HbattlePetAbil:ABILITY:MAXHP:POWER:SPEED|h,
so we can compute the expected value range of every hit without replaying the whole battle.

Usage:  python -m bp.validate.damage_check data/logs/BPLogger_*.lua [--show-bad]
"""
from __future__ import annotations

import argparse
import glob
import re
from collections import Counter

from ..data.lua import load_saved_variables
from ..sim import constants as C
from ..sim.data import game_data
from ..sim.ops import family_mult

LINK = r"\|HbattlePetAbil:(\d+):(\d+):(\d+):(\d+)\|h\[([^\]]*)\]\|h"
DMG = re.compile(LINK + r".*? dealt (\d+) damage to (your|enemy) (?:\|T[^|]*\|t)?(.+?)(?: \(([^)]*)\))?\.$")
HEAL = re.compile(LINK + r".*? healed (\d+) damage from (your|enemy) (?:\|T[^|]*\|t)?(.+?)\.$")
DMG_OPS = {24, 222, 29, 169, 96, 384, 385, 226, 65, 170, 27, 358, 179}
HEAL_OPS = {23, 104, 171, 100, 254}


def ability_rows(aid, ops, seen=None):
    """All (points, variance, periodic) of rows with the given opcodes, including applied auras."""
    gd = game_data()
    seen = seen or set()
    if aid in seen or aid not in gd.abilities:
        return []
    seen.add(aid)
    rows = []
    for t in gd.abilities[aid]["turns"]:
        for e in t["effects"]:
            p = e["params"]
            if e["op"] in ops and p.get("Points"):
                rows.append((p["Points"] + p.get("BonusPoints", 0) * 0, p.get("Variance", 0), bool(p.get("IsPeriodic")) or t["type"] == 0 and gd.abilities[aid]["family"] < 0))
                if p.get("BonusPoints"):
                    rows.append((p["Points"] + p["BonusPoints"], p.get("Variance", 0), False))
            if e["aura"]:
                rows.extend(ability_rows(e["aura"], ops, seen))
    return rows


def pet_family(battle, side_word, name):
    owner = 0 if side_word == "your" else 1
    for p in battle["start"]["pets"][owner]:
        if p["name"] == name:
            return p["family"] - 1  # API families are 1-based
    return None


def check_line(line, battle, tol=1):
    m = DMG.search(line)
    kind = "dmg"
    if not m:
        m = HEAL.search(line)
        kind = "heal"
        if not m:
            return None
    aid, _maxhp, power, _speed = (int(x) for x in m.groups()[:4])
    amount = int(m.group(6))
    if amount == 0:
        return None
    gd = game_data()
    ab = gd.abilities.get(aid)
    if ab is None:
        return ("unknown_ability", aid, line)
    rows = ability_rows(aid, DMG_OPS if kind == "dmg" else HEAL_OPS)
    if not rows:
        return ("no_rows", aid, line)
    fam_mult = 1.0
    tag = m.group(9) if kind == "dmg" else None
    crit = bool(tag and ("Critical" in tag or "крит" in tag))
    if kind == "dmg":
        tfam = pet_family(battle, m.group(7), m.group(8))
        if tfam is not None:
            fam_mult = family_mult(ab["family"], tfam)
    bonus = 1 + power / 20
    ranges = []
    for pts, var, periodic in rows:
        base = pts * bonus * fam_mult
        lo, hi = base * (1 - var / 200), base * (1 + var / 200)
        if crit:
            lo, hi = int(lo) * C.CRIT_MULT, int(hi) * C.CRIT_MULT
        ranges.append((int(lo), int(hi) + 1, pts, var))
    ok = any(lo - tol <= amount <= hi + tol for lo, hi, *_ in ranges)
    return ("ok" if ok else "bad", aid, line, amount, ranges, crit, fam_mult)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--show-bad", action="store_true")
    args = ap.parse_args()
    files = [f for pat in args.files for f in glob.glob(pat)]
    stats = Counter()
    bad_by_ability = Counter()
    gd = game_data()
    for f in files:
        db = load_saved_variables(f)["BPLoggerDB"]
        for b in db.get("battles", []):
            for r in b["rounds"]:
                for line in r.get("log", []):
                    res = check_line(line, b)
                    if res is None:
                        continue
                    stats[res[0]] += 1
                    if res[0] == "bad":
                        bad_by_ability[gd.abilities[res[1]]["name"]] += 1
                        if args.show_bad:
                            clean = re.sub(r"\|T[^|]*\|t|\|c\w{8}|\|r|\|H[^|]*\|h|\|h", "", res[2])
                            print(f"BAD {clean}  got {res[3]} expected {[(lo, hi) for lo, hi, *_ in res[4]]} crit={res[5]} x{res[6]:.2f}")
    print(dict(stats))
    if bad_by_ability:
        print("bad by ability:", bad_by_ability.most_common(20))


if __name__ == "__main__":
    main()
