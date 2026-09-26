"""Replay recorded battles (BPLogger) in the simulator and compare every number.

For each recorded round we:
  * resync health from the previous snapshot (so one mismatch does not snowball),
  * feed the player's recorded action and the enemy action inferred from the combat log,
  * run the round with hits forced / no crits / no variance,
  * match every logged damage/heal line to a simulator line (same ability + target) and check that
    the logged amount fits the simulator value within the ability's variance (x1.5 when crit).

Usage:  python -m bp.validate.replay "data/logs/*.lua" [-v]
"""
from __future__ import annotations

import argparse
import glob
import re
from collections import Counter, defaultdict

from ..data.lua import load_saved_variables
from ..sim import constants as C
from ..sim.battle import Battle
from ..sim.data import PetSpec, game_data

LINK = re.compile(r"\|HbattlePetAbil:(\d+):(\d+):(\d+):(\d+)\|h\[([^\]]*)\]\|h")
CLEAN = re.compile(r"\|T[^|]*\|t|\|c\w{8}|\|r|\|H[^|]*\|h|\|h")
DMG = re.compile(r"^\[([^\]]+)\] dealt (\d+) damage to (your|enemy) (.+?)(?: \((\w+)\))?\.$")
HEAL = re.compile(r"^\[([^\]]+)\] healed (\d+) damage from (your|enemy) (.+?)\.$")
SIM_DMG = re.compile(r"^\[\d+\] (.+?) dealt (\d+) damage to (.+?)( \(Critical\))?$")
SIM_HEAL = re.compile(r"^\[\d+\] (.+?) healed (\d+) damage from (.+)$")


SIDE = ("your", "enemy")


def clean(line):
    return CLEAN.sub("", line)


def build_battle(rec):
    start = rec["start"]
    teams = []
    for owner in (0, 1):
        specs = []
        for p in start["pets"][owner]:
            # GetSpeed() already includes speed modifiers (Flying passive +50%): recover the base value
            pct = (p.get("st") or {}).get(C.S_SPEED_PCT, 0)
            speed = round(p["speed"] * 100 / (100 + pct)) if pct else p["speed"]
            specs.append(PetSpec(species=p["species"], abilities=[a or None for a in p["abilities"]],
                                 level=p["level"], quality=p["quality"], breed=3,
                                 stats=(p["maxhp"], p["power"], speed)))
        teams.append(specs)
    b = Battle(teams[0], teams[1], seed=0, log=True)
    for owner in (0, 1):
        for pet, p in zip(b.teams[owner].pets, start["pets"][owner]):
            pet.name = f"{SIDE[owner]} {p['name']}"  # side prefix: both teams may have the same pet
            pet.hp = p["hp"]
        b.teams[owner].active = (start["active"][owner] or 1) - 1
    b.force_hit, b.force_crit, b.force_variance = True, False, 0.0
    return b


def _self_aura(aid, aura_id):
    ab = game_data().abilities.get(aid)
    return bool(ab) and any(e["op"] in (26, 28, 63) and e["aura"] == aura_id for t in ab["turns"] for e in t["effects"])


def line_caster_side(raw):
    """0 = our pet cast it, 1 = enemy, None = unknown; from the line's wording."""
    c = clean(raw)
    links = LINK.findall(raw)
    m = re.search(r" (?:dealt \d+ damage to|applied \[[^\]]+\] to|healed \d+ damage from) (your|enemy) ", c)
    if not m or not links:
        return None
    target_side = 0 if m.group(1) == "your" else 1
    aid = int(links[0][0])
    if "applied" in c and len(links) > 1 and _self_aura(aid, int(links[1][0])):
        return target_side  # self-buff: the caster is the target
    if "healed" in c:
        return target_side
    return 1 - target_side


def enemy_swap_anywhere(b, lines):
    t = b.teams[1]
    for raw in lines:
        m = re.match(r"^(.+) is now enemy active pet\.$", clean(raw))
        if m:
            for i, p in enumerate(t.pets):
                if p.name == "enemy " + m.group(1) and i != t.active:
                    return 3 + i
    return None


def enemy_action(b, lines, our_aid=None):
    """Infer the enemy's action from the round's log lines."""
    t = b.teams[1]
    body = [l for l in lines if not l.startswith("Round")]
    for raw in body[:1]:  # a voluntary swap happens first; later swap lines are forced (Death Grip, deaths)
        c = clean(raw)
        m = re.match(r"^(.+) is now enemy active pet\.$", c)
        if m:
            for i, p in enumerate(t.pets):
                if p.name == "enemy " + m.group(1) and i != t.active:
                    return 3 + i
    pet = t.active_pet
    ours = b.teams[0].active_pet
    slots = game_data().species[pet.spec.species]["abilities"]
    for raw in lines:
        m = LINK.search(raw)
        if not m:
            continue
        aid, hp, power, speed = (int(x) for x in m.groups()[:4])
        # who cast it? the link carries the caster's current health, power and speed
        c = clean(raw)
        if pet.dead:
            continue
        same_stats = (pet.power, pet.speed) == (ours.power, ours.speed)
        if same_stats:
            # identical stat lines on both sides: go by the wording
            if line_caster_side(raw) != 1:
                continue
        else:
            def score(p):
                return (p.power == power) + (p.speed == speed) - abs(p.hp - hp) / max(1, p.max_hp)
            if score(pet) < score(ours):
                continue
        if aid not in pet.abilities and any(aid in o for o in slots):
            # PvP hides enemy abilities until used: learn them from the log
            for i, opts in enumerate(slots):
                if aid in opts:
                    pet.abilities[i] = aid
        if aid in pet.abilities:
            return pet.abilities.index(aid)
    return C.A_PASS


def our_action(b, rnd):
    act = rnd.get("action")
    if act:
        kind = act[0]
        if kind == "ability":
            return act[1] - 1
        if kind == "swap":
            return 3 + act[1] - 1
        if kind == "pass":
            return C.A_PASS
        return None
    for raw in rnd.get("log", []):
        m = re.match(r"^(.+) is now your active pet\.$", clean(raw))
        if m:
            for i, p in enumerate(b.teams[0].pets):
                if p.name == "your " + m.group(1):
                    return 3 + i
    return None


# the UI shows the Beast (<50% hp) and Flying (>50% hp) passives as auras; the simulator computes those
PSEUDO_AURAS = {C.PASSIVE_ABILITY[C.BEAST], C.PASSIVE_ABILITY[C.FLYING]}


_DUR = {}


def _applied_duration(aid):
    if not _DUR:
        for a in game_data().abilities.values():
            for t in a["turns"]:
                for e in t["effects"]:
                    if e["aura"] and "Duration" in e["params"]:
                        _DUR.setdefault(e["aura"], e["params"]["Duration"])
    return _DUR.get(aid)


def _auras(b, holder, snap_auras):
    from ..sim.battle import Aura, Pet
    out = []
    gd = game_data()
    for x in snap_auras or []:
        aid = x["id"]
        if aid in PSEUDO_AURAS or aid not in gd.abilities:
            continue
        caster = None
        if x.get("co") in (1, 2) and x.get("ci"):
            team = b.teams[x["co"] - 1]
            if x["ci"] - 1 < len(team.pets):
                caster = team.pets[x["ci"] - 1]
        a = Aura(aid=aid, ability=gd.abilities[aid], holder=holder if isinstance(holder, Pet) else None,
                 caster=caster, duration=x["turns"] if x["turns"] is not None else -1, uid=b.next_uid())
        a.fresh = False
        # ticks already done = original duration - turns remaining + 1 (the application round ticks too)
        d0 = _applied_duration(aid)
        a.age = max(1, d0 - a.duration + 1) if d0 and a.duration >= 0 else 1
        a.blocks_left = gd.abilities[aid]["states"].get(C.S_BLOCK_COUNT, 0)
        old = next((x for x in (holder.auras if holder is not None else []) if x.aid == aid), None)
        if old is not None:
            a.blocks_left = old.blocks_left  # the snapshot does not show how many blocks are left
        out.append(a)
    return out


def resync(b, snap):
    """Copy health, auras, weather and cooldowns from a BPLogger snapshot into the simulator."""
    if not snap:
        return
    gd = game_data()
    for owner in (0, 1):
        for pet, p in zip(b.teams[owner].pets, snap["pets"][owner]):
            if p.get("hp") is not None:
                pet.hp = p["hp"]
                pet.dead = p["hp"] <= 0
            if "auras" in p:
                pet.auras = _auras(b, pet, p["auras"])
                pet.mech_revived = pet.mech_revived or any(x["id"] == C.PASSIVE_ABILITY[C.MECHANICAL] for x in p["auras"])
            for i, cd in enumerate(p.get("cd") or []):
                if i < 3 and isinstance(cd, list) and len(cd) >= 2:
                    pet.cooldowns[i] = max(cd[1] or 0, cd[2] or 0)
        if snap["active"][owner]:
            b.teams[owner].active = snap["active"][owner] - 1
        pad = snap.get(f"pad{owner + 1}")
        if pad is not None:
            b.teams[owner].auras = _auras(b, b.teams[owner], pad)
    w = _auras(b, None, snap.get("weather"))
    b.weather = w[0] if w else None


MISS = re.compile(r"^\[([^\]]+)\] (?:missed|was dodged by) (your|enemy) (.+?)\.$")


BLOCKED = re.compile(r"^\[([^\]]+)\] was blocked from striking (your|enemy) (.+?)\.$")


def force_from_log(b, lines):
    """Reproduce the logged misses and crits, and who moved first on speed ties."""
    outcomes, crit, amounts, heals = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for raw in lines:
        c = clean(raw)
        m = MISS.match(c)
        if m:
            outcomes[(m.group(1), f"{m.group(2)} {m.group(3)}")].append(False)
            continue
        m = DMG.match(c)
        if m:
            key = (m.group(1), f"{m.group(3)} {m.group(4)}")
            outcomes[key].append(True)
            amounts[key].append(int(m.group(2)))
            crit[key].append(m.group(5) == "Critical")  # per hit, in order
            continue
        m = HEAL.match(c)
        if m:
            heals[(m.group(1), f"{m.group(3)} {m.group(4)}")].append(int(m.group(2)))
            continue
        m = BLOCKED.match(c)
        if m:
            outcomes[(m.group(1), f"{m.group(2)} {m.group(3)}")].append(True)
    b.forced_miss, b.forced_crit, b.forced_amount, b.forced_heal = outcomes, crit, amounts, heals
    b.force_first = None
    ours, theirs = b.teams[0].active_pet, b.teams[1].active_pet
    for raw in lines:
        m = LINK.search(raw)
        if not m:
            continue
        aid = int(m.group(1))
        c = clean(raw)
        if aid in ours.abilities and aid not in theirs.abilities:
            b.force_first = 0
        elif aid in theirs.abilities and aid not in ours.abilities:
            b.force_first = 1
        elif aid in ours.abilities:  # both sides have it: the wording tells who cast it
            b.force_first = line_caster_side(raw)
        if b.force_first is not None:
            break


def ability_variance(name):
    gd = game_data()
    v = 0
    for a in gd.abilities.values():  # several abilities/auras can share a name
        if a["name"] == name:
            for t in a["turns"]:
                for e in t["effects"]:
                    v = max(v, e["params"].get("Variance", 0))
    return v


def compare(log_lines, sim_lines):
    """Match logged number lines to simulator lines. Returns list of (status, text)."""
    sim = defaultdict(list)
    for l in sim_lines:
        m = SIM_DMG.match(l) or SIM_HEAL.match(l)
        if m:
            is_crit = "dealt" in l and bool(m.group(4))
            sim[(m.group(1), m.group(3), "dmg" if "dealt" in l else "heal")].append((int(m.group(2)), is_crit))
    out = []
    for raw in log_lines:
        c = clean(raw)
        m = DMG.match(c)
        kind = "dmg"
        if not m:
            m = HEAL.match(c)
            kind = "heal"
        if not m:
            continue
        name, amount, target = m.group(1), int(m.group(2)), f"{m.group(3)} {m.group(4)}"
        tag = m.group(5) if kind == "dmg" else None
        if amount == 0:
            continue
        cands = sim.get((name, target, kind), [])
        if not cands:
            out.append(("missing", f"{c}  (sim: no such line)"))
            continue
        var = ability_variance(name)
        crit = tag == "Critical"
        best = None
        for cand in cands:
            v, sim_crit = cand
            lo, hi = v * (1 - var / 200), v * (1 + var / 200)
            if crit and not sim_crit:
                lo, hi = int(lo) * C.CRIT_MULT, int(hi) * C.CRIT_MULT
            tol = 1.01 + (0.005 * amount if crit else 0)  # crit rounding at the edge of the roll range
            if lo - tol <= amount <= hi + tol:
                best = cand
                break
        if best is not None:
            cands.remove(best)
            out.append(("ok", c))
        else:
            out.append(("bad", f"{c}  (sim: {cands}{' x1.5' if crit else ''} var {var})"))
    return out


def replay(rec, verbose=False):
    b = build_battle(rec)
    stats = Counter()
    report = []
    prev = rec.get("start")
    for rnd in rec["rounds"]:
        lines = rnd.get("log", [])
        if not any(l.startswith("Round") for l in lines):
            # intro or forced replacement between rounds (both sides may replace at once)
            a = our_action(b, rnd)
            e = enemy_swap_anywhere(b, lines)
            need_a, need_e = b.needs_replacement(0), b.needs_replacement(1)
            if (need_a and a is not None) or (need_e and e is not None):
                b.step(a if need_a and a is not None else C.A_PASS, e if need_e and e is not None else C.A_PASS)
            prev = rnd.get("after") or prev
            continue
        resync(b, prev)
        if (rnd.get("action") or [None])[0] == "forfeit":
            break
        a = our_action(b, rnd)
        if a is None:
            a = C.A_PASS  # no button pressed: forced continuation (multi-round ability) or stunned
        our_aid = b.teams[0].active_pet.abilities[a] if a in C.A_ABILITY else None
        e = enemy_action(b, lines, our_aid)
        if b.needs_replacement(1):
            e = enemy_swap_anywhere(b, lines) or e
        force_from_log(b, lines)
        b.log.clear()
        legal_a, legal_e = b.legal_actions(0), b.legal_actions(1)
        if a not in legal_a:
            a = C.A_PASS
        if e not in legal_e:
            e = C.A_PASS
        try:
            b.step(a, e)
        except Exception as ex:  # keep going, report
            report.append(("error", f"round {rnd.get('n')}: {ex!r}"))
            stats["error"] += 1
            break
        if b.over:
            # the game still finishes the round after the deciding death (thorns, pad objects);
            # it cannot change the result, so ignore those lines
            loser = {0: "Enemy ", 1: "Your "}.get(b.winner)
            deaths = [i for i, l in enumerate(lines) if clean(l).endswith(" died.")
                      and (loser is None or clean(l).startswith(loser))]
            if deaths:
                lines = lines[:deaths[-1] + 1]
        res = compare(lines, b.log)
        for st, txt in res:
            stats[st] += 1
            if verbose or st != "ok":
                report.append((st, f"R{b.round}: {txt}"))
        prev = rnd.get("after") or prev
        if b.over:
            break
    return stats, report



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    total = Counter()
    for f in [f for pat in args.files for f in glob.glob(pat)]:
        db = load_saved_variables(f)["BPLoggerDB"]
        for i, rec in enumerate(db.get("battles", [])):
            if not rec.get("start"):
                continue
            stats, report = replay(rec, args.verbose)
            total.update(stats)
            print(f"== {f} battle {i + 1}: {rec.get('target')} {dict(stats)}")
            for st, txt in report:
                print(f"  [{st}] {txt}")
    print("TOTAL", dict(total))


if __name__ == "__main__":
    main()
