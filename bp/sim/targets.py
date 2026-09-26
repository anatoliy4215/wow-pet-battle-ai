"""Effect target resolution.

DB2 does not store who an effect hits. Team-wide effects are N copies of the same row, but identical
rows can also be a multi-hit on one pet (docs/opcodes.md §0.1). We derive a per-row target from
wow-petsim's hand-written rules (aligned by effect kind and order) and fall back to the visual
RangeTypeEnum (2 = team).

Target codes:  E0/E1/E2 enemy active/backup1/backup2,  A0/A1/A2 own active(self)/backups,
               EP/AP enemy/own team pad,  None = opcode default.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KIND = {}
TEAM_TEXT = re.compile(r"enemy's pets|all enemy pets|each enemy pet|all enemies", re.I)
for _op in (24, 222, 29, 62, 68, 363, 141, 149, 197, 59, 66, 45, 97, 384, 385, 96, 169, 170, 226, 65, 27, 358, 86, 370):
    KIND[_op] = "damage"
for _op in (23, 53, 32, 44, 104, 61, 171, 100, 254):
    KIND[_op] = "heal"
for _op in (26, 28, 50, 52, 54, 63, 131, 178, 229, 230, 168, 172, 248, 137, 223, 150, 76, 77, 75):
    KIND[_op] = "buff"
for _op in (145, 139, 107):
    KIND[_op] = "swap"
KIND[135] = "kill"
for _op in (49, 85, 136):
    KIND[_op] = "removeBuff"

TRIGGER_EVENT = {"startTurn": 6, "endTurn": 7, "postAuraTick": 12, "onAbility": 9, "onSwapIn": 10,
                 "onSwapOut": 11, "damageReceive": 1, "directDamageReceive": 1, "damageDeal": 2,
                 "applied": 0, "removed": 5, "expired": 5}

PS_TARGET = {"activeEnemy": ["E0"], "backupEnemy1": ["E1"], "backupEnemy2": ["E2"], "backupEnemy": ["E1", "E2"],
             "petsEnemy": ["E0", "E1", "E2"], "teamEnemy": ["EP"], "source": ["A0"], "activeAlly": ["A0"],
             "backupAlly1": ["A1"], "backupAlly2": ["A2"], "backupAlly": ["A1", "A2"],
             "petsAlly": ["A0", "A1", "A2"], "teamAlly": ["AP"]}


@lru_cache(maxsize=1)
def _petsim():
    with open(ROOT / "data" / "raw" / "petsim" / "petsim_abilities.json", encoding="utf-8") as f:
        abil = {a["id"]: json.loads(a["effect"]) for a in json.load(f) if a["effect"]}
    with open(ROOT / "data" / "raw" / "petsim" / "petsim_other.json", encoding="utf-8") as f:
        buffs = {}
        for b in json.load(f)["buff"]:
            if b.get("abilityId") and b.get("effect"):
                buffs.setdefault(b["abilityId"], json.loads(b["effect"]))
    return abil, buffs


def _expand(ps_effects, event, kind, is_aura=False):
    """Target list for petsim effects of `kind` that belong to trigger `event` (None = main turn)."""
    out = []
    last = None
    for e in ps_effects:
        trig = [TRIGGER_EVENT.get(t) for t in (e.get("triggers") or []) if isinstance(t, str)]
        ev = trig[0] if trig else None
        tl = [t for t in e.get("targets", []) if isinstance(t, str)]
        if tl and tl[0] != "previous":
            last = tl
        if ev != event or e.get("effect") != kind:
            continue
        rep = (e.get("parameters") or {}).get("repeat", 1)
        rep = rep if isinstance(rep, int) else 1
        cur = []
        for t in tl:
            if t == "previous":
                cur.extend(PS_TARGET.get(last[0], [None]) if last else [None])
            elif t == "source" and is_aura:
                cur.append("C")  # in an aura rule "source" is whoever applied it (Siphon Life heals its caster)
            elif t == "triggerSourcePet":
                # who set the trigger off: the attacker for 'when damaged' (Touch of the Animus heals the
                # striker, log 2026-09-25), the pet itself for 'when dealing damage' (Leech Life)
                cur.append("T" if ev == 1 else "A0")
            else:
                cur.extend(PS_TARGET.get(t, [None]))
        out.extend(cur * rep if cur else [None] * rep)
    return out


# hand fixes where the petsim rules are missing or wrong: {ability: {(turn index, effect index): code}}
OVERRIDES = {
    1912: {(0, 0): "A0"},  # Void Portal hurts its user (log 2026-09-25: 357 to our Happiness)
    540: {(1, 1): "E0"},  # Chew (self buff) bites the enemy when it ends, not its holder (log 2026-09-25)
    658: {(0, 0): "T"},   # Plagued Blood: the pet that strikes the plagued target is healed (log 2026-09-25)
}


@lru_cache(maxsize=None)
def plan(aid, ability_json_key):
    """{(turn_index, effect_index): target code} for one ability/aura."""
    from .data import game_data
    ab = game_data().abilities[aid]
    ps_abil, ps_buff = _petsim()
    ps = ps_abil.get(aid) or ps_buff.get(aid)
    is_aura = aid in ps_buff and aid not in ps_abil
    res = {}
    for ti, t in enumerate(ab["turns"]):
        event = t["event"] if t["type"] == 1 else None
        if is_aura and event is None:
            event = 7  # aura ticks are "endTurn" in petsim
        by_kind = {}
        for ei, e in enumerate(t["effects"]):
            k = KIND.get(e["op"])
            if k:
                by_kind.setdefault(k, []).append(ei)
        for kind, idxs in by_kind.items():
            targets = _expand(ps, event, kind, is_aura) if ps else []
            if targets and len(targets) >= 1:
                for j, ei in enumerate(idxs):
                    res[(ti, ei)] = targets[j] if j < len(targets) else targets[-1]
            elif len(idxs) > 1:
                # no rule: consecutive identical copies with a team visual (or a tooltip that says the whole
                # enemy team, log: Tidal Wave hits all three pets) -> spread over the team
                rows = [t["effects"][i] for i in idxs]
                team_text = kind == "damage" and TEAM_TEXT.search(ab.get("description") or "")
                if (all(r["range"] == 2 for r in rows) or team_text) and len({(r["op"], r["aura"]) for r in rows}) == 1:
                    side = "A" if kind in ("heal",) or rows[0]["op"] in (26, 28, 63) else "E"
                    for j, ei in enumerate(idxs[:3]):
                        res[(ti, ei)] = f"{side}{j}"
    # 'None' entries mean default; normalise "A0" on self-default ops to None
    res.update(OVERRIDES.get(aid, {}))
    return {k: v for k, v in res.items() if v is not None}


AURA_OPS = (50, 52, 54, 168, 172, 178)


@lru_cache(maxsize=1)
def chain_rows():
    """(ability, turn, effect) of debuffs that DB2 marks independent (ChainFailure 0) but that fail when the
    attack right before them misses - petsim marks them chainFailure (log 2026-09-25: a missed Jar of
    Smelly Liquid applied no Weakness)."""
    from .data import game_data
    ps_abil, _ = _petsim()
    out = set()
    for aid, a in game_data().abilities.items():
        ps = ps_abil.get(aid) or []
        if not any(e.get("effect") == "buff" and (e.get("parameters") or {}).get("chainFailure") for e in ps):
            continue
        for ti, t in enumerate(a["turns"]):
            if t["type"] != 0:
                continue
            effs = t["effects"]
            for ei in range(1, len(effs)):
                if effs[ei]["op"] in AURA_OPS and not effs[ei]["params"].get("ChainFailure")                         and KIND.get(effs[ei - 1]["op"]) == "damage":
                    out.add((aid, ti, ei))
    return out
