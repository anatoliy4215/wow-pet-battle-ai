"""Build data/processed/achievements.json - pet battle achievements the user still needs.

Source: wago.tools DB2 CSV exports (Achievement enUS/ruRU, CriteriaTree, Criteria, ModifierTree,
BattlePetSpecies) saved in data/raw/db2/<build>/.  Missing tables are downloaded.

How pet battle achievements are encoded:
  Criteria.Type 158 = win a pet battle, Type 8 = complete achievement (Asset = achievement id),
  Type 161 = capture a battle pet.  Win-conditions live in the criteria's ModifierTree:
    type 81  asset = opponent creature id (NPC tamer)
    type 91  asset = opponent battle pet species id (legendary/elite wild pets)
    type 152 asset 3, secondary = pet family (0-based) required for the whole team
    type 34  asset = minimum level of all your pets (25)
    type 151 asset 1, secondary = species that must be on your team (1426 = Elekk Plushie)
    type 90  PvP battle, type 41 area id
  CriteriaTree.Operator 4 = all children, 8 = at least Amount children.

Usage:  python -m bp.data.achievements [--build 12.1.0.69933]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "achievements.json"
DEFAULT_BUILD = "12.1.0.69933"

FAMILIES = ["Humanoid", "Dragonkin", "Flying", "Undead", "Critter",
            "Magic", "Elemental", "Beast", "Aquatic", "Mechanical"]

# Metas whose children (all family achievements) are all wanted.
FAMILY_METAS = [
    16512,  # Family Battler of the Dragon Isles
    40980,  # Family Battler of Khaz Algar
    41551,  # Family Battler of Undermine
    60956,  # Family Battler of Northrend
    61040,  # Family Battler of Eastern Kingdoms
    61051,  # Family Battler of Kalimdor
    62460,  # Family Battler of Outland
    62461,  # Family Battler of Cataclysm
]
# Achievements explicitly listed by the user as not completed (armory, ruRU names).
TARGETS = [
    6592, 6597, 6620, 6621, 6622, 7499, 8300, 9069,
    9686, 9687, 9688, 9690, 9691, 9692, 9693, 9694, 9695, 9696,  # Legion families (9689 done)
    12088, 12089, 12091, 12092, 12093, 12094, 12095, 12096, 12097, 12098, 12099, 12100,
    12281, 12282, 12283, 12284, 12285, 12286, 12287, 12289, 12290,  # PvP family brawlers
    13270, 13271, 13272, 13273, 13274, 13275, 13277, 13278, 13279, 13280, 13281,
    13625, 13626, 13766,
    14868, 14869, 14870, 14871, 14872, 14873, 14874, 14875, 14876, 14877, 14879, 14881,
    61094,  # Old World Family Battler
    6614,   # Outland Tamer (capture) - shares the ruRU name with 62460, included for safety
] + FAMILY_METAS

OTHER_NOTES = {
    6592: "Win 5000 pet battles (any: wild, NPC or PvP).",
    6597: "Win 250 PvP pet battles.",
    6620: "Win 5 consecutive PvP pet battles without any of your pets dying.",
    6621: "Win any pet battle (wild pets count) in Orgrimmar, Thunder Bluff, Undercity and Silvermoon.",
    8300: "Win 250 PvP pet battles via Find Battle with a full team of level 25 pets.",
    6614: "Capture a battle pet in each Outland zone (collection, not tamers). "
          "Same ruRU name as 62460 Family Battler of Outland - probably not the one meant.",
}

session = requests.Session()
session.headers["User-Agent"] = "bp-research/0.1 (personal pet battle simulator)"


def table(db2: Path, name: str, locale="enUS"):
    suffix = "" if locale == "enUS" else f"_{locale}"
    path = db2 / f"{name}{suffix}.csv"
    if not path.exists():
        url = f"https://wago.tools/db2/{name}/csv?build={db2.name}&locale={locale}"
        r = session.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["ID"]): r for r in csv.DictReader(f)}


class DB:
    def __init__(self, db2: Path):
        self.ach = table(db2, "Achievement")
        self.ach_ru = table(db2, "Achievement", "ruRU")
        self.ct = table(db2, "CriteriaTree")
        self.cr = table(db2, "Criteria")
        self.mt = table(db2, "ModifierTree")
        self.species = table(db2, "BattlePetSpecies")
        self.ct_kids = defaultdict(list)
        for r in self.ct.values():
            self.ct_kids[int(r["Parent"])].append(r)
        for v in self.ct_kids.values():
            v.sort(key=lambda r: int(r["OrderIndex"]))
        self.mt_kids = defaultdict(list)
        for r in self.mt.values():
            self.mt_kids[int(r["Parent"])].append(r)

    def modifiers(self, mid):
        """Flatten a modifier tree into a list of (type, asset, secondary)."""
        out = []
        stack = [mid]
        while stack:
            m = stack.pop()
            r = self.mt.get(m)
            if r is None:
                continue
            if int(r["Type"]):
                out.append((int(r["Type"]), int(r["Asset"]), int(r["SecondaryAsset"])))
            stack.extend(int(k["ID"]) for k in self.mt_kids[m])
        return out

    def leaves(self, tree_id):
        """Criteria leaves of a criteria tree: list of (tree_row, criteria_row)."""
        out = []
        r = self.ct[tree_id]
        cid = int(r["CriteriaID"])
        if cid:
            out.append((r, self.cr[cid]))
        for k in self.ct_kids[tree_id]:
            out.extend(self.leaves(int(k["ID"])))
        return out


def creature_names(db: DB):
    """creature id -> name, from criteria tree descriptions of single-target defeat criteria."""
    names = {}
    for r in db.ct.values():
        cid = int(r["CriteriaID"])
        desc = r["Description_lang"].strip()
        if not cid or not desc or cid not in db.cr:
            continue
        c = db.cr[cid]
        if c["Type"] != "158":
            continue
        creatures = [a for t, a, _ in db.modifiers(int(c["Modifier_tree_ID"])) if t == 81]
        if len(creatures) == 1:
            names.setdefault(creatures[0], desc)
    return names


def build(build_id=DEFAULT_BUILD):
    db2 = RAW / "db2" / build_id
    db = DB(db2)
    names = creature_names(db)
    tamers = json.loads((ROOT / "data" / "processed" / "tamers.json").read_text(encoding="utf-8"))
    by_npc = {v["npc"]: k for k, v in tamers.items()}
    by_species = {}  # single-pet (wild/legendary) tamers keyed by their pet species
    for k, v in tamers.items():
        if len(v["pets"]) == 1:
            by_species.setdefault(v["pets"][0]["species"], k)

    ids = list(dict.fromkeys(TARGETS))
    for meta in FAMILY_METAS:
        for _, c in db.leaves(int(db.ach[meta]["Criteria_tree"])):
            if c["Type"] == "8" and int(c["Asset"]) not in ids:
                ids.append(int(c["Asset"]))

    result = []
    for aid in ids:
        a = db.ach[aid]
        root = db.ct[int(a["Criteria_tree"])]
        leaves = db.leaves(int(a["Criteria_tree"]))
        entry = {
            "achievement_id": aid,
            "name_en": a["Title_lang"],
            "name_ru": db.ach_ru[aid]["Title_lang"],
            "description_en": a["Description_lang"],
            "kind": "other",
            "required_family": None,
            "tamers": [],
            "notes": "",
        }
        types = {c["Type"] for _, c in leaves}
        if types == {"8"}:
            entry["kind"] = "meta"
            entry["children"] = [int(c["Asset"]) for _, c in leaves]
            if aid == 12290:
                entry["notes"] = "Children are PvP Find Battle family achievements (not NPC tamers)."
        elif types == {"158"}:
            tamer_list, families, levels, team_species, other_mods = [], [], set(), set(), set()
            pvp = False
            for tr, c in leaves:
                mods = db.modifiers(int(c["Modifier_tree_ID"]))
                creatures = [x for t, x, _ in mods if t == 81]
                species = [x for t, x, _ in mods if t == 91]
                fam = None
                for t, x, y in mods:
                    if t == 152:
                        fam = y
                    elif t == 34:
                        levels.add(x)
                    elif t == 151:
                        team_species.add(y)
                    elif t == 90:
                        pvp = True
                    elif t not in (81, 91):
                        other_mods.add((t, x, y))
                if not creatures and not species:
                    continue
                t = {"npc_id": None, "name": tr["Description_lang"].strip()}
                if creatures:
                    t["npc_id"] = creatures[0]
                    if len(creatures) > 1:
                        t["alt_npc_ids"] = creatures[1:]
                    t["name"] = t["name"] or names.get(creatures[0]) or \
                        (tamers[by_npc[creatures[0]]]["name"] if creatures[0] in by_npc else "")
                    tid = by_npc.get(creatures[0])
                else:
                    sp = species[0]
                    t["species_id"] = sp
                    npc = int(db.species[sp]["CreatureID"]) if sp in db.species else 0
                    t["npc_id"] = npc or None
                    tid = by_npc.get(npc) or by_species.get(sp)
                t["tamer_id"] = tid
                if fam is not None:
                    families.append(fam)
                    t["_family"] = fam
                tamer_list.append(t)
            if pvp and not families:
                families = [y for _, c in leaves
                            for t, x, y in db.modifiers(int(c["Modifier_tree_ID"])) if t == 152]
            main_fam = max(set(families), key=families.count) if families else None
            for t in tamer_list:
                f = t.pop("_family", None)
                if f is not None and f != main_fam:
                    # Blizzard data quirk (e.g. 40162 reuses the Flying criterion for one trainer)
                    t["required_family_override"] = f
                    entry["notes"] += (f"DB2 quirk: {t['name']} criterion requires an all-"
                                       f"{FAMILIES[f]} team (shared criterion). ")
            entry["required_family"] = main_fam
            if pvp or not tamer_list:
                entry["notes"] = OTHER_NOTES.get(aid) or entry["description_en"]
            else:
                entry["kind"] = "family_tamer" if families else "tamer_list"
                entry["tamers"] = tamer_list
                need = int(root["Amount"]) if root["Operator"] == "8" else len(tamer_list)
                entry["required_count"] = need
                bits = []
                if need < len(tamer_list):
                    bits.append(f"Defeat any {need} of {len(tamer_list)} listed.")
                else:
                    bits.append(f"Defeat all {len(tamer_list)}.")
                if entry["required_family"] is not None:
                    bits.append(f"Whole team must be {FAMILIES[entry['required_family']]}.")
                if levels:
                    bits.append(f"All pets level {max(levels)}.")
                for sp in sorted(team_species):
                    bits.append(f"Team must include species {sp}.")
                if other_mods:
                    bits.append(f"Other modifiers: {sorted(other_mods)}.")
                missing = sum(1 for t in tamer_list if t["tamer_id"] is None)
                if missing:
                    bits.append(f"{missing} target(s) without team data in tamers.json.")
                entry["notes"] += " ".join(bits)
        else:
            entry["notes"] = OTHER_NOTES.get(aid) or entry["description_en"]
        result.append(entry)

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=DEFAULT_BUILD)
    args = ap.parse_args()
    res = build(args.build)
    for e in res:
        n = len(e["tamers"])
        have = sum(1 for t in e["tamers"] if t["tamer_id"] is not None)
        fam = FAMILIES[e["required_family"]] if e["required_family"] is not None else "-"
        extra = f" children={e['children']}" if e["kind"] == "meta" else f" {have}/{n} with team data"
        print(f"{e['achievement_id']:>6} {e['kind']:<12} {fam:<10} {e['name_en']}{extra}")
    print("wrote", OUT, len(res), "achievements")


if __name__ == "__main__":
    main()
