"""Convert the BPLogger pet-journal export into data/processed/collection.json.

Breed is inferred by matching the exported stats against every breed of the species.

Usage:  python -m bp.data.collection [path to BPLogger.lua]
"""
import json
import sys
from pathlib import Path

from ..sim.data import PetSpec, compute_stats, game_data
from .lua import load_saved_variables

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = Path(r"D:\World of Warcraft\_retail_\WTF\Account")


def find_saved():
    files = sorted(DEFAULT.glob("*/SavedVariables/BPLogger.lua"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def infer_breed(species, level, quality, stats):
    sp = game_data().species.get(species)
    if not sp:
        return None
    best, err = None, 1e9
    for b in sp["breeds"] or range(3, 13):
        s = compute_stats(PetSpec(species, [0, 0, 0], level, quality, b))
        e = sum(abs(x - y) for x, y in zip(s, stats))
        if e < err:
            best, err = b, e
    return best if err <= 3 else None


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_saved()
    db = load_saved_variables(path)["BPLoggerDB"]
    col = db.get("collection", {})
    out = []
    for p in col.get("pets", []):
        q = p["quality"] - 1  # journal rarity is 1-based
        breed = infer_breed(p["species"], p["level"], q, (p["maxhp"], p["power"], p["speed"]))
        out.append({"species": p["species"], "name": p["name"], "level": p["level"], "quality": q,
                    "breed": breed, "stats": [p["maxhp"], p["power"], p["speed"]], "family": p["family"] - 1,
                    "battle": bool(p.get("battle", 1))})
    dest = ROOT / "data" / "processed" / "collection.json"
    dest.write_text(json.dumps({"date": col.get("date"), "pets": out}, ensure_ascii=False, indent=0), encoding="utf-8")
    n25 = [p for p in out if p["level"] == 25 and p["battle"]]
    print(f"{len(out)} pets, {len(n25)} battle-ready level 25, breed unknown for {sum(p['breed'] is None for p in out)}")
    fam = {}
    for p in n25:
        fam[p["family"]] = fam.get(p["family"], 0) + 1
    names = ["Humanoid", "Dragonkin", "Flying", "Undead", "Critter", "Magic", "Elemental", "Beast", "Aquatic", "Mechanical"]
    print("level 25 by family:", {names[k]: v for k, v in sorted(fam.items())})


if __name__ == "__main__":
    main()
