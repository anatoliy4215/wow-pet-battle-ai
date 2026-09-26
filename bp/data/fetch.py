"""Download raw pet battle data into data/raw/.

Sources:
  * wago.tools  - game DB2 tables (the ground truth for mechanics)
  * wow-petsim  - NPC tamer teams and AI scripts, names, hand-written ability rules
  * BattlePetBreedID addon - breeds available per species

Usage:  python -m bp.data.fetch [--build 12.1.0.69933] [--force]
"""
import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

DB2_TABLES = [
    "BattlePetAbility",
    "BattlePetAbilityEffect",
    "BattlePetAbilityState",
    "BattlePetAbilityTurn",
    "BattlePetBreedQuality",
    "BattlePetBreedState",
    "BattlePetEffectProperties",
    "BattlePetSpecies",
    "BattlePetSpeciesState",
    "BattlePetSpeciesXAbility",
    "BattlePetState",
    "BattlePetVisual",
]
LOCALES = ["enUS", "ruRU"]
LOCALIZED = {"BattlePetAbility", "BattlePetSpecies"}

PETSIM = {
    "petsim_pets.json": "https://api.wow-petsim.com/get/pets",
    "petsim_abilities.json": "https://api.wow-petsim.com/get/abilities",
    "petsim_other.json": "https://api.wow-petsim.com/get/other",
    "petsim_names_en.json": "https://api.wow-petsim.com/translation/en",
}
BREED_ID_LUA = "https://raw.githubusercontent.com/MMOSimca/BattlePetBreedID/HEAD/PetData.lua"

session = requests.Session()
session.headers["User-Agent"] = "bp-research/0.1 (personal pet battle simulator)"


def live_build():
    builds = session.get("https://wago.tools/api/builds", timeout=30).json()
    return builds["wow"][0]["version"]


def download(url, dest: Path, force=False):
    if dest.exists() and not force:
        return False
    r = session.get(url, timeout=120)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    time.sleep(0.5)  # be polite
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    build = args.build or live_build()
    db2_dir = RAW / "db2" / build
    print("build", build)
    for t in DB2_TABLES:
        for loc in (LOCALES if t in LOCALIZED else ["enUS"]):
            suffix = "" if loc == "enUS" else f"_{loc}"
            url = f"https://wago.tools/db2/{t}/csv?build={build}&locale={loc}"
            if download(url, db2_dir / f"{t}{suffix}.csv", args.force):
                print("  got", t, loc)

    for name, url in PETSIM.items():
        try:
            if download(url, RAW / "petsim" / name, args.force):
                print("  got", name)
        except requests.HTTPError as e:
            print("  FAILED", name, e)

    if download(BREED_ID_LUA, RAW / "breedid" / "PetData.lua", args.force):
        print("  got PetData.lua")

    (RAW / "meta.json").write_text(json.dumps({"build": build}, indent=1))


if __name__ == "__main__":
    main()
