"""Normalize raw downloads into data/processed/*.json.

Output:
  species.json   {species_id: {name, name_ru, family, base:[h,p,s], abilities:[[a,b],[c,d],[e,f]],
                                breeds:[..], creature, obtainable}}
  abilities.json {ability_id: {name, name_ru, family, cooldown, flags, states:{state:value},
                               turns:[{order, type, event, effects:[{op, aura, order, params:{..}}]}]}}
  states.json    {state_id: lua_name}
  effects.json   {op: [param labels]}
  breeds.json    {breed_id: [h,p,s] breed points}, quality multipliers
  tamers.json    {tamer_id: {npc, name, zone, wild, pets:[{slot, species, creature, breed, quality,
                                                            level, abilities:[3 ids], behaviour}]}}

Families are 0-based as in the game tables: 0 Humanoid, 1 Dragonkin, 2 Flying, 3 Undead, 4 Critter,
5 Magic, 6 Elemental, 7 Beast, 8 Aquatic, 9 Mechanical.

Usage:  python -m bp.data.build
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# petsim encodes breeds as two letters, B=balanced, H/P/S=health/power/speed
PETSIM_BREED = {"bb": 3, "pp": 4, "ss": 5, "hh": 6, "hp": 7, "ps": 8, "hs": 9, "pb": 10, "sb": 11, "hb": 12}


def read_csv(name, build):
    with open(RAW / "db2" / build / f"{name}.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(*parts):
    with open(RAW.joinpath(*parts), encoding="utf-8") as f:
        return json.load(f)


def parse_breed_lua():
    text = (RAW / "breedid" / "PetData.lua").read_text(encoding="utf-8")
    breeds = {}
    for sid, body in re.findall(r"BreedsPerSpecies\[(\d+)\] = \{([^}]*)\}", text):
        breeds[int(sid)] = [int(x) for x in body.split(",") if x.strip()]
    return breeds


def build_abilities(build):
    names_ru = {r["ID"]: r["Name_lang"] for r in read_csv("BattlePetAbility_ruRU", build)}
    props = {r["ID"]: [r[f"ParamLabel_{i}"] for i in range(6)] for r in read_csv("BattlePetEffectProperties", build)}
    ab_states = defaultdict(dict)
    for r in read_csv("BattlePetAbilityState", build):
        ab_states[r["BattlePetAbilityID"]][int(r["BattlePetStateID"])] = int(r["Value"])
    visual_range = {r["ID"]: int(r["RangeTypeEnum"]) for r in read_csv("BattlePetVisual", build)}
    effects = defaultdict(list)
    for e in read_csv("BattlePetAbilityEffect", build):
        op = e["BattlePetEffectPropertiesID"]
        labels = props.get(op, [""] * 6)
        params = {}
        for i in range(6):
            v = int(e[f"Param_{i}"])
            if v == 0 and (not labels[i] or labels[i].lower() == "unused"):
                continue
            label = labels[i] or f"p{i}"
            params[label] = v
        effects[e["BattlePetAbilityTurnID"]].append(
            {"order": int(e["OrderIndex"]), "op": int(op), "aura": int(e["AuraBattlePetAbilityID"]), "params": params,
             "range": visual_range.get(e["BattlePetVisualID"], -1)}
        )
    turns = defaultdict(list)
    for t in read_csv("BattlePetAbilityTurn", build):
        turns[t["BattlePetAbilityID"]].append(
            {
                "order": int(t["OrderIndex"]),
                "type": int(t["TurnTypeEnum"]),
                "event": int(t["EventTypeEnum"]),
                "effects": sorted(effects[t["ID"]], key=lambda e: e["order"]),
            }
        )
    out = {}
    for r in read_csv("BattlePetAbility", build):
        aid = r["ID"]
        out[int(aid)] = {
            "name": r["Name_lang"],
            "name_ru": names_ru.get(aid, ""),
            "description": r["Description_lang"],
            "family": int(r["PetTypeEnum"]),
            "cooldown": int(r["Cooldown"]),
            "flags": int(r["Flags"]),
            "states": ab_states.get(aid, {}),
            "turns": sorted(turns[aid], key=lambda t: t["order"]),
        }
    effect_labels = {int(k): v for k, v in props.items()}
    return out, effect_labels


def build_species(build):
    petsim = {p["petId"]: p for p in read_json("petsim", "petsim_pets.json")["pets"]}
    names = read_json("petsim", "petsim_names_en.json")["pet"]
    names_ru = {r["ID"]: r for r in read_csv("BattlePetSpecies_ruRU", build)}
    stats = defaultdict(dict)
    for r in read_csv("BattlePetSpeciesState", build):
        stats[r["BattlePetSpeciesID"]][int(r["BattlePetStateID"])] = int(r["Value"])
    slots = defaultdict(lambda: [[], [], []])
    for r in sorted(read_csv("BattlePetSpeciesXAbility", build), key=lambda r: int(r["RequiredLevel"])):
        slots[r["BattlePetSpeciesID"]][int(r["SlotEnum"])].append(int(r["BattlePetAbilityID"]))
    breeds = parse_breed_lua()
    out = {}
    for r in read_csv("BattlePetSpecies", build):
        sid = r["ID"]
        st = stats[sid]
        ps = petsim.get(int(sid))
        creature = int(r["CreatureID"])
        out[int(sid)] = {
            "name": names.get(str(creature)) or names.get(sid) or "",
            "family": int(r["PetTypeEnum"]),
            "base": [8 + st.get(19, 0) / 200, 8 + st.get(18, 0) / 200, 8 + st.get(20, 0) / 200],
            "abilities": slots[sid],
            "breeds": breeds.get(int(sid)) or ([PETSIM_BREED[ps["breed"]]] if ps and ps.get("breed") in PETSIM_BREED else []),
            "creature": creature,
            "flags": int(r["Flags"]),
            "source": int(r["SourceTypeEnum"]),  # -1 = no journal source (unreleased / NPC only)
            # 0x20 HideFromJournal, 0x80 CantBattle; NPC-only species have 3 abilities
            "battle": not int(r["Flags"]) & 0xA0 and all(len(a) == 2 for a in slots[sid]),
            "states": {k: v for k, v in st.items() if k not in (18, 19, 20)},
        }
    return out


def build_breeds(build):
    breeds = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in read_csv("BattlePetBreedState", build):
        idx = {19: 0, 18: 1, 20: 2}.get(int(r["BattlePetStateID"]))
        if idx is not None:
            breeds[int(r["BattlePetBreedID"])][idx] = (int(r["Value"]) - 1600) / 200
    quality = {int(r["ID"]): float(r["StateMultiplier"]) * 2 for r in read_csv("BattlePetBreedQuality", build)}
    return {"breeds": dict(breeds), "quality": quality}


def build_tamers(species):
    other = read_json("petsim", "petsim_other.json")
    names = read_json("petsim", "petsim_names_en.json")
    creature_to_species = {p["id"]: p["petId"] for p in read_json("petsim", "petsim_pets.json")["pets"]}
    tamers = {}
    for row in other["tamer"]:
        if row["pet"] is None:  # empty slot (single-pet bosses)
            continue
        t = tamers.setdefault(
            row["tamerId"],
            {
                "npc": row["npc"],
                "name": names["tamer"].get(str(row["npc"])) or names["npc"].get(str(row["npc"]), ""),
                "zone": names["zone"].get(str(row["zone"]), row["zone"]),
                "wild": bool(row["tamerWild"]),
                "active": bool(row["active"]),
                "pets": [],
            },
        )
        sid = creature_to_species.get(row["pet"])
        choices = [int(c) for c in (row["abilities"] or "111")]
        sp_slots = species.get(sid, {}).get("abilities", [[], [], []])
        abilities = []
        for slot, choice in enumerate(choices):
            opts = sp_slots[slot] if slot < len(sp_slots) else []
            abilities.append(opts[choice - 1] if len(opts) >= choice else (opts[0] if opts else None))
        t["pets"].append(
            {
                "slot": row["petSlot"],
                "species": sid,
                "creature": row["pet"],
                "breed": PETSIM_BREED.get(row["breed"] or "bb", 3),
                "quality": row["quality"],
                "level": row["level"],
                "abilities": abilities,
                "behaviour": json.loads(row["behaviour"]) if row["behaviour"] else None,
                "special": row["specialCode"],
            }
        )
    # tamers missing from petsim, scraped from Wowhead (keyed by npc id; no AI script -> first usable ability)
    wh = RAW / "wowhead" / "old_world_tamers.json"
    if wh.exists():
        for npc, t in json.loads(wh.read_text(encoding="utf-8")).items():
            npc = int(npc)
            if any(x["npc"] == npc for x in tamers.values()):
                continue
            pets = []
            for i, p in enumerate(t["pets"]):
                breed = next((3 + b for b in range(10) if p["breeds"] >> b & 1), 3)
                pets.append({"slot": i + 1, "species": p["species"], "creature": None, "breed": breed,
                             "quality": p["minquality"], "level": p["level"], "abilities": p["abilities"],
                             "behaviour": None, "special": None})
            tamers[npc] = {"npc": npc, "name": t["name"], "zone": t["zone"], "wild": False, "active": True,
                           "pets": pets, "source": "wowhead"}
    for t in tamers.values():
        t["pets"].sort(key=lambda p: p["slot"])
    return tamers


def main():
    build = json.loads((RAW / "meta.json").read_text())["build"]
    OUT.mkdir(parents=True, exist_ok=True)
    abilities, effect_labels = build_abilities(build)
    species = build_species(build)
    states = {int(r["ID"]): r["LuaName"] for r in read_csv("BattlePetState", build)}
    tamers = build_tamers(species)
    files = {
        "abilities.json": abilities,
        "effects.json": effect_labels,
        "species.json": species,
        "states.json": states,
        "breeds.json": build_breeds(build),
        "tamers.json": tamers,
    }
    for name, data in files.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{name}: {len(data)}")


if __name__ == "__main__":
    main()
