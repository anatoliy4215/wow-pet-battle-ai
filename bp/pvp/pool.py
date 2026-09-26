"""PvP pet pool and team sampling.

Pool = every species a player can own (journal-visible, can battle, has a journal source), each with
all of its breeds, rare quality, level 25.  Species that are identical in battle (same family, base
stats, abilities and passive states - recolours) form one class; uniform sampling is over classes so
a model with 12 identical cats does not see cats 12x as often.

Opponent teams are a mix of
  * uniform random pets (coverage: the model must handle anything),
  * "meta" pets / full meta teams from data/pvp/meta.json (what players really bring),
  * teams added at run time (e.g. teams the team builder found strong, or seen in the user's logs).

data/pvp/disabled.json lists species the team builder must never suggest (not used in training).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from ..sim.data import PetSpec, game_data

ROOT = Path(__file__).resolve().parents[2]
PVP_DIR = ROOT / "data" / "pvp"
BREED_NAMES = {3: "B/B", 4: "P/P", 5: "S/S", 6: "H/H", 7: "H/P", 8: "P/S", 9: "H/S", 10: "P/B", 11: "S/B", 12: "H/B"}
BREED_IDS = {v: k for k, v in BREED_NAMES.items()}


def pvp_species():
    gd = game_data()
    return sorted(s for s, v in gd.species.items() if v["battle"] and v.get("source", 0) >= 0)


@lru_cache(maxsize=1)
def classes():
    """[[species ids]] of battle-equivalent species."""
    gd = game_data()
    by = defaultdict(list)
    for s in pvp_species():
        v = gd.species[s]
        key = (v["family"], tuple(v["base"]), tuple(tuple(x) for x in v["abilities"]), tuple(sorted(v["states"].items())))
        by[key].append(s)
    return [sorted(x) for x in by.values()]


def breeds_of(species):
    b = game_data().species[species]["breeds"]
    return list(b) if b else [3]


def resolve_species(x):
    if isinstance(x, int):
        return x
    return game_data().species_by_name(x)


def _choices(species, abilities):
    """abilities: '121' string, or list of 3 ability names/ids -> '121' string (None = random)."""
    if abilities is None:
        return None
    if isinstance(abilities, str) and len(abilities) == 3 and set(abilities) <= {"1", "2", "?"}:
        return abilities  # '?' = either option (random each battle)
    gd = game_data()
    slots = gd.species[species]["abilities"]
    out = ""
    for k, a in enumerate(abilities):
        aid = a if isinstance(a, int) else None
        if aid is None:
            aid = next((x for x in slots[k] if gd.abilities[x]["name"].lower() == str(a).lower()), None)
        if aid not in slots[k]:
            raise ValueError(f"{gd.species[species]['name']}: ability {a!r} is not in slot {k + 1}")
        out += str(slots[k].index(aid) + 1)
    return out


def load_meta(path=None):
    """data/pvp/meta.json -> (pets, teams) with resolved ids.

    pets:  [{"species": 1234 | "Anomalus", "breeds": ["P/P", ...] (optional), "abilities": "121" |
             [names] | [["121", w], ["112", w]] (optional), "weight": w}]
    teams: [{"pets": [pet entry, pet entry, pet entry], "weight": w}]
    """
    path = Path(path or PVP_DIR / "meta.json")
    if not path.exists():
        return [], []
    raw = json.loads(path.read_text(encoding="utf-8"))

    def pet(e):
        s = resolve_species(e["species"])
        br = [BREED_IDS.get(b, b) for b in e.get("breeds", [])] or None
        ab = e.get("abilities")
        if ab and isinstance(ab, list) and isinstance(ab[0], list) and len(ab[0]) == 2 and not isinstance(ab[0][0], int):
            loadouts = [(_choices(s, a), w) for a, w in ab]
        elif ab:
            loadouts = [(_choices(s, ab), 1.0)]
        else:
            loadouts = None
        return {"species": s, "breeds": br, "loadouts": loadouts, "weight": float(e.get("weight", 1.0))}

    pets = [pet(e) for e in raw.get("pets", [])]
    teams = [{"pets": [pet(p) for p in t["pets"]], "weight": float(t.get("weight", 1.0))} for t in raw.get("teams", [])]
    return pets, teams


def load_observed(path=None, weight=1.0):
    """data/pvp/observed.json (bp.pvp.observed, from the user's PvP logs) -> (pets, teams) like load_meta."""
    path = Path(path or PVP_DIR / "observed.json")
    if not path.exists():
        return [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    gd = game_data()
    pets, by_species = [], {}
    for e in raw.get("pets", []):
        s = e["species"]
        if s not in gd.species or not gd.species[s]["battle"]:
            continue
        br = [int(b) for b in e.get("breeds", {})] or None
        ch = [(c, float(n)) for c, n in e.get("choices", {}).items()] or None
        pet = {"species": s, "breeds": br, "loadouts": ch, "weight": weight * e["count"]}
        pets.append(pet)
        by_species[s] = pet
    teams = []
    for t in raw.get("teams", []):
        if all(s in by_species for s in t["species"]):
            teams.append({"pets": [by_species[s] for s in t["species"]], "weight": weight * t["count"]})
    return pets, teams


def load_disabled(path=None):
    path = Path(path or PVP_DIR / "disabled.json")
    if not path.exists():
        return set()
    return {resolve_species(x) for x in json.loads(path.read_text(encoding="utf-8")).get("species", [])}


class TeamSampler:
    def __init__(self, seed=None, meta_pet_frac=0.35, meta_team_frac=0.15, meta_path=None):
        self.rng = random.Random(seed)
        self.classes = classes()
        self.meta_pets, self.meta_teams = load_meta(meta_path)
        op, ot = load_observed(weight=3.0)  # real opponents of the user count more than web lists
        self.meta_pets += op
        self.meta_teams += ot
        self.meta_pet_frac = meta_pet_frac if self.meta_pets else 0.0
        self.meta_team_frac = meta_team_frac if self.meta_teams else 0.0
        self.extra_teams = []  # [(team entry, weight)] added at run time

    def _spec(self, species, breeds=None, loadouts=None):
        rng = self.rng
        breed = rng.choice(breeds or breeds_of(species))
        if loadouts:
            choices = rng.choices([c for c, _ in loadouts], [w for _, w in loadouts])[0]
        else:
            choices = None
        if choices is None:
            choices = "???"
        choices = "".join(c if c in "12" else rng.choice("12") for c in choices)
        return PetSpec.from_choices(species, choices, breed=breed)

    def random_pet(self):
        return self._spec(self.rng.choice(self.rng.choice(self.classes)))

    def meta_pet(self):
        e = self.rng.choices(self.meta_pets, [p["weight"] for p in self.meta_pets])[0]
        return self._spec(e["species"], e["breeds"], e["loadouts"])

    def team(self):
        r = self.rng.random()
        if r < self.meta_team_frac or (self.extra_teams and r < self.meta_team_frac + 0.1):
            src = self.meta_teams + [t for t, _ in self.extra_teams]
            w = [t["weight"] for t in self.meta_teams] + [w for _, w in self.extra_teams]
            t = self.rng.choices(src, w)[0]
            pets = [self._spec(p["species"], p["breeds"], p["loadouts"]) for p in t["pets"]]
            self.rng.shuffle(pets)
            return pets
        pets, seen = [], set()
        while len(pets) < 3:
            p = self.meta_pet() if self.rng.random() < self.meta_pet_frac else self.random_pet()
            if p.species in seen:
                continue
            seen.add(p.species)
            pets.append(p)
        return pets
