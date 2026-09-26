"""Loads data/processed/*.json and builds pet stat blocks."""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def _load(name):
    with open(PROCESSED / name, encoding="utf-8") as f:
        return json.load(f)


def _int_keys(d):
    return {int(k): v for k, v in d.items()}


class GameData:
    def __init__(self):
        self.abilities = _int_keys(_load("abilities.json"))
        for aid, a in self.abilities.items():
            a["states"] = _int_keys(a["states"])
            a["_id"] = aid
        self.species = _int_keys(_load("species.json"))
        for s in self.species.values():
            s["states"] = _int_keys(s["states"])
        self.states = _int_keys(_load("states.json"))
        b = _load("breeds.json")
        self.breeds = _int_keys(b["breeds"])
        # BattlePetBreedQuality ids 7..12 = Poor..Legendary
        self.quality = {int(k) - 7: v for k, v in b["quality"].items()}
        self.tamers = _int_keys(_load("tamers.json"))
        self._by_name = {}
        for aid, a in self.abilities.items():
            self._by_name.setdefault(a["name"].lower(), aid)

    def ability_id(self, name):
        return self._by_name.get(name.lower())

    def species_by_name(self, name):
        name = name.lower()
        for sid, s in self.species.items():
            if s["name"].lower() == name:
                return sid
        raise KeyError(name)


@lru_cache(maxsize=1)
def game_data() -> GameData:
    return GameData()


@dataclass
class PetSpec:
    """A pet as it enters the battle (what a Rematch team line describes)."""
    species: int
    abilities: list  # 3 ability ids
    level: int = 25
    quality: int = 3  # 0 poor .. 3 rare .. 5 legendary
    breed: int = 3
    # explicit stat override (from logs / tamer data), else computed
    stats: tuple | None = None
    extra_states: dict = field(default_factory=dict)

    @classmethod
    def from_choices(cls, species, choices="111", **kw):
        """choices: 3 chars '1'/'2' picking the ability in each slot."""
        sp = game_data().species[species]
        abilities = []
        for slot, c in enumerate(choices):
            opts = sp["abilities"][slot]
            abilities.append(opts[int(c) - 1] if len(opts) >= int(c) else opts[0])
        if "breed" not in kw and sp["breeds"]:
            kw["breed"] = sp["breeds"][0]
        return cls(species=species, abilities=abilities, **kw)


def compute_stats(spec: PetSpec):
    """(health, power, speed) using the standard formulas.

    health = ((base + breed) * 5 * level * q) + 100, power/speed = (base + breed) * level * q,
    rounded to nearest integer as shown in the pet journal.
    """
    if spec.stats:
        return spec.stats
    gd = game_data()
    base = gd.species[spec.species]["base"]
    breed = gd.breeds.get(spec.breed, [0.5, 0.5, 0.5])
    q = gd.quality.get(spec.quality, 1.3)
    h = (base[0] + breed[0]) * 5 * spec.level * q + 100
    p = (base[1] + breed[1]) * spec.level * q
    s = (base[2] + breed[2]) * spec.level * q
    return round_half_up(h), round_half_up(p), round_half_up(s)


def round_half_up(x):
    return int(x + 0.5 + 1e-9)
