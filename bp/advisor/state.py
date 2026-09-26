"""BPAdvisor QR payload -> simulator Battle (our side = 0) + the current phase.

Payload v1 (little endian), written by addon/BPAdvisor/BPAdvisor.lua:
  'B' 1 round flags(bit0 pick a pet, bit1 first pick) activeAlly activeEnemy
  weather auras, then for ally and enemy: pad auras, n pets, per pet:
    species u16, level u8, quality u8, hp u16, maxhp u16, power u16, speed u16, speed% + 128 u8,
    3x ability id u16 (0 = enemy ability not seen yet), 3x cooldown u8, usable bits u8, auras
  auras: n u8, then per aura: id u16, turns + 1 u8 (0 = permanent), caster owner*16 + index u8
"""
from __future__ import annotations

from ..pvp.features import PHASE_LEAD, PHASE_REPLACE, PHASE_ROUND
from ..sim import constants as C
from ..sim.battle import Battle
from ..sim.data import PetSpec, game_data
from ..validate.replay import _auras


class Reader:
    def __init__(self, b):
        self.b, self.i = b, 0

    def u8(self):
        v = self.b[self.i]
        self.i += 1
        return v

    def u16(self):
        v = self.b[self.i] | (self.b[self.i + 1] << 8)
        self.i += 2
        return v

    def auras(self):
        out = []
        for _ in range(self.u8()):
            aid, t, c = self.u16(), self.u8(), self.u8()
            out.append({"id": aid, "turns": t - 1 if t else -1, "co": c >> 4, "ci": c & 15})
        return out


def decode(data: bytes):
    r = Reader(data)
    if r.u8() != 66 or r.u8() != 1:
        raise ValueError("not a BPAdvisor payload")
    st = {"round": r.u8(), "flags": r.u8(), "active": [r.u8(), r.u8()], "weather": r.auras(), "pads": [], "pets": []}
    for _ in range(2):
        st["pads"].append(r.auras())
        pets = []
        for _ in range(r.u8()):
            p = {"species": r.u16(), "level": r.u8(), "quality": r.u8(), "hp": r.u16(), "maxhp": r.u16(),
                 "power": r.u16(), "speed": r.u16(), "speed_pct": r.u8() - 128,
                 "abilities": [r.u16(), r.u16(), r.u16()], "cd": [r.u8(), r.u8(), r.u8()], "usable": r.u8()}
            p["auras"] = r.auras()
            pets.append(p)
        st["pets"].append(pets)
    return st


def build(st):
    """-> (battle, phase). Unseen enemy abilities get a placeholder and stay hidden (not in `revealed`)."""
    gd = game_data()
    teams = []
    for owner in (0, 1):
        specs = []
        for p in st["pets"][owner]:
            slots = gd.species[p["species"]]["abilities"]
            ab = [a if a else slots[k][0] for k, a in enumerate(p["abilities"])]
            pct = p["speed_pct"]
            speed = round(p["speed"] * 100 / (100 + pct)) if pct > -100 and pct else p["speed"]
            specs.append(PetSpec(species=p["species"], abilities=ab, level=p["level"] or 25, quality=3, breed=3,
                                 stats=(p["maxhp"], p["power"], speed)))
        teams.append(specs)
    b = Battle(teams[0], teams[1], seed=0)
    b.round = st["round"]
    for owner in (0, 1):
        team = b.teams[owner]
        for pet, p in zip(team.pets, st["pets"][owner]):
            pet.hp = p["hp"]
            pet.dead = p["hp"] <= 0
            pet.auras = []
            for s in range(3):
                pet.cooldowns[s] = p["cd"][s]
            if owner == 1:
                pet.revealed = {s for s in range(3) if p["abilities"][s]}
            else:
                pet.revealed = {0, 1, 2}
        if st["active"][owner]:
            team.active = st["active"][owner] - 1
    # auras after all pets exist (casters may be on the other team)
    for owner in (0, 1):
        team = b.teams[owner]
        for pet, p in zip(team.pets, st["pets"][owner]):
            pet.auras = _auras(b, pet, p["auras"])
            pet.mech_revived = any(x["id"] == C.PASSIVE_ABILITY[C.MECHANICAL] for x in p["auras"])
            pet.undead_used = any(x["id"] == C.UNDEAD_AURA for x in p["auras"])
        team.auras = _auras(b, team, st["pads"][owner])
    w = _auras(b, None, st["weather"])
    b.weather = w[0] if w else None
    # the game reports max health with modifiers (Sunny Day...): recover the base the simulator scales
    for pet in b.all_pets():
        pct = pet.state(C.S_MAX_HEALTH_PCT)
        if pct:
            pet.base_max_hp = round(pet.base_max_hp * 100 / (100 + pct))
    if st["flags"] & 2:
        phase = PHASE_LEAD
    elif st["flags"] & 1 or b.needs_replacement(0):
        phase = PHASE_REPLACE
    else:
        phase = PHASE_ROUND
    return b, phase


def encode_snapshot(snap, round_=0, flags=0):
    """BPLogger snapshot -> payload bytes, the same layout the addon writes (for tests)."""
    out = bytearray([66, 1, min(255, round_), flags, snap["active"][0] or 0, snap["active"][1] or 0])

    def u16(v):
        v = max(0, min(65535, int(v or 0)))
        out.extend((v & 255, v >> 8))

    def auras(lst):
        lst = lst or []
        out.append(len(lst))
        for a in lst:
            u16(a["id"])
            t = a.get("turns")
            out.append(0 if t is None or t < 0 else min(255, t + 1))
            out.append(((a.get("co") or 0) << 4) | (a.get("ci") or 0))

    auras(snap.get("weather"))
    for owner in (0, 1):
        auras(snap.get(f"pad{owner + 1}"))
        pets = snap["pets"][owner]
        out.append(len(pets))
        for p in pets:
            u16(p["species"]); out.append(p.get("level") or 25); out.append(p.get("quality") or 4)
            u16(p["hp"]); u16(p["maxhp"]); u16(p["power"]); u16(p["speed"])
            out.append(max(0, min(255, (p.get("st") or {}).get(25, 0) + 128)))
            for s in range(3):
                u16((p.get("abilities") or [0, 0, 0])[s])
            usable = 0
            for s in range(3):
                cd = (p.get("cd") or [[1, 0, 0]] * 3)[s]
                out.append(max(cd[1] or 0, cd[2] or 0))
                usable |= (1 << s) if cd[0] else 0
            out.append(usable)
            auras(p.get("auras"))
    return bytes(out)
