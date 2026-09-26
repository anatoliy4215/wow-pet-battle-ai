"""Fast battle copy for look-ahead search.

copy.deepcopy of a mid-battle state costs ~1.5 ms and dominated the search time. Here only the
mutable parts are copied (pets' health / cooldowns / states / auras, team pads, weather) and every
pet reference inside auras is re-pointed at the copy. Game data (ability dicts, species) and the
ops module are shared, exactly as with the deepcopy memo used before.
"""
from __future__ import annotations

import random

from .battle import Aura, Battle, Pet, Team


def _copy(obj, cls):
    new = cls.__new__(cls)
    new.__dict__.update(obj.__dict__)
    return new


def fast_clone(b: Battle, seed=None) -> Battle:
    nb = _copy(b, Battle)
    nb.rng = random.Random(seed)
    nb.log = list(b.log) if b.logging else []
    nb.unknown_ops = dict(b.unknown_ops)
    pets, teams = {}, {}
    nb.teams = []
    for t in b.teams:
        nt = _copy(t, Team)
        nt.battle = nb
        teams[id(t)] = nt
        nt.pets = []
        for p in t.pets:
            q = _copy(p, Pet)
            q.team = nt
            q.cooldowns = list(p.cooldowns)
            q.abilities = list(p.abilities)
            q.states = dict(p.states)
            q.ramp = dict(p.ramp)
            q.revealed = set(p.revealed)
            pets[id(p)] = q
            nt.pets.append(q)
        nb.teams.append(nt)

    def aura(a):
        na = _copy(a, Aura)
        if a.holder is not None:
            na.holder = pets.get(id(a.holder), a.holder)
        if a.caster is not None:
            na.caster = pets.get(id(a.caster), a.caster)
        if a.team is not None:
            na.team = teams.get(id(a.team), a.team)
        return na

    for t, nt in zip(b.teams, nb.teams):
        nt.auras = [aura(a) for a in t.auras]
        for p, q in zip(t.pets, nt.pets):
            q.auras = [aura(a) for a in p.auras]
    nb.weather = aura(b.weather) if b.weather is not None else None
    if hasattr(b, "_actors"):
        nb._actors = [pets.get(id(p), p) for p in b._actors]
    return nb
