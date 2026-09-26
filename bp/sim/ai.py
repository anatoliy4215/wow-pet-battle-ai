"""Controllers: NPC tamer script AI (tdBattlePetScript subset as used by wow-petsim) and simple bots.

Supported script lines:  use(Name:id) [cond & cond | cond],  use(random),  standby
Conditions: self./enemy. hpp, hp, hp.full, aura(X).exists, ability(X).usable, round;
            round (battle), last_played, rand;  operators <,<=,>,>=,=,!=, "% a,b", "~a,b,c", leading "!".
"""
from __future__ import annotations

import re

from . import constants as C
from .data import game_data

_USE = re.compile(r"^\s*(use|standby)\s*(?:\(([^)]*)\))?\s*(?:\[(.*)\])?\s*$")
_CMP = re.compile(r"^(.*?)\s*(<=|>=|!=|=|<|>|%|~)\s*(.*)$")


class Controller:
    def choose(self, battle, side) -> int:
        raise NotImplementedError


class RandomController(Controller):
    def choose(self, battle, side):
        return battle.rng.choice(battle.legal_actions(side))


class FirstAbilityController(Controller):
    """Always the first legal ability (a trivial baseline)."""
    def choose(self, battle, side):
        acts = battle.legal_actions(side)
        abil = [a for a in acts if a in C.A_ABILITY]
        return abil[0] if abil else acts[0]


class ScriptController(Controller):
    """Plays a tamer team following per-pet behaviour scripts."""

    def __init__(self, scripts, swap_when_cc=False):
        self.scripts = [[self._parse(line) for line in (s or [])] for s in scripts]
        self.swap_when_cc = swap_when_cc  # some tamers swap out a stunned / sleeping pet (seen in logs)
        self.last_used = {}  # (pet index, ability id) -> round used

    @staticmethod
    def _parse(line):
        m = _USE.match(line)
        if not m:
            return None
        kind, arg, cond = m.groups()
        aid = None
        if kind == "use" and arg and arg.strip() != "random":
            if ":" in arg:
                aid = int(arg.rsplit(":", 1)[1])
            else:
                aid = game_data().ability_id(arg.strip())
        return {"kind": kind, "ability": aid, "random": kind == "use" and (not arg or arg.strip() == "random"), "cond": cond}

    def choose(self, battle, side):
        acts = battle.legal_actions(side)
        team = battle.teams[side]
        if battle.needs_replacement(side):
            return acts[0]  # next alive pet in slot order
        pet = team.active_pet
        if acts == [C.A_PASS]:
            return C.A_PASS
        if pet.state(C.S_TURN_LOCK) > 0:  # stunned / asleep
            swaps = [a for a in acts if a in C.A_SWAP]
            return swaps[0] if (self.swap_when_cc and swaps) else C.A_PASS
        usable = {pet.abilities[a]: a for a in acts if a in C.A_ABILITY}
        rules = self.scripts[pet.index] if pet.index < len(self.scripts) else []
        for r in rules:
            if r is None:
                continue
            if r["cond"] and not self._eval(r["cond"], battle, side, pet, r):
                continue
            if r["kind"] == "standby":
                return C.A_PASS
            if r["random"]:
                if usable:
                    return self._mark(battle, pet, battle.rng.choice(list(usable.items())))
                continue
            if r["ability"] in usable:
                return self._mark(battle, pet, (r["ability"], usable[r["ability"]]))
        if usable:
            return self._mark(battle, pet, next(iter(usable.items())))
        return C.A_PASS

    def _mark(self, battle, pet, item):
        aid, act = item
        self.last_used[(pet.index, aid)] = battle.round + 1
        return act

    # ---- condition evaluation --------------------------------------------------------------
    def _eval(self, expr, battle, side, pet, rule):
        # '|' has lower precedence than '&'
        return any(all(self._atom(a.strip(), battle, side, pet, rule) for a in part.split("&"))
                   for part in expr.split("|"))

    def _atom(self, atom, battle, side, pet, rule):
        neg = atom.startswith("!")
        if neg:
            atom = atom[1:].strip()
        m = _CMP.match(atom)
        if m and not atom.endswith(".exists") and not atom.endswith(".usable") or (m and m.group(2) in "%~"):
            lhs, opr, rhs = m.groups()
            v = self._value(lhs.strip(), battle, side, pet, rule)
            if v is None:
                res = False
            elif opr == "%":
                a, b = (int(x) for x in rhs.split(","))
                res = v % a == b
            elif opr == "~":
                res = v in {int(x) for x in rhs.split(",")}
            else:
                n = float(rhs)
                res = {"<": v < n, "<=": v <= n, ">": v > n, ">=": v >= n, "=": v == n, "!=": v != n}[opr]
        else:
            v = self._value(atom, battle, side, pet, rule)
            res = bool(v)
        return not res if neg else res

    def _value(self, key, battle, side, pet, rule):
        nxt = battle.round + 1
        if key == "rand":
            return battle.rng.randint(0, 100)
        if key == "round":
            return nxt
        if key == "last_played":
            r = self.last_used.get((pet.index, rule["ability"]))
            return 999 if r is None else nxt - r
        who, _, rest = key.partition(".")
        if who not in ("self", "enemy"):
            return None
        p = pet if who == "self" else battle.teams[1 - side].active_pet
        if rest == "hpp":
            return 100 * p.hp / p.max_hp
        if rest == "hp":
            return p.hp
        if rest == "hp.full":
            return p.hp >= p.max_hp
        if rest == "round":
            return nxt - getattr(p, "swapped_in", 0)
        if rest == "dead":
            return p.dead
        m = re.match(r"(aura|ability)\((.*?)\)\.(exists|usable|duration)", rest)
        if m:
            kind, name, prop = m.groups()
            aid = int(name.rsplit(":", 1)[1]) if ":" in name else game_data().ability_id(name)
            if kind == "aura":
                a = p.aura(aid)
                return (a is not None) if prop == "exists" else (a.duration if a else 0)
            if aid in p.abilities:
                return p.cooldowns[p.abilities.index(aid)] == 0
            return False
        return None
