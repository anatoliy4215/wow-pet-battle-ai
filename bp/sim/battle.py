"""Pet battle engine.

The engine interprets DB2 ability data (turns -> effects -> opcodes, see ops.py) and hard-codes only
what the game tables do not describe: round structure, damage formula, family passives.

Round structure (one call to Battle.step):
  1. both sides submit an action: ability slot 0-2, swap to pet 0-2 (actions 3-5) or pass (6)
  2. swaps resolve first
  3. EV_ROUND_START triggers (auras and the chosen abilities, e.g. Quick Attack speed boost)
  4. faster pet acts, then the slower one (speed ties are random)
  5. EV_ROUND_END triggers, aura ticks, durations/cooldowns count down, family passives
  6. dead active pets are replaced by the controller before the next round (free action)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import constants as C
from .data import PetSpec, compute_stats, game_data


@dataclass(eq=False)
class Aura:
    aid: int
    ability: dict
    holder: "Pet | None"  # None = weather
    caster: "Pet | None"
    duration: int  # rounds remaining, <0 = permanent
    tick_first: bool = True
    fresh: bool = True
    stacks: int = 1
    uid: int = 0
    age: int = 0
    next_attack: bool = False  # Duration -1 debuffs that expire after the holder attacks
    blocks_left: int = 0
    team: object = None

    @property
    def name(self):
        return self.ability["name"]


@dataclass(eq=False)
class Pet:
    spec: PetSpec
    team: "Team"
    index: int
    family: int = 0
    name: str = ""
    base_max_hp: int = 0
    hp: int = 0
    base_power: int = 0
    base_speed: int = 0
    abilities: list = field(default_factory=list)
    cooldowns: list = field(default_factory=lambda: [0, 0, 0])
    auras: list = field(default_factory=list)
    states: dict = field(default_factory=dict)  # persistent states set by effects
    dead: bool = False
    mech_revived: bool = False
    undead_used: bool = False
    channel: tuple | None = None  # (slot, next turn index) for multi-round abilities
    dealt_damage_round: bool = False
    dragon_trigger: bool = False
    ramp: dict = field(default_factory=dict)  # ability id -> uses (ramping damage)
    swapped_in: int = 0  # round number after which this pet became active
    last_ability: int | None = None
    revealed: set = field(default_factory=set)  # ability slots the opponent has seen used (PvP)

    # ---- stats -------------------------------------------------------------------------------
    def state(self, sid, battle=None):
        v = self.states.get(sid, 0)
        v += game_data().species[self.spec.species]["states"].get(sid, 0) if sid not in (C.S_POWER, C.S_SPEED, C.S_STAMINA) else 0
        for a in self.auras:
            v += a.ability["states"].get(sid, 0) * a.stacks
        b = battle or self.team.battle
        if b is not None:
            if self.team.active_pet is self:
                for a in self.team.auras:
                    v += a.ability["states"].get(sid, 0)
            w = b.weather
            if w is not None and sid in w.ability["states"]:
                if not (self.family == C.ELEMENTAL and w.ability["states"][sid] < 0 and sid not in C.WEATHER_STATES):
                    v += w.ability["states"][sid]
        # conditional family passives
        if self.family == C.BEAST and sid == C.S_DMG_DEALT_PCT and self.hp * 2 < self.max_hp:
            v += C.BEAST_BONUS
        if self.family == C.FLYING and sid == C.S_SPEED_PCT and self.hp * 2 > self.max_hp:
            v += C.FLYING_SPEED_PCT
        return v

    @property
    def max_hp(self):
        pct = self.state(C.S_MAX_HEALTH_PCT)
        return max(1, int(self.base_max_hp * (100 + pct) / 100) + self.state(C.S_MAX_HEALTH_BONUS))

    @property
    def power(self):
        return max(0, self.base_power + self.state(C.S_POWER))

    @property
    def speed(self):
        return max(0, int((self.base_speed + self.state(C.S_SPEED)) * (100 + self.state(C.S_SPEED_PCT)) / 100))

    def has_aura(self, aid):
        return any(a.aid == aid for a in self.auras)

    def aura(self, aid):
        for a in self.auras:
            if a.aid == aid:
                return a
        return None

    def is_active(self):
        return self.team.active_pet is self

    def __repr__(self):
        return f"<{self.name} {self.hp}/{self.max_hp}{' dead' if self.dead else ''}>"


class Team:
    def __init__(self, battle, side, specs):
        self.battle = battle
        self.side = side
        self.pets = [make_pet(s, self, i) for i, s in enumerate(specs)]
        self.active = 0
        self.auras: list[Aura] = []  # auras attached to the team (traps, fields)

    @property
    def active_pet(self) -> Pet:
        return self.pets[self.active]

    @property
    def alive(self):
        return [p for p in self.pets if not p.dead]

    @property
    def enemy(self) -> "Team":
        return self.battle.teams[1 - self.side]


def make_pet(spec: PetSpec, team, index) -> Pet:
    gd = game_data()
    sp = gd.species[spec.species]
    h, p, s = compute_stats(spec)
    pet = Pet(spec=spec, team=team, index=index, family=sp["family"], name=sp["name"] or str(spec.species),
              base_max_hp=h, hp=h, base_power=p, base_speed=s, abilities=(list(spec.abilities) + [None] * 3)[:3])
    pet.states.update(spec.extra_states)
    return pet


class Battle:
    def __init__(self, team_a, team_b, seed=None, log=False):
        self.rng = random.Random(seed)
        self.teams = [Team(self, 0, team_a), Team(self, 1, team_b)]
        self.weather: Aura | None = None
        self.round = 0
        self.winner = None  # 0, 1, or -1 draw
        self.logging = log
        self.log: list[str] = []
        self._uid = 0
        self.unknown_ops: dict[int, int] = {}
        # test hooks: force RNG outcomes (None = random)
        self.force_hit = None
        self.force_crit = None
        self.force_variance = None  # -1..1 fraction of the variance range
        self.forced_miss = None  # {(ability name, target name): n} misses to reproduce (log replay)
        self.forced_crit = None
        self.forced_amount = None
        self.forced_heal = None  # {(ability name, target name): [amounts]} logged rolls to reproduce
        self.force_first = None  # side that acts first this round (speed ties in log replay)
        from . import ops  # noqa: F401 (registers handlers)
        self.ops = ops
        # boss / elite passives are species states; the game shows them as permanent auras
        for pet in self.all_pets():
            st = game_data().species[pet.spec.species]["states"]
            if st.get(C.S_BOSS):
                ops.add_aura(self, pet, C.BOSS_AURA, pet, -1)
            elif st.get(C.S_ELITE):
                ops.add_aura(self, pet, C.ELITE_AURA, pet, -1)

    # ---- helpers -----------------------------------------------------------------------------
    def emit(self, msg):
        if self.logging:
            self.log.append(f"[{self.round}] {msg}")

    def team_auras(self, team):
        return team.auras

    def all_pets(self):
        return self.teams[0].pets + self.teams[1].pets

    def active_pets(self):
        return [t.active_pet for t in self.teams]

    @property
    def over(self):
        return self.winner is not None

    def next_uid(self):
        self._uid += 1
        return self._uid

    # ---- action legality ---------------------------------------------------------------------
    def needs_replacement(self, side):
        t = self.teams[side]
        return t.active_pet.dead and bool(t.alive)

    def legal_actions(self, side):
        """Legal actions for the upcoming round, as a list of action ids."""
        t = self.teams[side]
        pet = t.active_pet
        if self.needs_replacement(side):
            return [3 + i for i, p in enumerate(t.pets) if not p.dead]
        if pet.channel is not None:
            return [C.A_PASS]  # forced continuation of a multi-round ability
        locked = pet.state(C.S_TURN_LOCK) > 0  # stunned / asleep: no abilities, but swapping is allowed
        acts = []
        for i, aid in enumerate(pet.abilities):
            if not locked and aid and pet.cooldowns[i] == 0 and self.ops.ability_usable(self, pet, aid):
                acts.append(i)
        if pet.state(C.S_SWAP_OUT_LOCK) <= 0:
            for i, p in enumerate(t.pets):
                if i != t.active and not p.dead and p.state(C.S_SWAP_IN_LOCK) <= 0:
                    acts.append(3 + i)
        acts.append(C.A_PASS)
        return acts

    # ---- swapping ----------------------------------------------------------------------------
    def swap(self, side, idx, forced=False):
        t = self.teams[side]
        old = t.active_pet
        if idx == t.active:
            return
        if not old.dead:
            self.ops.fire(self, C.EV_SWAP_OUT, old)
            old.channel = None
            old.states.pop(C.S_RAMPING_ID, None)  # log: Sleeping Gas back to base damage after a swap
            old.states.pop(C.S_RAMPING_USES, None)
            # auras flagged to drop on swap are removed by their own swap-out triggers
        t.active = idx
        new = t.active_pet
        new.swapped_in = self.round
        self.emit(f"{'A' if side == 0 else 'B'} swaps to {new.name}{' (forced)' if forced else ''}")
        self.ops.fire(self, C.EV_SWAP_IN, new)

    def replace_dead(self, side, idx):
        assert self.teams[side].pets[idx].dead is False
        self.swap(side, idx, forced=True)

    # ---- main round --------------------------------------------------------------------------
    def step(self, action_a, action_b):
        """Resolve a full round. Forced replacements (dead active pet) are handled without a round."""
        actions = [action_a, action_b]
        # free replacements
        replaced = False
        for side in (0, 1):
            if self.needs_replacement(side):
                idx = actions[side] - 3
                pets = self.teams[side].pets
                if not (0 <= idx < len(pets)) or pets[idx].dead:
                    # the need arose during this call (a swap-in trap killed the new pet): first alive pet
                    idx = next(i for i, p in enumerate(pets) if not p.dead)
                self.replace_dead(side, idx)
                replaced = True
        if replaced:
            return
        self.round += 1
        for p in self.all_pets():
            p.dealt_damage_round = False
            p.states[C.S_WAS_DAMAGED] = 0
            p.states[C.S_DID_DAMAGE] = 0

        # 2. swaps
        chosen = [None, None]
        for side in (0, 1):
            a = actions[side]
            pet = self.teams[side].active_pet
            if pet.channel is not None:
                chosen[side] = ("channel", pet.channel[0])
            elif a in C.A_SWAP:
                self.swap(side, a - 3)
                chosen[side] = ("swap", None)
            elif a in C.A_ABILITY and pet.state(C.S_TURN_LOCK) <= 0 and pet.abilities[a]:
                chosen[side] = ("ability", a)
            else:
                chosen[side] = ("pass", None)

        actors = [self.teams[0].active_pet, self.teams[1].active_pet]
        self._actors = actors

        # 3. round start triggers
        for side in (0, 1):
            pet = self.teams[side].active_pet
            kind, slot = chosen[side]
            ab_id = pet.abilities[slot] if kind in ("ability", "channel") else None
            self.ops.fire(self, C.EV_ROUND_START, pet, ability_id=ab_id)

        # 4. order
        pa, pb = self.teams[0].active_pet, self.teams[1].active_pet
        if pa.speed == pb.speed and self.force_first is not None:
            order = [self.force_first, 1 - self.force_first]
        elif pa.speed != pb.speed:
            order = [0, 1] if pa.speed > pb.speed else [1, 0]
        else:
            order = [0, 1] if self.rng.random() < 0.5 else [1, 0]
        self.first_side = order[0]
        for side in order:
            pet = self.teams[side].active_pet
            if pet.dead or self.over or pet is not actors[side]:
                continue  # dead, or forced out by an enemy swap effect: the action is lost
            kind, slot = chosen[side]
            if kind not in ("ability", "channel"):
                continue
            if pet.state(C.S_TURN_LOCK) > 0 and kind == "ability":
                continue  # got stunned earlier this round
            pet.revealed.add(slot)
            # an opponent who swaps or passes does not act: the caster counts as first
            # (log 2026-09-25: Batter's 'if first' hit landed while we swapped / passed)
            idle = chosen[1 - side][0] not in ("ability", "channel")
            self.ops.use_ability(self, pet, slot, first=(side == order[0]) or idle, last=(side == order[1]))
            self.check_end()

        # 5. end of round
        if not self.over:
            self.end_of_round(chosen)
        self.check_end()
        self.ops.clear_dead_auras(self)
        if self.round >= C.MAX_ROUNDS and not self.over:
            self.winner = -1

    def end_of_round(self, chosen):
        ops = self.ops
        for side in (0, 1):
            pet = self.teams[side].active_pet
            kind, slot = chosen[side]
            ab_id = pet.abilities[slot] if kind in ("ability", "channel") and pet is self._actors[side] else None
            ops.fire(self, C.EV_ROUND_END, pet, ability_id=ab_id)
        if self.over:
            return
        ops.clear_dead_auras(self, untimed_only=True)  # see clear_dead_auras
        ops.tick_auras(self)
        if self.over:
            return
        for pet in self.all_pets():
            if not pet.dead and pet.family == C.DRAGONKIN and pet.dragon_trigger:
                ops.add_aura(self, pet, C.DRAGONKIN_AURA, pet, 1)
            pet.dragon_trigger = False
        for side in (0, 1):
            pet = self.teams[side].active_pet
            kind, slot = chosen[side]
            ab_id = pet.abilities[slot] if kind in ("ability", "channel") and pet is self._actors[side] else None
            # the used ability's own post-aura-tick turn (Feign Death: drop the dodge, swap out)
            ops.fire(self, C.EV_POST_TICK, pet, ability_id=ab_id)
        ops.countdown_auras(self)
        # Humanoid recovery comes after expiring auras (log 2026-09-25: full heal in the round Darkness
        # faded, half heal while it lasted); a benched Fiendish Imp healed after its field hit (older log)
        for pet in self.all_pets():
            if not pet.dead and pet.family == C.HUMANOID and pet.dealt_damage_round:
                ops.run_passive_ability(self, pet, C.HUMANOID_RECOVERY_AURA)
        ops.clear_dead_auras(self)
        for pet in self.all_pets():
            for i in range(3):
                if pet.cooldowns[i] > 0:
                    pet.cooldowns[i] -= 1

    def check_end(self):
        if self.winner is not None:
            return
        a_alive = bool(self.teams[0].alive)
        b_alive = bool(self.teams[1].alive)
        if not a_alive and not b_alive:
            self.winner = -1
        elif not a_alive:
            self.winner = 1
        elif not b_alive:
            self.winner = 0
