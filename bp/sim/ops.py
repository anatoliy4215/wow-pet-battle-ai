"""Effect interpreter: executes ability turns / aura ticks / event triggers.

Context for every effect list:
  me      - the pet whose ability / aura this is (aura holder for auras, user for abilities)
  target  - default "target": enemy active pet for abilities, the holder for aura ticks,
            the other party for damage-taken / damage-dealt triggers
  caster  - whose power scales the numbers (aura applier for auras)
  family  - ability family used for strong/weak

Opcode semantics: docs/opcodes.md. Per-row targets: targets.py. Unknown opcodes are counted in
battle.unknown_ops so we can see what is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from . import constants as C
from .battle import Aura, Pet, Team
from .data import game_data
from .targets import chain_rows, plan

HANDLERS = {}

DAMAGE_OPS_DIRECT = set()


def op(*codes):
    def deco(fn):
        for c in codes:
            HANDLERS[c] = fn
        return fn
    return deco


@dataclass
class Ctx:
    me: Pet
    target: Pet | None
    caster: Pet
    family: int
    ability_id: int
    aura: Aura | None = None
    periodic: bool = False
    failed: bool = False  # previous effect failed (for ChainFailure / gates)
    missed: bool = False  # previous effect missed specifically
    stop: bool = False
    first: bool = False  # the acting pet moved first this round
    last: bool = False   # the acting pet moved last this round
    in_trigger: bool = False
    row_target: object = None  # resolved target for the current row (Pet or Team)
    prev_target: object = None
    extra: dict = field(default_factory=dict)
    blocked: bool = False


# ---------------------------------------------------------------------------------------------
# entry points used by battle.py
# ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _needs_corpse(aid):
    """Abilities that start with op144 'select an unused allied corpse' (Consume Corpse)."""
    ab = game_data().abilities.get(aid, {})
    for t in ab.get("turns", []):
        if t["type"] == 0:
            return bool(t["effects"]) and t["effects"][0]["op"] == 144
    return False


def _free_corpse(pet):
    return next((p for p in pet.team.pets if p.dead and p is not pet and not p.states.get(C.S_CONSUMED_CORPSE)), None)


def ability_usable(battle, pet, aid):
    # the game greys out Consume Corpse until an ally has died (user, 2026-09-26: advised with no corpse)
    if _needs_corpse(aid) and _free_corpse(pet) is None:
        return False
    return True


def use_ability(battle, pet: Pet, slot: int, first=False, last=False):
    aid = pet.abilities[slot]
    ab = game_data().abilities[aid]
    turns = [(i, t) for i, t in enumerate(ab["turns"]) if t["type"] == 0]
    if pet.channel is not None:
        slot, idx = pet.channel
    else:
        idx = 0
        if ab["cooldown"]:
            pet.cooldowns[slot] = ab["cooldown"] + 1  # counts down at end of this round
        # ability states (BattlePetAbilityState) are added to the caster on first cast
    if not turns:
        return
    ti, turn = turns[min(idx, len(turns) - 1)]
    pet.channel = (slot, idx + 1) if idx + 1 < len(turns) else None
    pet.last_ability = aid
    battle.emit(f"{pet.name} uses {ab['name']}")
    enemy = pet.team.enemy.active_pet
    ctx = Ctx(me=pet, target=enemy, caster=pet, family=ab["family"], ability_id=aid, first=first, last=last)
    pet.using = aid  # the ability's own trigger turns (e.g. Leech Life's on-damage-dealt heal) are live now
    try:
        run_effects(battle, turn["effects"], ctx, aid, ti)
    finally:
        pet.using = None
    if not battle.over:
        fire(battle, C.EV_ABILITY, pet, other=enemy)


def resolve_target(battle, ctx: Ctx, code):
    me = ctx.me  # aura rules are written from the holder's point of view (Cyclone on your pad hits your pets)
    if code is None:
        return None
    if code == "C":
        return ctx.caster
    if code == "T":
        return ctx.target  # the pet that set off the trigger (the attacker, for 'when damaged')
    side = me.team if code[0] == "A" else me.team.enemy
    if code[1] == "P":
        return side
    n = int(code[1])
    if n == 0:
        return side.active_pet
    backups = [p for i, p in enumerate(side.pets) if i != side.active]
    return backups[n - 1] if n - 1 < len(backups) else None


def run_effects(battle, effects, ctx: Ctx, aid=None, turn_index=None):
    tplan = plan(aid, 0) if aid is not None and turn_index is not None else {}
    for ei, e in enumerate(effects):
        if ctx.stop or battle.over:
            break
        p = e["params"]
        if ctx.failed and (p.get("ChainFailure") or (aid, turn_index, ei) in chain_rows()):
            continue
        h = HANDLERS.get(e["op"])
        if h is None:
            battle.unknown_ops[e["op"]] = battle.unknown_ops.get(e["op"], 0) + 1
            continue
        code = tplan.get((turn_index, ei))
        tgt = resolve_target(battle, ctx, code)
        ctx.row_target = tgt
        if code is not None and tgt is None:
            ok = False  # e.g. no backup pet left
        else:
            ok = h(battle, e, ctx)
        if tgt is not None:
            ctx.prev_target = tgt
        if ok is not None:
            ctx.failed = not ok
            ctx.missed = ok is False


def _tgt(ctx: Ctx, default_enemy=True):
    """Pet targeted by the current row (override, else default)."""
    t = ctx.row_target
    if isinstance(t, Team):
        return t.active_pet
    if t is not None:
        return t
    return ctx.target if default_enemy else ctx.me


def fire(battle, event, pet: Pet, other: Pet | None = None, ability_id=None, aura_filter=None):
    """Run TurnType=1 triggers for `event` on `pet`'s auras (and on ability_id if given)."""
    if pet is None or (battle.over and event != C.EV_DAMAGE_TAKEN):
        return
    depth = getattr(battle, "_fire_depth", 0)
    if depth >= 3:  # triggers caused by triggers: stop runaway chains
        return
    battle._fire_depth = depth + 1
    try:
        _fire(battle, event, pet, other, ability_id, aura_filter)
    finally:
        battle._fire_depth = depth


def _fire(battle, event, pet, other, ability_id, aura_filter):
    gd = game_data()
    sources = []
    if aura_filter is not None:
        sources.append((aura_filter.ability, aura_filter, aura_filter.caster or pet))
    else:
        for a in list(pet.auras):
            sources.append((a.ability, a, a.caster or pet))
        if pet.is_active():
            for a in list(pet.team.auras):
                sources.append((a.ability, a, a.caster or pet))
        if battle.weather is not None and event != C.EV_REMOVE:
            sources.append((battle.weather.ability, battle.weather, battle.weather.caster or pet))
        if ability_id:
            sources.append((gd.abilities[ability_id], None, pet))
        elif getattr(pet, "using", None) and event in (C.EV_DAMAGE_DEALT, C.EV_DAMAGE_TAKEN):
            sources.append((gd.abilities[pet.using], None, pet))
    for ab, aura, caster in sources:
        for ti, t in enumerate(ab["turns"]):
            if t["type"] != 1 or t["event"] != event:
                continue
            if aura is not None and aura_filter is None and not _aura_alive(battle, pet, aura):
                break
            tgt = other if other is not None else pet.team.enemy.active_pet
            if event == C.EV_REMOVE and other is None:
                tgt = pet
            if caster is not None and caster.team is not pet.team and aura is not None and event in (C.EV_DAMAGE_TAKEN, C.EV_DAMAGE_DEALT, C.EV_ABILITY):
                pass
            ctx = Ctx(me=pet, target=tgt, caster=caster if aura is not None else pet, family=ab["family"],
                      ability_id=ab["_id"], aura=aura, in_trigger=True,
                      first=getattr(battle, "first_side", -1) == pet.team.side)
            run_effects(battle, t["effects"], ctx, ab["_id"], ti)


def _aura_alive(battle, pet, aura):
    return aura in pet.auras or aura in pet.team.auras or aura is battle.weather


def tick_auras(battle):
    """Aura processing: run each aura's type-0 turn for its current age (weather, pads, pets)."""
    first = getattr(battle, "first_side", 0)
    order = [battle.teams[first], battle.teams[1 - first]]
    holders = []
    if battle.weather is not None:
        holders.append((None, battle.weather))
    # pet auras tick before team-pad objects (log: Bleeding hit an immortal Undead before Egg Barrage)
    for team in order:
        for pet in team.pets:
            for a in list(pet.auras):
                holders.append((pet, a))
    for team in order:
        for a in list(team.auras):
            holders.append((team, a))
    for holder, a in holders:
        if battle.over:
            return
        turns = [(i, t) for i, t in enumerate(a.ability["turns"]) if t["type"] == 0]
        age = a.age
        a.age += 1
        if not turns:
            continue
        if age < len(turns):
            ti, turn = turns[age]
        elif len(turns) == 1:
            ti, turn = turns[0]
        else:
            continue
        if not turn["effects"]:
            continue
        if isinstance(holder, Pet):
            if holder.dead or not _aura_alive(battle, holder, a):
                continue
            me = tgt = holder
        elif isinstance(holder, Team):
            if a not in holder.auras:
                continue
            # pad objects (turrets, Egg Barrage, bombs) keep ticking when the pet in front is dead
            me = holder.active_pet
            # an object on its caster's own pad attacks the enemy; one placed on the enemy pad hits its holder
            own = a.caster is not None and a.caster.team is holder
            tgt = holder.enemy.active_pet if own else me
            ctx = Ctx(me=me, target=tgt, caster=a.caster or me, family=a.ability["family"], ability_id=a.aid, aura=a)
            run_effects(battle, turn["effects"], ctx, a.aid, ti)
            continue
        else:  # weather: its rows hit the active pets of both teams (caster's team first)
            caster = a.caster
            heads = battle.active_pets()
            if caster is not None and caster.team.side == 1:
                heads = heads[::-1]
            for k, eff in enumerate(turn["effects"]):
                pet = heads[k % 2]
                if pet.dead:
                    continue
                ctx = Ctx(me=pet, target=pet, caster=caster or pet, family=a.ability["family"],
                          ability_id=a.aid, aura=a, row_target=pet)
                run_effects(battle, [eff], ctx)
            continue
        if tgt.dead:
            continue
        ctx = Ctx(me=me, target=tgt, caster=a.caster or me, family=a.ability["family"], ability_id=a.aid,
                  aura=a)
        run_effects(battle, turn["effects"], ctx, a.aid, ti)


def clear_dead_auras(battle, untimed_only=False):
    """Dead pets lose their auras. Before the aura phase only "until next attack" (no duration) ones go:
    log: Attack Reduction faded from a dead Haunt caster at once, while a dead Webspinner's Rabies still
    boosted its Poisoned tick that round."""
    for pet in battle.all_pets():
        if pet.dead and pet.auras:
            if untimed_only:
                pet.auras[:] = [a for a in pet.auras if a.duration >= 0]
            else:
                pet.auras.clear()


def countdown_auras(battle):
    """Durations: an aura does not lose a round in the round it was applied (verified on logs).
    Duration 0 = expires at the end of the application round, -1 = never."""
    items = []
    if battle.weather is not None:
        items.append((None, battle.weather))
    for team in battle.teams:
        items.extend((team, a) for a in list(team.auras))
        for pet in team.pets:
            items.extend((pet, a) for a in list(pet.auras))
    for holder, a in items:
        if a.duration < 0:
            if a.next_attack and not a.fresh and isinstance(holder, Pet) and holder.dealt_damage_round:
                remove_aura(battle, holder, a)
            a.fresh = False
            continue
        if a.fresh:
            a.fresh = False
            if a.duration == 0:
                remove_aura(battle, holder, a)
            continue
        a.duration -= 1
        if a.duration <= 0:
            remove_aura(battle, holder, a)


# ---------------------------------------------------------------------------------------------
# aura management
# ---------------------------------------------------------------------------------------------

def add_aura(battle, holder, aid, caster, duration, max_allowed=0, next_attack=False):
    gd = game_data()
    ab = gd.abilities.get(aid)
    if ab is None or holder is None:
        return None
    lst = holder.auras
    existing = [a for a in lst if a.aid == aid]
    cap = max(1, max_allowed or 1)
    if existing and len(existing) >= cap:
        if cap == 1:
            a = existing[0]
            a.duration = duration
            a.caster = caster
            a.age = 0
            a.fresh = True
            return a
        oldest = existing[0]
        lst.remove(oldest)
    a = Aura(aid=aid, ability=ab, holder=holder if isinstance(holder, Pet) else None, caster=caster,
             duration=duration, uid=battle.next_uid())
    a.age = 0
    a.next_attack = next_attack
    a.team = holder if isinstance(holder, Team) else None
    block = ab["states"].get(C.S_BLOCK_COUNT)
    a.blocks_left = block or 0
    lst.append(a)
    battle.emit(f"{ab['name']} applied to {_name(holder)}")
    pet = holder if isinstance(holder, Pet) else holder.active_pet
    fire(battle, C.EV_APPLY, pet, aura_filter=a)
    return a


def _name(holder):
    if isinstance(holder, Team):
        return f"team {holder.side}"
    return getattr(holder, "name", "?")


def remove_aura(battle, holder, a):
    if holder is None:
        if battle.weather is a:
            battle.weather = None
            battle.emit(f"{a.name} is no longer the weather")
        return
    lst = holder.auras
    if a not in lst:
        return
    lst.remove(a)
    battle.emit(f"{a.name} fades from {_name(holder)}")
    pet = holder if isinstance(holder, Pet) else holder.active_pet
    if not pet.dead:
        # fire the removed aura's own ev5 turn (it is no longer in the list)
        for ti, t in enumerate(a.ability["turns"]):
            if t["type"] == 1 and t["event"] == C.EV_REMOVE:
                ctx = Ctx(me=pet, target=pet, caster=a.caster or pet, family=a.ability["family"],
                          ability_id=a.aid, aura=a, in_trigger=True)
                run_effects(battle, t["effects"], ctx, a.aid, ti)


def set_weather(battle, aid, caster, duration):
    ab = game_data().abilities[aid]
    if battle.weather is not None:
        battle.emit(f"{battle.weather.name} is no longer the weather")
    battle.weather = Aura(aid=aid, ability=ab, holder=None, caster=caster, duration=duration, uid=battle.next_uid())
    battle.weather.age = 0
    battle.weather.next_attack = False
    battle.emit(f"weather changed to {ab['name']}")


# ---------------------------------------------------------------------------------------------
# hit / damage / heal
# ---------------------------------------------------------------------------------------------

def probe_count(battle, kind, pet):
    """Counters for bp.pvp.probe (set only on probe copies): what happened to whom this round."""
    c = getattr(battle, "probe_counts", None)
    if c is not None and pet is not None:
        key = (kind, pet.team.side)
        c[key] = c.get(key, 0) + 1


def roll_hit(battle, ctx: Ctx, target: Pet, accuracy):
    if accuracy is None:
        accuracy = 100
    if target is None or target.dead:
        return False
    enemy = target.team is not ctx.me.team
    if enemy and target.state(C.S_UNTARGETTABLE) > 0 and not ctx.periodic:
        # abilities like Quake / Tympanic Tantrum give the caster the same state for the round
        # (op79 at round start), which lets them hit underground / flying targets
        reach = ((target.state(C.S_UNDERGROUND) > 0 and ctx.caster.state(C.S_UNDERGROUND) > 0)
                 or (target.state(C.S_FLYING) > 0 and ctx.caster.state(C.S_FLYING) > 0))
        if not reach:
            probe_count(battle, "unreachable", target)
            return False
    acc = accuracy + ctx.caster.state(C.S_ACCURACY)
    if enemy:
        acc -= target.state(C.S_DODGE)
        lvl_diff = target.spec.level - ctx.caster.spec.level
        if lvl_diff > 0:
            acc -= lvl_diff * C.LEVEL_HIT_PENALTY
    key = (_src(ctx), target.name)
    if battle.forced_miss and battle.forced_miss.get(key):
        return battle.forced_miss[key].pop(0)  # log replay: reproduce the recorded hit/miss sequence
    if getattr(battle, "probe_expect", False):
        # bp.pvp.probe: every roll lands, the damage of this hit is scaled by its chance instead
        battle._hit_scale = max(0.0, min(1.0, acc / 100))
        if battle._hit_scale < 1:
            battle.probe_counts["rolled", target.team.side] = 1
        return True
    if acc >= 100:
        return True
    if battle.force_hit is not None:
        return battle.force_hit
    return battle.rng.random() * 100 < acc


def family_mult(att_family, def_family):
    if att_family < 0:
        return 1.0
    if C.STRONG[att_family] == def_family:
        return C.STRONG_MULT
    if C.WEAK[att_family] == def_family:
        return C.WEAK_MULT
    return 1.0


def _vary(battle, x, variance):
    if not variance:
        return x
    half = x * variance / 200
    if battle.force_variance is not None:
        return x + half * battle.force_variance
    r = max(battle.rng.random() for _ in range(3))  # best of 3 rolls, then +/- with 50% (blue post)
    return x + half * r if battle.rng.random() < 0.5 else x - half * r


def calc_damage(battle, ctx: Ctx, target: Pet, points, variance=0, can_crit=True, raw=None):
    """raw: already-scaled amount (percent-of-health damage), skips power scaling."""
    caster = ctx.caster
    dmg = raw if raw is not None else points * (1 + caster.power / 20)
    fam = ctx.family if ctx.family is not None else -1
    target_fam = target.state(C.S_TYPE_OVERRIDE) - 1 if target.state(C.S_TYPE_OVERRIDE) > 0 else target.family
    dealt = caster.state(C.S_DMG_DEALT_PCT)
    taken = target.state(C.S_DMG_TAKEN_PCT)
    pt_id = caster.state(C.S_PETTYPE_ID)
    if pt_id and fam >= 0 and pt_id - 1 == fam:
        dealt += caster.state(C.S_PETTYPE_DMG_DEALT)
    if battle.forced_amount and variance:
        q = battle.forced_amount.get((_src(ctx), target.name))
        if q:
            battle._forced_pending = (q, dmg, variance)
    dmg = _vary(battle, dmg, variance)
    if raw is None:
        # percent-of-health damage ignores type and damage modifiers
        # (log: Glowing Toxin 6% = 122 through Boss -50%, Toxic Skin 5% through Stoneskin)
        dmg *= family_mult(fam, target_fam)
        dmg *= (100 + dealt) / 100
        dmg *= max(1 - C.MAX_DMG_REDUCTION, (100 + taken) / 100)  # reductions cap at 75% (tooltips)
    if ctx.periodic:
        if target.family == C.AQUATIC:
            dmg *= C.AQUATIC_DOT_TAKEN
        if caster.family == C.AQUATIC:
            dmg *= C.AQUATIC_DOT_DEALT
        # log: Toxic Wound adds 7 x AttackBonus of its caster to each DoT tick (71 + 100 = 171)
        dmg += scaled_flat(battle, target, C.S_ADD_PERIODIC_DMG_TAKEN)
    # flat modifiers scale with the power of whoever applied the aura (Sandstorm: -5 * AttackBonus).
    # Percent-of-health damage ignores them (log: Toxic Skin 5% = 79 through Stoneskin).
    if raw is None:
        dmg += scaled_flat(battle, target, C.S_FLAT_DMG_TAKEN) + scaled_flat(battle, caster, C.S_FLAT_DMG_DEALT)
    dmg = max(0, int(dmg))
    crit = False
    if can_crit and raw is None and dmg > 0:  # DoTs and weather ticks can crit too (log-verified)
        key = (_src(ctx), target.name)
        if battle.forced_crit and battle.forced_crit.get(key):
            roll = battle.forced_crit[key].pop(0)  # log replay: crit pattern of the recorded hits
        else:
            roll = battle.force_crit if battle.force_crit is not None else battle.rng.random() * 100 < C.BASE_CRIT + caster.state(C.S_CRIT)
        if roll:
            crit = True
            dmg = int(dmg * (C.CRIT_MULT + caster.state(C.S_CRIT_DMG_BONUS) / 100))  # floor(floor(x)*1.5), log-verified
    if target.family == C.MAGIC or target.has_aura(C.BOSS_AURA):
        dmg = min(dmg, int(target.max_hp * C.MAGIC_CAP + 0.5))
    pending = getattr(battle, "_forced_pending", None)
    if pending:
        battle._forced_pending = None
        q, mid, var = pending
        want = q[0]
        spread = dmg * var / 200 + 1.5
        if abs(want - dmg) <= spread:  # plausible roll: reproduce it exactly
            q.pop(0)
            dmg = want
    return dmg, crit


def scaled_flat(battle, pet, sid):
    total = 0.0
    sources = list(pet.auras)
    if pet.team.active_pet is pet:
        sources += pet.team.auras
    if battle.weather is not None and not (pet.family == C.ELEMENTAL and sid == C.S_FLAT_DMG_DEALT):
        sources.append(battle.weather)
    for a in sources:
        v = a.ability["states"].get(sid)
        if v:
            c = a.caster or pet
            total += v * (1 + c.power / 20) * a.stacks
    v = pet.states.get(sid, 0)
    return total + v


def deal_damage(battle, ctx: Ctx, target: Pet, amount, crit=False, periodic=False, source="", nonlethal=False):
    if target is None or target.dead or amount <= 0:
        return 0
    enemy_hit = target.team is not ctx.caster.team
    if not periodic and enemy_hit:
        for a in list(ctx.caster.team.auras):
            if a.ability["states"].get(C.S_BLOCKED_FRIENDLY) and a.blocks_left > 0:
                a.blocks_left -= 1
                battle.emit(f"{source} was blocked by {a.name}")
                probe_count(battle, "blocked", target)
                if a.blocks_left <= 0:
                    remove_aura(battle, ctx.caster.team, a)
                return 0
        # block auras on the pet or its team pad (Shield Block, Beaver Dam, Silk Cocoon)
        for holder in (target, target.team):
            for a in list(holder.auras):
                if a.blocks_left > 0:
                    a.blocks_left -= 1
                    battle.emit(f"{source} was blocked by {a.name}")
                    probe_count(battle, "blocked", target)
                    if a.blocks_left <= 0:
                        remove_aura(battle, holder, a)
                    ctx.blocked = True  # log: a blocked Rake does not apply its Attack Reduction
                    return 0
    low = target.state(C.S_IGNORE_DMG_BELOW)
    if low and amount < low:
        return 0
    full = amount  # log: Soul Steal on an immortal Undead dealt 15 but healed from the full 208
    if nonlethal or target.state(C.S_UNKILLABLE) > 0:
        # unkillable (Undead's immortal round, Survival): the log shows only what was taken (17, then 0)
        amount = min(amount, target.hp - 1) if target.hp > 1 else 0
        if amount <= 0 and nonlethal:
            return 0
    before = target.hp
    target.hp -= amount
    battle.emit(f"{source} dealt {amount} damage to {target.name}{' (Critical)' if crit else ''}")
    target.states[C.S_WAS_DAMAGED] = 1
    target.states[C.S_LAST_HIT_TAKEN] = full
    ctx.caster.states[C.S_DID_DAMAGE] = 1
    ctx.caster.states[C.S_LAST_HIT_DEALT] = full
    if enemy_hit:
        ctx.caster.dealt_damage_round = True
        if ctx.caster.family == C.DRAGONKIN and before * 2 >= target.max_hp > target.hp * 2:
            ctx.caster.dragon_trigger = True
    if target.hp <= 0:
        handle_death(battle, target, ctx.caster)
    if not periodic:
        # also fires on the killing blow (log: Heat Up / Toxic Skin hit back after their holder died)
        fire(battle, C.EV_DAMAGE_TAKEN, target, other=ctx.caster if ctx.caster is not target else None)
    if battle.over:
        return amount
    if enemy_hit and not ctx.in_trigger:
        # log: Lightning Storm procs on a dead Flicker's Glowing Toxin tick, but its own bonus hit
        # does not proc it again (one bonus per attack)
        fire(battle, C.EV_DAMAGE_DEALT, ctx.caster, other=target)
    return amount


def run_passive_ability(battle, pet, aid):
    ab = game_data().abilities[aid]
    for ti, t in enumerate(ab["turns"]):
        if t["type"] == 0:
            ctx = Ctx(me=pet, target=pet, caster=pet, family=-1, ability_id=aid)
            run_effects(battle, t["effects"], ctx, aid, ti)


def handle_death(battle, pet: Pet, killer: Pet | None):
    if pet.state(C.S_UNKILLABLE) > 0:
        pet.hp = max(pet.hp, 1)
        return
    if pet.family == C.UNDEAD and not pet.undead_used and pet.state(C.S_TYPE_OVERRIDE) == 0:
        pet.undead_used = True
        pet.hp = 1
        battle.emit(f"{pet.name} returns to life (Undead)")
        run_passive_ability(battle, pet, C.DAMNED_ABILITY)  # remove polymorph, 1% hp, aura 242 (1 round)
        return
    if pet.family == C.MECHANICAL and not pet.mech_revived and pet.state(C.S_TYPE_OVERRIDE) == 0:
        pet.mech_revived = True
        pet.hp = 1
        battle.emit(f"{pet.name} comes back to life (Failsafe)")
        run_passive_ability(battle, pet, C.FAILSAFE_ABILITY)  # 20% hp, marker aura 244
        return
    kill(battle, pet)


def kill(battle, pet: Pet):
    if pet.dead:
        return
    pet.hp = 0
    pet.dead = True
    pet.channel = None
    battle.emit(f"{pet.name} dies")
    battle.check_end()  # its auras are cleared at the end of the round (on-hit triggers still fire)


def heal(battle, caster: Pet, target: Pet, amount, source=""):
    if target is None or target.dead or amount <= 0:
        return 0
    amount = int(amount)
    new = min(target.max_hp, target.hp + amount)
    got = new - target.hp
    target.hp = new
    battle.emit(f"{source} healed {got} damage from {target.name}")
    return got


def calc_heal(battle, ctx: Ctx, target: Pet, points, variance=0):
    h = points * (1 + ctx.caster.power / 20)
    base = h
    h = _vary(battle, h, variance)
    q = battle.forced_heal.get((_src(ctx), target.name)) if battle.forced_heal and variance else None
    h *= (100 + ctx.caster.state(C.S_HEAL_DEALT_PCT)) / 100
    h *= max(0.0, (100 + target.state(C.S_HEAL_TAKEN_PCT)) / 100)
    if q and abs(q[0] - h) <= h * variance / 200 + 1.5:  # log replay: reproduce the recorded roll
        return q.pop(0)
    return int(h)


def _src(ctx):
    return game_data().abilities.get(ctx.ability_id, {}).get("name", "?")


def _state_ok(pet, sid):
    return not sid or (pet is not None and pet.state(sid) > 0)


def _periodic(ctx, p):
    return bool(p.get("IsPeriodic")) or ctx.periodic


def _hit_and_damage(battle, ctx, p, points=None, raw=None, variance=None, nonlethal=False, autohit=False):
    tgt = _tgt(ctx)
    if tgt is None or tgt.dead:
        return False
    periodic = _periodic(ctx, p)
    sub = Ctx(**{**ctx.__dict__, "periodic": periodic})
    battle.__dict__.pop("_hit_scale", None)  # probe mode: only this effect's own roll scales it
    if not autohit and not roll_hit(battle, sub, tgt, p.get("Accuracy", 100)):
        battle.emit(f"{_src(ctx)} missed {tgt.name}")
        return False
    dmg, crit = calc_damage(battle, sub, tgt, p.get("Points", 0) if points is None else points,
                            p.get("Variance", 0) if variance is None else variance, raw=raw)
    scale = battle.__dict__.pop("_hit_scale", None)
    if scale is not None and scale < 1:
        dmg = int(dmg * scale)
    deal_damage(battle, sub, tgt, dmg, crit, periodic, _src(ctx), nonlethal=nonlethal)
    if sub.blocked:
        return False  # counts as failed for ChainFailure followers (Rake's Attack Reduction), other hits go on
    return True


# ---------------------------------------------------------------------------------------------
# damage opcodes
# ---------------------------------------------------------------------------------------------

@op(24, 222, 179)
def op_damage(battle, e, ctx):
    return _hit_and_damage(battle, ctx, e["params"])


@op(169)
def op_damage_state_req(battle, e, ctx):
    """Scorched Earth weather tick: RequiredTargetState 47 (Passive_Elemental) means elementals are
    exempt (log: Fel Flame not hit, everyone else takes the tick)."""
    p = e["params"]
    ts = p.get("RequiredTargetState")
    tgt = _tgt(ctx)
    if ts in C.PASSIVE_STATES:
        if tgt is not None and (tgt.family == C.PASSIVE_STATES.index(ts) or tgt.state(ts) > 0):
            return None
        return _hit_and_damage(battle, ctx, p)
    return op_damage_if_state(battle, e, ctx)


@op(29)
def op_damage_if_state(battle, e, ctx):
    p = e["params"]
    if not _state_ok(ctx.me, p.get("RequiredCasterState")):
        return None
    if not _state_ok(_tgt(ctx), p.get("RequiredTargetState")):
        return None
    return _hit_and_damage(battle, ctx, p)


@op(96)
def op_damage_autohit_if_state(battle, e, ctx):
    p = e["params"]
    auto = (p.get("CasterState") and ctx.me.state(p["CasterState"]) > 0) or \
           (p.get("TargetState") and _tgt(ctx) is not None and _tgt(ctx).state(p["TargetState"]) > 0)
    return _hit_and_damage(battle, ctx, p, autohit=bool(auto))


@op(384)
def op_damage_if_first(battle, e, ctx):
    if not ctx.first:
        return None
    return _hit_and_damage(battle, ctx, e["params"])


@op(385)
def op_damage_if_last(battle, e, ctx):
    if ctx.first:
        return None
    return _hit_and_damage(battle, ctx, e["params"])


@op(141)
def op_damage_split(battle, e, ctx):
    p = e["params"]
    pts = p.get("Points", 0)
    if p.get("BonusState") and ctx.me.state(p["BonusState"]) > 0:
        pts += p.get("BonusPoints", 0)
    enemies = [x for x in ctx.me.team.enemy.pets if not x.dead]
    if not enemies:
        return False
    # the whole split is computed at once: an "until the next attack" debuff that fades on the
    # first part still weakens the others (log: Carpnado 63 + 63 under Attack Reduction)
    hits = []
    for x in enemies:
        sub = Ctx(**{**ctx.__dict__, "row_target": x})
        if roll_hit(battle, sub, x, p.get("Accuracy", 100)):
            dmg, crit = calc_damage(battle, sub, x, pts / len(enemies))
            hits.append((sub, x, dmg, crit))
        else:
            battle.emit(f"{_src(ctx)} missed {x.name}")
    for sub, x, dmg, crit in hits:
        deal_damage(battle, sub, x, dmg, crit, False, _src(ctx))
    return bool(hits)


@op(226, 65)
def op_damage_bonus_state(battle, e, ctx):
    p = e["params"]
    pts = p.get("Points", 0)
    bs = p.get("BonusState")
    tgt = _tgt(ctx)
    weather = bs in C.WEATHER_STATES
    if bs and ((tgt is not None and tgt.state(bs) > 0) or (weather and ctx.me.state(bs) > 0)):
        pts += p.get("BonusPoints", 0)  # log: Sludge Claw doubles only if the *target* is poisoned
    return _hit_and_damage(battle, ctx, p, points=pts)


@op(170)
def op_damage_if_weather(battle, e, ctx):
    p = e["params"]
    ws = p.get("weatherState")
    if ws and ctx.me.state(ws) <= 0:
        return None
    return _hit_and_damage(battle, ctx, p)


@op(27, 358)
def op_ramping_damage(battle, e, ctx):
    p = e["params"]
    me = ctx.me
    if me.states.get(C.S_RAMPING_ID) != ctx.ability_id:
        me.states[C.S_RAMPING_ID] = ctx.ability_id
        me.states[C.S_RAMPING_USES] = 0
    uses = me.states.get(C.S_RAMPING_USES, 0)
    cap = p.get("PointsMax") or 10 ** 6
    pts = min(p.get("Points", 0) + uses * p.get("PointsIncreasePerUse", 0), cap)
    me.states[C.S_RAMPING_USES] = uses + 1
    return _hit_and_damage(battle, ctx, p, points=pts)


@op(62)
def op_damage_pct_max_hp(battle, e, ctx):
    p = e["params"]
    tgt = _tgt(ctx)
    if tgt is None:
        return False
    return _hit_and_damage(battle, ctx, p, raw=tgt.max_hp * p.get("Points", 0) / 100)


@op(68)
def op_damage_pct_caster_max_hp(battle, e, ctx):
    p = e["params"]
    return _hit_and_damage(battle, ctx, p, raw=ctx.caster.max_hp * p.get("Points", 0) / 100)


@op(363)
def op_damage_pct_current_hp(battle, e, ctx):
    p = e["params"]
    tgt = _tgt(ctx)
    if tgt is None:
        return False
    return _hit_and_damage(battle, ctx, p, raw=tgt.hp * p.get("Points", 0) / 100)


@op(197)
def op_damage_pct_last_taken(battle, e, ctx):
    p = e["params"]
    last = ctx.caster.states.get(C.S_LAST_HIT_TAKEN, 0)
    return _hit_and_damage(battle, ctx, p, raw=last * p.get("Power", 0) / 100)


@op(204)
def op_reflect(battle, e, ctx):
    p = e["params"]
    last = ctx.me.states.get(C.S_LAST_HIT_TAKEN, 0)
    return _hit_and_damage(battle, ctx, p, raw=last * p.get("Points", 0) / 100)


@op(149)
def op_damage_nonlethal(battle, e, ctx):
    return _hit_and_damage(battle, ctx, e["params"], nonlethal=True)


@op(59)
def op_damage_if_lower_hp(battle, e, ctx):
    tgt = _tgt(ctx)
    if tgt is None or ctx.me.hp >= tgt.hp:
        return None
    return _hit_and_damage(battle, ctx, e["params"])


@op(66, 45, 97)
def op_damage_boost(battle, e, ctx):
    p = e["params"]
    tgt = _tgt(ctx)
    if tgt is None:
        return False
    boost = p.get("Boost", 0)
    cond = {66: tgt.hp * 4 < tgt.max_hp, 45: ctx.me.speed < tgt.speed, 97: ctx.me.speed > tgt.speed}[e["op"]]
    pts = p.get("Points", 0) * ((100 + boost) / 100 if cond else 1)
    return _hit_and_damage(battle, ctx, p, points=pts)


@op(370)
def op_damage_same_species(battle, e, ctx):
    p = e["params"]
    same = sum(1 for x in ctx.me.team.pets if x is not ctx.me and x.spec.species == ctx.me.spec.species)
    return _hit_and_damage(battle, ctx, p, points=p.get("Points", 0) + same * p.get("BonusDamage", 0))


@op(86)
def op_damage_repeat_aura(battle, e, ctx):
    p = e["params"]
    tgt = _tgt(ctx)
    ok = _hit_and_damage(battle, ctx, p)
    if ok and tgt is not None and p.get("AuraID") and tgt.has_aura(e["aura"]):
        add_aura(battle, tgt, p["AuraID"], ctx.caster, p.get("Duration2", 1))
    return ok


# ---------------------------------------------------------------------------------------------
# heals
# ---------------------------------------------------------------------------------------------

def _heal_target(ctx):
    t = ctx.row_target
    if isinstance(t, Team):
        return t.active_pet
    return t if t is not None else ctx.me


@op(23)
def op_heal(battle, e, ctx):
    p = e["params"]
    tgt = _heal_target(ctx)
    if tgt is None or tgt.dead:
        return False
    heal(battle, ctx.caster, tgt, calc_heal(battle, ctx, tgt, p.get("Points", 0), p.get("Variance", 0)), _src(ctx))
    return True


@op(104, 171)
def op_heal_if_state(battle, e, ctx):
    p = e["params"]
    if not _state_ok(ctx.me, p.get("RequiredCasterState")):
        return None
    if p.get("RequiredTargetState") and not (ctx.target is not None and ctx.target.state(p["RequiredTargetState"]) > 0
                                             or p["RequiredTargetState"] == C.S_IS_DEAD and ctx.target is not None and ctx.target.dead):
        return None
    return op_heal(battle, e, ctx)


@op(61)
def op_heal_pct_if_state(battle, e, ctx):
    p = e["params"]
    if not _state_ok(ctx.me, p.get("RequiredCasterState")):
        return None
    tgt = _heal_target(ctx)
    # despite the tooltip ("% of total health") the log shows a normal power-scaled heal:
    # Consume Corpse 50 points = 625 at 230 power, 312 in Darkness, 225 at 160 power in Darkness (2026-09-25)
    heal(battle, ctx.caster, tgt, calc_heal(battle, ctx, tgt, p.get("Points", 0)), _src(ctx))
    return True


@op(100)
def op_heal_if_type(battle, e, ctx):
    p = e["params"]
    tgt = _heal_target(ctx)
    if p.get("RequiredTargetPetType") and tgt.family + 1 != p["RequiredTargetPetType"]:
        return None
    if p.get("RequiredCasterPetType") and ctx.me.family + 1 != p["RequiredCasterPetType"]:
        return None
    return op_heal(battle, e, ctx)


@op(254)
def op_heal_low_hp_bonus(battle, e, ctx):
    p = e["params"]
    tgt = _heal_target(ctx)
    amt = calc_heal(battle, ctx, tgt, p.get("Points", 0), p.get("Variance", 0))
    if tgt.hp * 100 < tgt.max_hp * p.get("Threshold", 0):
        amt = amt * (100 + p.get("Modifier", 0)) / 100
    heal(battle, ctx.caster, tgt, amt, _src(ctx))
    return True


@op(53)
def op_heal_pct(battle, e, ctx):
    p = e["params"]
    tgt = _heal_target(ctx)
    # heal modifiers apply (log 2026-09-25: Humanoid Recovery 4% healed 34 instead of 68 in Darkness)
    heal(battle, ctx.caster, tgt, _heal_mods(ctx, tgt, tgt.max_hp * p.get("Percentage", 0) / 100), _src(ctx))
    return True


def _heal_mods(ctx, target, amount):
    """Healing dealt/taken modifiers (log: Darkness halves Soul Steal / Consume heals)."""
    amount *= (100 + ctx.me.state(C.S_HEAL_DEALT_PCT)) / 100
    amount *= max(0.0, (100 + target.state(C.S_HEAL_TAKEN_PCT)) / 100)
    return int(amount)


@op(32)
def op_heal_pct_dealt(battle, e, ctx):
    p = e["params"]
    last = ctx.me.states.get(C.S_LAST_HIT_DEALT, 0)
    heal(battle, ctx.me, ctx.me, _heal_mods(ctx, ctx.me, last * p.get("Points", 0) / 100), _src(ctx))
    return True


@op(44)
def op_heal_pct_taken(battle, e, ctx):
    p = e["params"]
    last = ctx.me.states.get(C.S_LAST_HIT_TAKEN, 0)
    heal(battle, ctx.me, ctx.me, _heal_mods(ctx, ctx.me, last * p.get("Points", 0) / 100), _src(ctx))
    return True


@op(67)
def op_equalize(battle, e, ctx):
    tgt = _tgt(ctx)
    if tgt is None:
        return False
    avg = (ctx.me.hp + tgt.hp) // 2
    new_me, new_t = min(ctx.me.max_hp, avg), min(tgt.max_hp, avg)
    for pet, new in ((tgt, new_t), (ctx.me, new_me)):
        d = new - pet.hp
        if d < 0:
            battle.emit(f"{_src(ctx)} dealt {-d} damage to {pet.name}")
        elif d > 0:
            battle.emit(f"{_src(ctx)} healed {d} damage from {pet.name}")
        pet.hp = new
    return True


@op(111)
def op_resurrect_self(battle, e, ctx):
    me = ctx.me
    me.dead = False
    me.hp = max(1, int(me.max_hp * e["params"].get("Percentage", 0) / 100 + 0.5))  # log: 1% of 1587 -> 16
    battle.emit(f"{_src(ctx)} healed {me.hp} damage from {me.name}")  # the game logs revives as heals
    return True


@op(112)
def op_resurrect_allies(battle, e, ctx):
    pct = e["params"].get("Percentage", 0)
    for x in ctx.me.team.pets:
        if x.dead and x is not ctx.me:
            x.dead = False
            x.hp = max(1, int(x.max_hp * pct / 100))
    return True


@op(128)
def op_set_health_pct(battle, e, ctx):
    tgt = _tgt(ctx, default_enemy=True)
    if tgt is None or tgt.dead:
        return False
    tgt.hp = max(1, int(tgt.max_hp * e["params"].get("Percentage", 0) / 100))
    return True


# ---------------------------------------------------------------------------------------------
# auras
# ---------------------------------------------------------------------------------------------

def _is_object(aid):
    st = game_data().abilities.get(aid, {}).get("states", {})
    return st.get(C.S_OBJECT) or st.get(C.S_BOMB)


def _duration(p, op_code):
    d = p.get("Duration", 0)
    if op_code in (28, 150) and d == 0:
        return -1  # until consumed / removed
    return d


def _immune(target, aura_id):
    st = game_data().abilities.get(aura_id, {}).get("states", {})
    cc = st.get(C.S_STUNNED) or st.get(C.S_TURN_LOCK) or st.get(C.S_SWAP_OUT_LOCK)
    if cc and target.family == C.CRITTER:
        return True
    if (st.get(C.S_STUNNED) or st.get(C.S_TURN_LOCK)) and target.state(C.S_RESILIENT) > 0:
        return True
    return False


@op(26, 28, 63, 150)
def op_aura_self(battle, e, ctx):
    p = e["params"]
    holder = ctx.row_target if ctx.row_target is not None else ctx.me
    if isinstance(holder, Pet) and _is_object(e["aura"]) and e["op"] in (150,):
        holder = holder.team
    if isinstance(holder, Pet) and holder.dead:
        return False
    add_aura(battle, holder, e["aura"], ctx.caster, _duration(p, e["op"]))
    return True


@op(248)
def op_aura_self_if_state(battle, e, ctx):
    if not _state_ok(ctx.me, e["params"].get("RequiredCasterState")):
        return None
    return op_aura_self(battle, e, ctx)


@op(137)
def op_aura_if_state(battle, e, ctx):
    p = e["params"]
    if p.get("CasterState") and ctx.me.state(p["CasterState"]) != p.get("CasterStateValue", 1):
        return None
    if p.get("TargetState") and (ctx.target is None or ctx.target.state(p["TargetState"]) != p.get("TargetStateValue", 1)):
        return None
    return op_aura_self(battle, e, ctx)


@op(223)
def op_aura_allies_of_type(battle, e, ctx):
    p = e["params"]
    for x in ctx.me.team.pets:
        if not x.dead and x.family + 1 == p.get("RequiredPetType"):
            add_aura(battle, x, e["aura"], ctx.caster, _duration(p, 26))
    return True


def _aura_target(battle, e, ctx, check_states=True, chance=None):
    p = e["params"]
    holder = ctx.row_target
    if holder is None:
        holder = ctx.target
        if _is_object(e["aura"]) and holder is not None:
            holder = holder.team
    pet = holder.active_pet if isinstance(holder, Team) else holder
    if pet is None or pet.dead:
        return False
    if check_states:
        if p.get("CasterState") and ctx.me.state(p["CasterState"]) <= 0:
            return None
        if p.get("TargetState") and pet.state(p["TargetState"]) <= 0:
            return None
    acc = p.get("Accuracy", 100) if chance is None else chance
    if isinstance(holder, Pet) and holder.team is not ctx.me.team:
        if not roll_hit(battle, ctx, pet, acc):
            battle.emit(f"{_src(ctx)} missed {pet.name}")
            return False
        if _immune(pet, e["aura"]):
            battle.emit(f"{pet.name} is immune")
            probe_count(battle, "immune", pet)
            return False
    elif acc < 100 and battle.rng.random() * 100 >= acc:
        return False
    add_aura(battle, holder, e["aura"], ctx.caster, _duration(p, e["op"]), max_allowed=p.get("MaxAllowed", 0),
             next_attack=p.get("Duration", 0) == -1)
    return True


@op(50, 52, 54, 168, 229)
def op_aura_target(battle, e, ctx):
    return _aura_target(battle, e, ctx)


@op(230)
def op_aura_haunt(battle, e, ctx):
    return _aura_target(battle, e, ctx, check_states=False)


@op(131)
def op_aura_if_first(battle, e, ctx):
    if not ctx.first:
        return None
    return _aura_target(battle, e, ctx)


@op(172)
def op_aura_chance_or_state(battle, e, ctx):
    p = e["params"]
    sure = (p.get("CasterState") and ctx.me.state(p["CasterState"]) > 0) or \
           (p.get("TargetState") and ctx.target is not None and ctx.target.state(p["TargetState"]) > 0)
    return _aura_target(battle, e, ctx, check_states=False, chance=200 if sure else p.get("Accuracy", 0))


@op(178)
def op_stun(battle, e, ctx):
    p = e["params"]
    tgt = _tgt(ctx)
    if tgt is None or tgt.dead:
        return False
    ts = p.get("TargetState")
    if not roll_hit(battle, ctx, tgt, p.get("Accuracy", 100)):
        return False
    if (ts and tgt.state(ts) == p.get("StatePoints", 1)) or _immune(tgt, e["aura"]):
        battle.emit(f"{tgt.name} is immune")
        probe_count(battle, "immune", tgt)
        return False
    add_aura(battle, tgt, e["aura"] or C.STUNNED_AURA, ctx.caster, p.get("Duration", 1))
    return True


@op(76, 77, 75)
def op_two_stage(battle, e, ctx):
    """1st use: marker aura (on caster / target / caster pad); 2nd use: consume it and deal damage."""
    p = e["params"]
    holder = {76: ctx.me, 77: _tgt(ctx), 75: ctx.me.team}[e["op"]]
    if holder is None:
        return False
    a = next((x for x in holder.auras if x.aid == e["aura"]), None)
    if a is None:
        add_aura(battle, holder, e["aura"], ctx.caster, p.get("Duration", -1) or -1)
        return True
    # the charge's own bonus still applies to the release (log: Pump 972 * 1.1 = 1069)
    ok = _hit_and_damage(battle, ctx, p)
    if a in holder.auras:
        remove_aura(battle, holder, a)
    return ok


@op(140)
def op_refresh_duration(battle, e, ctx):
    tgt = _tgt(ctx)
    if tgt is None:
        return None
    for a in tgt.auras:
        if a.aid == e["aura"]:
            a.duration = e["params"].get("NewDuration", a.duration)
    return None


@op(49)
def op_remove_aura(battle, e, ctx):
    ch = e["params"].get("Chance", 0)
    if 0 < ch < 100 and battle.rng.random() * 100 >= ch:
        return None
    for holder in (ctx.me, ctx.me.team):
        a = next((x for x in holder.auras if x.aid == e["aura"]), None)
        if a is not None:
            remove_aura(battle, holder, a)
            break
    return None


@op(85)
def op_remove_auras_with_state(battle, e, ctx):
    sid = e["params"].get("State")
    tgt = ctx.row_target or ctx.target
    holders = [tgt] if isinstance(tgt, Team) else ([tgt, tgt.team] if tgt is not None else [])
    for h in holders:
        for a in list(h.auras):
            if a.ability["states"].get(sid):
                remove_aura(battle, h, a)
    return None


@op(136)
def op_dispel_all(battle, e, ctx):
    tgt = ctx.row_target or ctx.target
    if isinstance(tgt, Team):
        tgt = tgt.active_pet
    if tgt is None:
        return None
    for a in list(tgt.auras):
        if a.aid not in C.PASSIVE_ABILITY:
            remove_aura(battle, tgt, a)
    return True


# ---------------------------------------------------------------------------------------------
# states and gates
# ---------------------------------------------------------------------------------------------

@op(31)
def op_set_state(battle, e, ctx):
    p = e["params"]
    ctx.me.states[p["State"]] = p.get("StateValue", 0)
    return None


@op(360)
def op_clear_state(battle, e, ctx):
    p = e["params"]
    ctx.me.states[p.get("State", 0)] = p.get("StateValue", 0)
    return None


@op(79)
def op_mod_state(battle, e, ctx):
    p = e["params"]
    s = p["State"]
    v = ctx.me.states.get(s, 0) + p.get("StateChange", 0)
    lo, hi = p.get("StateMin", 0), p.get("StateMax", 0)
    if hi:
        v = min(hi, v)
    if lo:
        v = max(lo, v)
    ctx.me.states[s] = v
    return None


@op(138)
def op_mod_state_if_target_state(battle, e, ctx):
    p = e["params"]
    tgt = ctx.target
    if tgt is not None and tgt.state(p.get("TargetTestState", 0)) >= max(1, p.get("TargetTestStateValue", 1)):
        return op_mod_state(battle, e, ctx)
    return None


def _writes_state(aid, sid):
    ab = game_data().abilities.get(aid)
    return bool(ab) and any(e["op"] in (31, 79, 360) and e["params"].get("State") == sid
                            for t in ab["turns"] for e in t["effects"])


def _gate_pet(ctx, sid=None):
    """Which pet a state gate looks at: the aura holder inside aura triggers; in abilities the target,
    unless the ability itself writes that state on its user (Leech Life's 'effect succeeded' flag)."""
    if ctx.aura is not None or (sid is not None and _writes_state(ctx.ability_id, sid)):
        return ctx.me
    return _tgt(ctx) or ctx.target


@op(157)
def op_require_state(battle, e, ctx):
    p = e["params"]
    pet = _gate_pet(ctx, p["State"])
    val = pet.state(p["State"]) if pet is not None else 0
    if p["State"] == C.S_IS_DEAD and pet is not None:
        val = int(pet.dead)
    if val != p.get("StatePoints", 0):
        ctx.stop = True
    return None


@op(156)
def op_stop_if_state(battle, e, ctx):
    pet = _gate_pet(ctx, e["params"].get("State"))
    if pet is not None and pet.state(e["params"].get("State", 0)) > 0:
        ctx.stop = True
    return None


@op(177)
def op_immune_if_state(battle, e, ctx):
    tgt = _tgt(ctx)
    if tgt is not None and tgt.state(e["params"].get("State", C.S_RESILIENT)) > 0:
        probe_count(battle, "immune", tgt)
        return False
    return True


@op(158)
def op_stop_if_prev_failed(battle, e, ctx):
    if ctx.failed:
        ctx.stop = True
    return None


@op(194)
def op_continue_if_prev_missed(battle, e, ctx):
    if not ctx.missed:
        ctx.stop = True
    return None


@op(159)
def op_reset_chain(battle, e, ctx):
    ctx.failed = False
    ctx.missed = False
    return None


# ---------------------------------------------------------------------------------------------
# weather, kill, swaps, misc
# ---------------------------------------------------------------------------------------------

@op(80)
def op_weather(battle, e, ctx):
    p = e["params"]
    set_weather(battle, e["aura"], ctx.caster, p.get("Duration", 0))
    return True


@op(135)
def op_kill(battle, e, ctx):
    p = e["params"]
    victim = ctx.me
    if p.get("CasterImmuneState") and victim.state(p["CasterImmuneState"]) > 0:
        return None
    if p.get("TargetImmuneState") and victim.state(p["TargetImmuneState"]) > 0:
        return None
    if not victim.dead:
        if victim.hp > 0:
            battle.emit(f"{_src(ctx)} dealt {victim.hp} damage to {victim.name}")  # logged as the last hp
        if p.get("enableReverse"):
            victim.states[C.S_HEALTH_BEFORE_INSTAKILL] = victim.hp  # Haunt: revived when the aura ends
        kill(battle, victim)
    return True


@op(165)
def op_revive_aura_caster(battle, e, ctx):
    c = ctx.caster
    if c is not None and c.dead and c.states.get(C.S_HEALTH_BEFORE_INSTAKILL):
        c.dead = False
        c.hp = max(1, (c.states.pop(C.S_HEALTH_BEFORE_INSTAKILL) + 1) // 2)  # log: 603 -> revived at 302
        battle.emit(f"{_src(ctx)} healed {c.hp} damage from {c.name}")
    return True


def _swap_side(ctx):
    t = ctx.row_target
    if isinstance(t, Pet):
        return t.team
    if isinstance(t, Team):
        return t
    return ctx.me.team if ctx.in_trigger else ctx.me.team.enemy


@op(145, 139, 107)
def op_swap(battle, e, ctx):
    team = _swap_side(ctx)
    cands = [i for i, x in enumerate(team.pets) if i != team.active and not x.dead]
    if not cands:
        return False
    if team is not ctx.me.team and not roll_hit(battle, ctx, team.active_pet, e["params"].get("Accuracy", 100) or 100):
        return False
    if e["op"] == 145:
        idx = max(cands, key=lambda i: team.pets[i].hp)
    elif e["op"] == 139:
        idx = min(cands, key=lambda i: team.pets[i].hp)
    else:
        idx = battle.rng.choice(cands)
    battle.swap(team.side, idx, forced=True)
    return True


@op(246)
def op_cooldown_modify(battle, e, ctx):
    p = e["params"]
    slot = p.get("Index", 0)
    if 0 <= slot < 3:
        ctx.me.cooldowns[slot] = max(0, ctx.me.cooldowns[slot] + p.get("CooldownModification", 0)) if p.get("CooldownModification") else 0
    return None


@op(129)
def op_lock_used_ability(battle, e, ctx):
    me = ctx.me
    aid = getattr(me, "last_ability", None)
    if aid in me.abilities:
        i = me.abilities.index(aid)
        me.cooldowns[i] = max(me.cooldowns[i], e["params"].get("LockDuration", 1) + 1)
    return None


@op(117)
def op_lock_slot(battle, e, ctx):
    tgt = _tgt(ctx)
    p = e["params"]
    if tgt is not None:
        i = p.get("Index", 0)
        if 0 <= i < 3:
            tgt.cooldowns[i] = max(tgt.cooldowns[i], p.get("LockDuration", 1) + 1)
    return None


@op(373, 249)
def op_replace_ability(battle, e, ctx):
    p = e["params"]
    slot = p.get("Slot", 0)
    if 0 <= slot < 3 and p.get("Ability"):
        ctx.me.abilities[slot] = p["Ability"]
    return None


@op(78)
def op_plant(battle, e, ctx):
    p = e["params"]
    n = min(ctx.me.states.get(p.get("State", C.S_PLANT), 0), p.get("MaxPoints", 10) or 10)
    if n:
        heal(battle, ctx.me, ctx.me, calc_heal(battle, ctx, ctx.me, p.get("Points", 0) * n), _src(ctx))
    return True


@op(144)
def op_select_corpse(battle, e, ctx):
    """Consume Corpse: take a dead ally's corpse (each corpse only once); none -> the ability does nothing."""
    corpse = _free_corpse(ctx.me)
    if corpse is None:
        ctx.stop = True
        return False
    corpse.states[C.S_CONSUMED_CORPSE] = 1
    return True


@op(116, 22, 0, 256)
def op_nop(battle, e, ctx):
    return None
