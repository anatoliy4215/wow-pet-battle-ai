"""Checks of the network's ability hints (bp/pvp/probe.py) on hand-made positions with known answers."""
from bp.pvp.probe import probe
from bp.sim import constants as C
from bp.sim.battle import Battle
from bp.sim.data import PetSpec, game_data

PASS = C.A_PASS
CRUSH, SANDSTORM, KICK = 0, 1, 2  # Qiraji Guardling "122": Crush / Sandstorm / Blackout Kick


def pet(name, choices="111"):
    return PetSpec.from_choices(game_data().species_by_name(name), choices, breed=3)


def battle(enemy, choices="111", mine=None):
    bench = [pet("Mechanical Squirrel"), pet("Black Rat")]  # teammates, so a death does not end the battle
    b = Battle([mine or pet("Qiraji Guardling", "122")] + bench, [pet(enemy, choices)] + bench, seed=3)
    b.force_crit = False
    b.force_hit = True
    b.force_variance = 0.0
    return b


def snapshot(b):
    return [(p.hp, p.dead, tuple(p.cooldowns), tuple(a.aid for a in p.auras), p.undead_used, p.mech_revived)
            for t in b.teams for p in t.pets], b.round, b.weather and b.weather.aid


def test_stun_on_critter_is_immune():
    b = battle("Cobbleshell")
    f = probe(b, 0)
    assert f[KICK, 0] == 1 and f[KICK, 4] == 1 and f[KICK, 1] == 0
    assert f[CRUSH, 1] > 0 and f[CRUSH, 4] == 0


def test_stun_on_non_critter_works():
    f = probe(battle("Bone Serpent", "122"), 0)
    assert f[KICK, 0] == 1 and f[KICK, 4] == 0


def test_probe_leaves_battle_untouched():
    b = battle("Cobbleshell")
    b.step(CRUSH, PASS)
    before = snapshot(b)
    probe(b, 0)
    probe(b, 1)
    assert snapshot(b) == before


def test_immortal_undead_takes_no_damage_but_crush_still_heals():
    b = battle("Scourged Whelpling")
    en, me = b.teams[1].active_pet, b.teams[0].active_pet
    en.hp = 5
    b.step(CRUSH, PASS)  # kills it: Undead returns for one round at 1 hp
    assert en.undead_used and not en.dead and en.state(C.S_UNKILLABLE) > 0
    me.hp = me.max_hp // 2
    f = probe(b, 0)
    assert f[CRUSH, 0] == 1 and abs(f[CRUSH, 1]) < 1e-6 and f[CRUSH, 2] == 0
    assert f[CRUSH, 3] > 0  # Humanoid recovery after dealing (0) damage


def test_flying_target_is_unreachable():
    b = battle("Bone Serpent", "122")
    b.step(PASS, 1)  # Lift-Off: in the air this round
    f = probe(b, 0)
    assert f[CRUSH, 5] == 1 and f[CRUSH, 1] == 0


def test_killing_blow():
    b = battle("Cobbleshell")
    en = b.teams[1].active_pet
    en.hp = 50
    f = probe(b, 0)
    assert f[CRUSH, 2] == 1
    assert abs(f[CRUSH, 1] - 50 / en.max_hp) < 1e-6


def test_failsafe_counts_as_a_life():
    b = battle("Iron Starlette")
    en = b.teams[1].active_pet
    en.hp = 50
    f = probe(b, 0)
    assert f[CRUSH, 2] == 1
    b.step(CRUSH, PASS)
    assert en.mech_revived and not en.dead
    f = probe(b, 0)
    assert f[CRUSH, 0] == 1 and f[CRUSH, 1] > 0


def test_block_absorbs_the_hit():
    b = battle("Mechanical Pandaren Dragonling", "112")
    b.step(PASS, 2)  # Decoy
    f = probe(b, 0)
    assert f[CRUSH, 6] == 1 and f[CRUSH, 1] == 0


def test_damage_matches_a_real_round():
    """Enemy passes, fixed dice: the probe's number is exactly what one simulated round deals."""
    b = battle("Cobbleshell")
    en = b.teams[1].active_pet
    f = probe(b, 0)
    hp = en.hp
    b.step(CRUSH, PASS)
    assert abs(f[CRUSH, 1] - (hp - en.hp) / en.max_hp) < 1e-6


def test_other_side_and_unusable():
    b = battle("Cobbleshell")
    b.step(KICK, PASS)  # Blackout Kick now on cooldown
    f = probe(b, 0)
    assert f[KICK, 0] == 0
    g = probe(b, 1)  # Cobbleshell's own view: Ooze Touch hurts the Guardling
    assert g[0, 0] == 1 and g[0, 1] > 0
