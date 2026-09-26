"""Mechanics checks. Numbers marked [log] come from a real combat log (Blizzard forum thread 1013569:
Anubisath Idol H/H vs Whispering Pandaren Spirit)."""
import pytest

from bp.sim import constants as C
from bp.sim.battle import Battle
from bp.sim.data import PetSpec, compute_stats, game_data

IDOL = 1155
DUSTY = 1135
CRUSH, DEFLECTION, SANDSTORM = 406, 490, 453
MOTH_BALLS, MOTH_DUST, COCOON = 507, 508, 506
PASS = C.A_PASS


def idol(abilities=(CRUSH, DEFLECTION, SANDSTORM)):
    return PetSpec(IDOL, list(abilities), 25, 3, 6)


def dusty(abilities=(MOTH_BALLS, MOTH_DUST, COCOON)):
    return PetSpec(DUSTY, list(abilities), 25, 5, 12)


def filler(species_name="Mechanical Squirrel"):
    sid = game_data().species_by_name(species_name)
    return PetSpec.from_choices(sid, "111")


def battle(a, b, **force):
    bt = Battle(a, b, seed=1, log=True)
    bt.force_hit = force.get("hit", True)
    bt.force_crit = force.get("crit", False)
    bt.force_variance = force.get("variance", 0.0)
    return bt


def dmg_lines(bt):
    return [l for l in bt.log if "dealt" in l]


def test_stats_match_log_and_tamer_data():
    assert compute_stats(idol()) == (1725, 276, 244)  # [log] Recovery 4% = 69
    assert compute_stats(dusty()) == (1769, 315, 315)


def test_plain_hit_and_humanoid_recovery():
    bt = battle([idol()], [dusty()])
    bt.step(0, 2)  # Crush vs Cocoon Strike (Dusty is faster)
    idol_pet, dusty_pet = bt.teams[0].pets[0], bt.teams[1].pets[0]
    # [log] "Cocoon Strike dealt 167 damage": 10 * (1 + 315/20) = 167.5 -> 167
    # Crush is blocked by Silk Cocoon, so no Recovery this round ([log] round 2: no Recovery line)
    assert idol_pet.hp == 1725 - 167
    assert dusty_pet.hp == dusty_pet.max_hp
    assert any("blocked" in l for l in bt.log)
    bt.step(0, PASS)  # Crush lands -> Recovery heals 4% = 69 [log]
    assert any("healed 69" in l for l in bt.log)


def test_sandstorm_flat_reduction_and_crit():
    bt = battle([idol()], [dusty()])
    bt.step(2, PASS)  # Idol: Sandstorm
    assert bt.weather is not None and bt.weather.aid == 454
    start = bt.teams[0].pets[0].hp
    bt.step(PASS, 0)  # Moth Balls: 4 hits
    hits = [l for l in dmg_lines(bt) if "Moth Balls" in l]
    assert len(hits) == 4
    # [log] 167.5 - 5*(1+276/20)=74 -> 93
    assert all(" 93 damage" in l for l in hits)
    assert bt.teams[0].pets[0].hp == start - 4 * 93
    bt2 = battle([idol()], [dusty()], crit=True)
    bt2.step(2, PASS)
    bt2.step(PASS, 0)
    assert all(" 139 damage" in l for l in dmg_lines(bt2) if "Moth Balls" in l)  # [log] floor(93*1.5)


def test_weather_duration():
    bt = battle([idol()], [dusty()])
    bt.step(2, PASS)  # round 1: Sandstorm, Duration 5
    for r in range(2, 7):
        assert bt.weather is not None, r
        bt.step(PASS, PASS)
    # [log] cast in round 2 -> "no longer the weather" in round 7, i.e. 5 rounds after the cast round
    assert bt.weather is None


def test_speed_reduction_duration():
    bt = battle([idol()], [dusty()])
    bt.step(PASS, 0)  # round 1 Moth Balls -> Speed Reduction (154, Duration 3)
    pet = bt.teams[0].pets[0]
    assert pet.has_aura(154)
    bt.step(PASS, PASS)
    bt.step(PASS, PASS)
    assert pet.has_aura(154)
    bt.step(PASS, PASS)  # [log] applied round 3, faded end of round 6
    assert not pet.has_aura(154)


def test_deflection_lasts_current_round_only():
    bt = battle([idol()], [dusty()], hit=None)
    bt.step(1, 0)  # Deflection (round start priority) vs Moth Balls: all dodged
    assert not [l for l in dmg_lines(bt) if "Moth Balls" in l]
    assert not bt.teams[0].pets[0].has_aura(903)  # [log] "Deflection fades" same round


def test_sleep_from_faster_pet_covers_two_rounds():
    bt = battle([idol()], [dusty()])
    bt.step(0, 1)  # Moth Dust (faster) -> Asleep: Idol loses this round
    pet = bt.teams[0].pets[0]
    assert not any("Crush" in l for l in dmg_lines(bt))
    assert bt.legal_actions(0) == [PASS]  # [log] still asleep next round


def test_undead_extra_round_then_death():
    gd = game_data()
    und = next(s for s, v in gd.species.items() if v["battle"] and v["family"] == C.UNDEAD)
    bt = battle([PetSpec.from_choices(und, "111"), filler()], [dusty()])
    pet = bt.teams[0].pets[0]
    pet.hp = 1
    bt.step(PASS, 2)  # killed by Cocoon Strike
    assert not pet.dead and pet.has_aura(C.UNDEAD_AURA)
    bt.step(PASS, PASS)  # immortal round
    assert pet.dead


def test_mechanical_failsafe():
    sq = game_data().species_by_name("Mechanical Squirrel")
    bt = battle([PetSpec.from_choices(sq, "111"), filler()], [dusty()])
    pet = bt.teams[0].pets[0]
    pet.hp = 1
    bt.step(PASS, 2)
    assert not pet.dead and pet.hp == int(pet.max_hp * 0.2)


def test_quick_attack_goes_first():
    bt = battle([idol((563, CRUSH, SANDSTORM))], [dusty()])
    bt.step(0, 2)  # slower Idol with Quick Attack must hit before Cocoon is up
    assert any("Quick Attack" in l for l in dmg_lines(bt))


def test_burrow_untargetable():
    bt = battle([idol((159, CRUSH, SANDSTORM))], [dusty()])
    bt.step(0, 0)  # Dusty acts first -> Idol not underground yet
    bt.step(PASS, 0)  # round 2: Idol underground, Moth Balls can't hit; Burrow strikes
    assert not [l for l in dmg_lines(bt) if l.startswith("[2]") and "Moth Balls" in l]
    assert any(l.startswith("[2]") and "Burrow dealt" in l for l in bt.log)


def test_team_hit_vs_multihit():
    enemy = [dusty(), dusty(), dusty()]
    bt = battle([idol((1035, CRUSH, SANDSTORM))], enemy)  # Spiritfire Beam: whole team
    bt.step(0, PASS)
    assert sum(p.hp < p.max_hp for p in bt.teams[1].pets) == 3
    bt = battle([idol((1100, CRUSH, SANDSTORM))], enemy)  # Blessed Hammer: 3 hits on the active pet
    bt.step(0, PASS)
    assert sum(p.hp < p.max_hp for p in bt.teams[1].pets) == 1


def test_feign_death_dodges_one_round_then_swaps():
    junk = game_data().species_by_name("Yellow Junkhopper")
    bt = battle([PetSpec(junk, [118, 568, 233], 25, 3, 7), filler()], [dusty()], hit=None)
    bt.step(1, 0)  # Feign Death vs Moth Balls: dodged, then the pet swaps out
    assert not [l for l in dmg_lines(bt) if "Moth Balls" in l]
    assert bt.teams[0].active == 1
    assert not bt.teams[0].pets[0].has_aura(804)  # dodge aura removed


def test_consume_corpse_needs_an_unused_allied_corpse():
    """In game Consume Corpse is greyed out until an ally has died; each corpse feeds it once."""
    gd = game_data()
    raven = gd.species_by_name("Spectral Raven")
    slot = next(k for k, opts in enumerate(gd.species[raven]["abilities"]) if 665 in opts)
    choices = "".join(str(opts.index(665) + 1) if k == slot else "1" for k, opts in enumerate(gd.species[raven]["abilities"]))
    bt = battle([PetSpec.from_choices(raven, choices), filler(), filler("Black Rat")], [filler(), filler(), filler()])
    me = bt.teams[0].active_pet
    assert slot not in bt.legal_actions(0)
    bt.teams[0].pets[1].hp, bt.teams[0].pets[1].dead = 0, True  # an ally fell earlier
    me.hp = me.max_hp // 2
    assert slot in bt.legal_actions(0)
    hp = me.hp
    bt.step(slot, PASS)
    assert me.hp > hp
    for _ in range(me.cooldowns[slot] + 1):
        bt.step(PASS, PASS)
    assert slot not in bt.legal_actions(0)  # the only corpse is used up
