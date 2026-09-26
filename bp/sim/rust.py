"""Bridge to the Rust simulator (rust/bpsim, module `bpsim`) and the differential check against
the Python engine.

  python -m bp.sim.rust --games 2000        # same battles in both engines, compare every round
  python -m bp.sim.bench_rust               # speed of both

Build / reinstall the module after changing rust/bpsim (the Cyrillic project path breaks
`maturin develop`, so build a wheel and install it):
  cd rust/bpsim && ../../.venv/Scripts/maturin.exe build --release -i ../../.venv/Scripts/python.exe
  .venv/Scripts/python.exe -m pip install --force-reinstall --no-deps rust/bpsim/target/wheels/*.whl
After changing game data or the target rules: python -m bp.sim.export_rust
"""
from __future__ import annotations

import argparse
import random
import time
from functools import lru_cache

from .data import PROCESSED, compute_stats


@lru_cache(maxsize=1)
def module():
    import bpsim
    path = PROCESSED / "sim_rust.json"
    if not path.exists():
        from .export_rust import export
        export(path)
    bpsim.load_data(str(path))
    return bpsim


def spec_tuple(spec):
    return (spec.species, [a or 0 for a in (list(spec.abilities) + [None] * 3)[:3]], spec.level,
            tuple(compute_stats(spec)), [(int(k), int(v)) for k, v in spec.extra_states.items()])


def rust_battle(team_a, team_b, seed, log=False):
    return module().Battle([spec_tuple(s) for s in team_a], [spec_tuple(s) for s in team_b], seed, log)


def _aura(a):
    c = a.caster
    return (a.aid, (c.team.side, c.index) if c is not None else None, a.duration, a.age, a.fresh, a.blocks_left,
            bool(a.next_attack))


def snapshot(b):
    """The Python battle in the shape of bpsim.Battle.snapshot()."""
    pets = []
    for t in b.teams:
        rows = []
        for p in t.pets:
            rows.append((p.hp, p.dead, list(p.cooldowns), [a or 0 for a in p.abilities], [_aura(a) for a in p.auras],
                         sorted((int(k), int(v)) for k, v in p.states.items()), tuple(p.channel) if p.channel else None,
                         (p.undead_used, p.mech_revived, p.swapped_in, p.last_ability, sorted(p.revealed))))
        pets.append(rows)
    return (b.round, b.winner, [b.teams[0].active, b.teams[1].active], _aura(b.weather) if b.weather else None,
            [[_aura(a) for a in t.auras] for t in b.teams], pets)


def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def compare_game(team_a, team_b, seed, max_steps=400):
    """Play one battle with random legal actions in both engines. -> None or a description of the
    first difference."""
    from .battle import Battle
    py = Battle(team_a, team_b, seed=seed, log=True)
    rs = rust_battle(team_a, team_b, seed, log=True)
    pick = random.Random(seed * 31 + 7)
    n_log = 0
    if _norm(snapshot(py)) != _norm(rs.snapshot()):
        return "state after setup differs"
    for step in range(max_steps):
        if py.over:
            break
        la, lb = py.legal_actions(0), py.legal_actions(1)
        ra, rb = list(rs.legal_actions(0)), list(rs.legal_actions(1))
        if (la, lb) != (ra, rb):
            return f"step {step}: legal actions py {la} {lb} rust {ra} {rb}"
        acts = (pick.choice(la), pick.choice(lb))
        py.step(*acts)
        rs.step(*acts)
        pl, rl = py.log[n_log:], rs.log[n_log:]
        if pl != rl:
            for i, (x, y) in enumerate(zip(pl + ["<end>"] * 5, rl + ["<end>"] * 5)):
                if x != y:
                    return f"step {step} actions {acts}: log line {i}\n  py:   {x}\n  rust: {y}\n  before: {pl[max(0, i - 3):i]}"
        n_log = len(py.log)
        sp, sr = _norm(snapshot(py)), _norm(rs.snapshot())
        if sp != sr:
            for k, (x, y) in enumerate(zip(sp, sr)):
                if x != y:
                    return f"step {step} actions {acts}: state part {k} differs\n  py:   {x}\n  rust: {y}"
    if bool(py.over) != bool(rs.over):
        return "end differs"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    from ..pvp.pool import TeamSampler
    ts = TeamSampler(seed=a.seed, meta_pet_frac=0.3, meta_team_frac=0.1)
    bad, t0 = [], time.time()
    for g in range(a.games):
        ta, tb = ts.team(), ts.team()
        err = compare_game(ta, tb, seed=a.seed * 100003 + g)
        if err:
            bad.append((g, [p.species for p in ta], [p.species for p in tb], err))
    print(f"{a.games - len(bad)}/{a.games} battles identical ({time.time() - t0:.0f}s)")
    for g, sa, sb, err in bad[:a.show]:
        print(f"game {g} {sa} vs {sb}: {err}")


if __name__ == "__main__":
    main()


def sweep_abilities(games_each=3, seed=7):
    """Every ability of the game data in slot 1 of some pet, a few random battles each: covers the
    opcodes that random teams rarely reach. -> list of (ability, error)."""
    from ..pvp.pool import TeamSampler
    from .data import PetSpec, game_data
    gd = game_data()
    ts = TeamSampler(seed=seed, meta_pet_frac=0.0, meta_team_frac=0.0)
    bad = []
    for aid in sorted(gd.abilities):
        if not gd.abilities[aid]["turns"]:
            continue
        for g in range(games_each):
            ta, tb = ts.team(), ts.team()
            p = ta[0]
            ta[0] = PetSpec(p.species, [aid] + list(p.abilities[1:]), p.level, p.quality, p.breed)
            err = compare_game(ta, tb, seed=seed * 1000003 + aid * 10 + g)
            if err:
                bad.append((aid, gd.abilities[aid]["name"], err))
                break
    return bad
