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


def export_state(b):
    """Python battle -> the state tuple of bpsim.Battle.from_state."""
    snap = snapshot(b)
    pets = [[row + (p.family,) for row, p in zip(rows, t.pets)] for rows, t in zip(snap[5], b.teams)]
    return snap[:5] + (pets, getattr(b, "first_side", None))


def to_rust(b, seed=0):
    """A Rust copy of a Python battle in its current state (fresh dice from `seed`)."""
    rs = module().Battle.from_state([spec_tuple(p.spec) for p in b.teams[0].pets],
                                    [spec_tuple(p.spec) for p in b.teams[1].pets], export_state(b), seed)
    return rs


def positions(n, seed=1, practice=0.2):
    """Python battles in all phases (lead pick, rounds, replacements), from games of greedy / random
    moves, some started from practice positions."""
    from ..pvp.bots import GreedyBot
    from ..pvp.env import PvPEnv
    from ..pvp.pool import TeamSampler
    from ..pvp.scenarios import start
    from .clone import fast_clone
    rng = random.Random(seed)
    env = PvPEnv(TeamSampler(seed=seed, meta_pet_frac=0.3, meta_team_frac=0.1), seed=seed + 1)
    bot = GreedyBot()
    out = []
    while len(out) < n:
        if rng.random() < practice:
            start(env, env.sampler, rng)
        else:
            env.reset()
            out.append((fast_clone(env.battle, 1), env.phase))
        while not env.done and len(out) < n:
            acts = {}
            for s in env.pending():
                legal = [a for a in range(7) if env.observe(s)[1][a] > 0]
                acts[s] = rng.choice(legal) if rng.random() < 0.4 else bot.act(env.battle, s, env.phase, rng)
            env.act(acts)
            if not env.done:
                out.append((fast_clone(env.battle, 1), env.phase))
    return out


def compare_encode(n=2000, seed=1):
    """encode / action_mask of Python vs Rust on the same positions, both sides. -> list of errors."""
    import numpy as np
    from ..pvp.features import KEYS, action_mask, encode
    errs = []
    for i, (b, phase) in enumerate(positions(n, seed)):
        rs = to_rust(b)
        for side in (0, 1):
            po, ro = encode(b, side, phase), rs.encode(side, phase)
            for k in KEYS:
                if po[k].shape != ro[k].shape or not np.array_equal(po[k].astype(ro[k].dtype), ro[k]):
                    diff = np.argwhere(po[k].astype(ro[k].dtype) != ro[k]) if po[k].shape == ro[k].shape else "shape"
                    errs.append((i, side, phase, k, diff[:4].tolist() if not isinstance(diff, str) else diff,
                                 po[k][tuple(diff[0])] if not isinstance(diff, str) and len(diff) else None,
                                 ro[k][tuple(diff[0])] if not isinstance(diff, str) and len(diff) else None))
                    break
            if not np.array_equal(action_mask(b, side, phase), rs.action_mask(side, phase)):
                errs.append((i, side, phase, "mask"))
    return errs


def compare_continue(n=1000, seed=3, steps=30):
    """A Python battle moved to Rust mid-game must go on identically (same seed, same moves)."""
    from .clone import fast_clone
    errs = []
    for i, (b, _) in enumerate(positions(n, seed)):
        py = fast_clone(b, 1000 + i)
        py.logging, py.log = True, []
        rs = to_rust(b, 1000 + i)
        rs.logging = True
        pick = random.Random(i)
        for s in range(steps):
            if py.over:
                break
            la, lb = py.legal_actions(0), py.legal_actions(1)
            if (la, lb) != (list(rs.legal_actions(0)), list(rs.legal_actions(1))):
                errs.append((i, s, "legal"))
                break
            acts = (pick.choice(la), pick.choice(lb))
            py.step(*acts)
            rs.step(*acts)
            if py.log != list(rs.log) or _norm(snapshot(py)) != _norm(rs.snapshot()):
                errs.append((i, s, "state/log"))
                break
    return errs
