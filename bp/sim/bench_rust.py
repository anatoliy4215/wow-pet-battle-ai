"""Speed: Python simulator vs the Rust one (one CPU core each).

  python -m bp.sim.bench_rust
"""
from __future__ import annotations

import random
import time

from ..pvp.pool import TeamSampler
from .battle import Battle
from .clone import fast_clone
from .rust import module, rust_battle, spec_tuple


def main(n_games=2000, seed=3):
    ts = TeamSampler(seed=seed, meta_pet_frac=0.3, meta_team_frac=0.1)
    pairs = [(ts.team(), ts.team()) for _ in range(200)]
    rows = []

    # 1. whole battles, random legal moves
    t0 = time.perf_counter()
    rounds = 0
    for g in range(n_games):
        a, b = pairs[g % len(pairs)]
        bt, pick = Battle(a, b, seed=g), random.Random(g)
        n = 0
        while not bt.over and n < 400:
            bt.step(pick.choice(bt.legal_actions(0)), pick.choice(bt.legal_actions(1)))
            n += 1
        rounds += bt.round
    py_t = time.perf_counter() - t0
    rows.append(("Python: whole battles", n_games / py_t, rounds / py_t))

    t0 = time.perf_counter()
    rounds = 0
    for g in range(n_games):
        a, b = pairs[g % len(pairs)]
        bt, pick = rust_battle(a, b, g), random.Random(g)
        n = 0
        while not bt.over and n < 400:
            bt.step(pick.choice(bt.legal_actions(0)), pick.choice(bt.legal_actions(1)))
            n += 1
        rounds += bt.round
    rp_t = time.perf_counter() - t0
    rows.append(("Rust, driven from Python", n_games / rp_t, rounds / rp_t))

    sp = [([spec_tuple(s) for s in a], [spec_tuple(s) for s in b]) for a, b in pairs]
    n_fast = n_games * 20
    games, rounds, secs = module().bench_random_games(sp, n_fast, seed)
    rows.append(("Rust, loop inside Rust", games / secs, rounds / secs))

    # 2. the look-ahead pattern: copy a mid-battle position and play one round in the copy
    a, b = pairs[0]
    pyb, rsb, pick = Battle(a, b, seed=1), rust_battle(a, b, 1), random.Random(1)
    for _ in range(3):
        acts = (pick.choice(pyb.legal_actions(0)), pick.choice(pyb.legal_actions(1)))
        pyb.step(*acts)
        rsb.step(*acts)
    act = (pyb.legal_actions(0)[0], pyb.legal_actions(1)[0])
    n = 20000
    t0 = time.perf_counter()
    for i in range(n):
        c = fast_clone(pyb, i)
        c.step(*act)
    py_c = (time.perf_counter() - t0) / n
    rs_c = module().bench_clone_step(rsb, n * 10, *act) / (n * 10)

    print(f"{'':28s} {'battles/s':>12s} {'rounds/s':>12s}")
    for name, g, r in rows:
        print(f"{name:28s} {g:12,.0f} {r:12,.0f}")
    print(f"\ncopy + 1 round (look-ahead):  Python {py_c * 1e6:.1f} us   Rust {rs_c * 1e6:.2f} us   x{py_c / rs_c:.0f}")
    print(f"whole battles: Rust {rows[2][1] / rows[0][1]:.0f}x faster inside Rust, "
          f"{rows[1][1] / rows[0][1]:.1f}x when every round is a call from Python")


if __name__ == "__main__":
    main()
