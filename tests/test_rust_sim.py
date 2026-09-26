"""The Rust simulator (rust/bpsim) must play exactly the same battles as the Python one."""
import pytest

bpsim = pytest.importorskip("bpsim")

from bp.pvp.pool import TeamSampler  # noqa: E402
from bp.sim.rust import compare_game, sweep_abilities  # noqa: E402


def test_random_battles_identical():
    ts = TeamSampler(seed=11, meta_pet_frac=0.3, meta_team_frac=0.1)
    errors = [compare_game(ts.team(), ts.team(), seed=g) for g in range(300)]
    assert [e for e in errors if e] == []


def test_every_ability_identical():
    assert sweep_abilities(games_each=1, seed=5) == []


def test_features_and_hints_identical():
    from bp.sim.rust import compare_continue, compare_encode
    assert compare_encode(300, seed=4) == []
    assert compare_continue(150, seed=5, steps=15) == []


def test_actor_pool_plays_games():
    import numpy as np
    from bp.sim.rust import module, spec_tuple
    ts = TeamSampler(seed=3)
    pool = module().ActorPool(8, seed=1, k_deep=4, k_cheap=2, depth_deep=2, deep_frac=0.5)
    pool.add_starts([([spec_tuple(s) for s in ts.team()], [spec_tuple(s) for s in ts.team()], None) for _ in range(12)])
    pool.kick()
    rng = np.random.default_rng(0)
    for _ in range(5000):
        obs, mask = pool.requests()
        n = len(mask)
        if n == 0:
            if pool.n_starts() == 0:
                break
            pool.kick()
            continue
        p = mask / mask.sum(1, keepdims=True)
        pool.reply(p.astype(np.float32), rng.uniform(-1, 1, n).astype(np.float32), np.full((n, 9), 0.5, np.float32))
    s = pool.take_samples()
    assert len(pool.take_games()) == 12
    assert np.allclose(s["pi"].sum(1), 1, atol=1e-5) and (s["pi"] * (1 - s["mask"])).sum() == 0
    assert set(np.unique(s["pw"])) <= {0.0, 1.0} and np.abs(s["zd"]).max() <= 1
