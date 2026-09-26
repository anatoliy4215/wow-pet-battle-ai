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
