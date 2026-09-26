//! Python module `bpsim`: the Rust battle engine.
//!
//!   import bpsim
//!   bpsim.load_data("data/processed/sim_rust.json")
//!   b = bpsim.Battle(team_a, team_b, seed, log)   # team: [(species, [a1, a2, a3], level, (hp, power, speed), [(state, value)])]
//!   b.step(action_a, action_b); b.legal_actions(side); b.clone(seed); b.snapshot()

mod data;
mod engine;
mod rng;

use engine::{PetRef, Spec};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;

type SpecTuple = (u32, Vec<Option<u32>>, i64, (i64, i64, i64), Vec<(i64, i64)>);

fn to_spec(t: &SpecTuple) -> Spec {
    let mut ab = [0u32; 3];
    for (i, a) in t.1.iter().take(3).enumerate() {
        ab[i] = a.unwrap_or(0);
    }
    Spec { species: t.0, abilities: ab, level: t.2, stats: t.3, extra_states: t.4.clone() }
}

#[pyfunction]
fn load_data(path: &str) -> PyResult<()> {
    data::init(path).map_err(PyValueError::new_err)
}

#[pyclass(name = "Battle", module = "bpsim")]
struct PyBattle {
    b: engine::Battle,
}

type AuraSnap = (u32, Option<(usize, usize)>, i64, i64, bool, i64, bool);
type PetSnap = (
    i64,
    bool,
    Vec<i64>,
    Vec<u32>,
    Vec<AuraSnap>,
    Vec<(i64, i64)>,
    Option<(usize, usize)>,
    (bool, bool, i64, Option<u32>, Vec<usize>),
);

fn aura_snap(a: &engine::Aura) -> AuraSnap {
    (a.aid, a.caster.map(|c| (c.side, c.idx)), a.duration, a.age, a.fresh, a.blocks_left, a.next_attack)
}

#[pymethods]
impl PyBattle {
    #[new]
    #[pyo3(signature = (team_a, team_b, seed=0, log=false))]
    fn new(team_a: Vec<SpecTuple>, team_b: Vec<SpecTuple>, seed: u64, log: bool) -> Self {
        let a: Vec<Spec> = team_a.iter().map(to_spec).collect();
        let b: Vec<Spec> = team_b.iter().map(to_spec).collect();
        PyBattle { b: engine::Battle::new(&a, &b, seed, log) }
    }

    fn step(&mut self, a: usize, b: usize) {
        self.b.step(a, b);
    }

    fn legal_actions(&self, side: usize) -> Vec<usize> {
        self.b.legal_actions(side)
    }

    fn needs_replacement(&self, side: usize) -> bool {
        self.b.needs_replacement(side)
    }

    #[pyo3(signature = (seed=0))]
    fn clone(&self, seed: u64) -> PyBattle {
        PyBattle { b: self.b.clone_with_seed(seed) }
    }

    #[getter]
    fn round(&self) -> i64 {
        self.b.round
    }
    #[getter]
    fn winner(&self) -> Option<i64> {
        self.b.winner
    }
    #[getter]
    fn over(&self) -> bool {
        self.b.over()
    }
    #[getter]
    fn log(&self) -> Vec<String> {
        self.b.log.clone()
    }
    #[getter]
    fn unknown_ops(&self) -> Vec<(i64, i64)> {
        self.b.unknown_ops.clone()
    }
    #[setter]
    fn set_logging(&mut self, v: bool) {
        self.b.logging = v;
    }
    #[setter]
    fn set_force_hit(&mut self, v: Option<bool>) {
        self.b.force_hit = v;
    }
    #[setter]
    fn set_force_crit(&mut self, v: Option<bool>) {
        self.b.force_crit = v;
    }
    #[setter]
    fn set_force_variance(&mut self, v: Option<f64>) {
        self.b.force_variance = v;
    }
    #[setter]
    fn set_force_first(&mut self, v: Option<usize>) {
        self.b.force_first = v;
    }

    fn max_hp(&self, side: usize, idx: usize) -> i64 {
        self.b.max_hp(PetRef { side, idx })
    }
    fn power(&self, side: usize, idx: usize) -> i64 {
        self.b.power(PetRef { side, idx })
    }
    fn speed(&self, side: usize, idx: usize) -> i64 {
        self.b.speed(PetRef { side, idx })
    }
    fn state(&self, side: usize, idx: usize, sid: i64) -> i64 {
        self.b.state(PetRef { side, idx }, sid)
    }

    /// Everything the Python engine keeps, for differential tests:
    /// (round, winner, [active a, active b], weather, [team auras x2], [pets x2])
    fn snapshot(&self) -> (i64, Option<i64>, Vec<usize>, Option<AuraSnap>, Vec<Vec<AuraSnap>>, Vec<Vec<PetSnap>>) {
        let b = &self.b;
        let teams_auras = b.teams.iter().map(|t| t.auras.iter().map(aura_snap).collect()).collect();
        let pets = b
            .teams
            .iter()
            .map(|t| {
                t.pets
                    .iter()
                    .map(|p| {
                        let mut st = p.states.0.clone();
                        st.sort();
                        let rev: Vec<usize> = (0..3).filter(|i| p.revealed & (1 << i) != 0).collect();
                        (
                            p.hp,
                            p.dead,
                            p.cooldowns.to_vec(),
                            p.abilities.to_vec(),
                            p.auras.iter().map(aura_snap).collect(),
                            st,
                            p.channel,
                            (p.undead_used, p.mech_revived, p.swapped_in, p.last_ability, rev),
                        )
                    })
                    .collect()
            })
            .collect();
        (
            b.round,
            b.winner,
            vec![b.teams[0].active, b.teams[1].active],
            b.weather.as_ref().map(aura_snap),
            teams_auras,
            pets,
        )
    }
}

/// Speed test inside Rust: `n` games with uniformly random legal actions, cycling through the given
/// team pairs. -> (games, rounds, seconds)
#[pyfunction]
fn bench_random_games(pairs: Vec<(Vec<SpecTuple>, Vec<SpecTuple>)>, n: usize, seed: u64) -> (usize, i64, f64) {
    let specs: Vec<(Vec<Spec>, Vec<Spec>)> =
        pairs.iter().map(|(a, b)| (a.iter().map(to_spec).collect(), b.iter().map(to_spec).collect())).collect();
    let mut pick = rng::PyRandom::new(seed ^ 0x5eed);
    let t0 = Instant::now();
    let mut rounds = 0i64;
    for g in 0..n {
        let (a, b) = &specs[g % specs.len()];
        let mut bt = engine::Battle::new(a, b, seed.wrapping_add(g as u64), false);
        let mut steps = 0;
        while !bt.over() && steps < 400 {
            let la = bt.legal_actions(0);
            let lb = bt.legal_actions(1);
            let x = la[pick.randbelow(la.len() as u64) as usize];
            let y = lb[pick.randbelow(lb.len() as u64) as usize];
            bt.step(x, y);
            steps += 1;
        }
        rounds += bt.round;
    }
    (n, rounds, t0.elapsed().as_secs_f64())
}

/// Speed test of the look-ahead pattern: clone a battle `n` times and play one round in each copy.
#[pyfunction]
fn bench_clone_step(battle: &PyBattle, n: usize, a: usize, b: usize) -> f64 {
    let t0 = Instant::now();
    for i in 0..n {
        let mut c = battle.b.clone_with_seed(i as u64);
        c.step(a, b);
    }
    t0.elapsed().as_secs_f64()
}

#[pymodule]
fn bpsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_data, m)?)?;
    m.add_function(wrap_pyfunction!(bench_random_games, m)?)?;
    m.add_function(wrap_pyfunction!(bench_clone_step, m)?)?;
    m.add_class::<PyBattle>()?;
    Ok(())
}
