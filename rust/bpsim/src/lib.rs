//! Python module `bpsim`: the Rust battle engine.
//!
//!   import bpsim
//!   bpsim.load_data("data/processed/sim_rust.json")
//!   b = bpsim.Battle(team_a, team_b, seed, log)   # team: [(species, [a1, a2, a3], level, (hp, power, speed), [(state, value)])]
//!   b.step(action_a, action_b); b.legal_actions(side); b.clone(seed); b.snapshot()

mod data;
mod engine;
mod features;
mod rng;
mod search;

use engine::{PetRef, Spec};
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

type PetState = (
    i64,
    bool,
    Vec<i64>,
    Vec<u32>,
    Vec<AuraSnap>,
    Vec<(i64, i64)>,
    Option<(usize, usize)>,
    (bool, bool, i64, Option<u32>, Vec<usize>),
    i64,
);
type BattleState = (i64, Option<i64>, Vec<usize>, Option<AuraSnap>, Vec<Vec<AuraSnap>>, Vec<Vec<PetState>>, Option<usize>);

fn aura_from(b: &mut engine::Battle, a: &AuraSnap) -> engine::Aura {
    b.uid += 1;
    engine::Aura {
        aid: a.0,
        caster: a.1.map(|(side, idx)| PetRef { side, idx }),
        duration: a.2,
        age: a.3,
        fresh: a.4,
        blocks_left: a.5,
        next_attack: a.6,
        stacks: 1,
        uid: b.uid,
    }
}

fn battle_from_state(team_a: &[SpecTuple], team_b: &[SpecTuple], state: BattleState, seed: u64) -> engine::Battle {
    let a: Vec<Spec> = team_a.iter().map(to_spec).collect();
    let bb: Vec<Spec> = team_b.iter().map(to_spec).collect();
    let mut b = engine::Battle::new(&a, &bb, seed, false);
    b.uid = 0;
    let (round, winner, active, weather, team_auras, pets, first_side) = state;
    b.round = round;
    b.winner = winner;
    b.first_side = first_side;
    for s in 0..2 {
        b.teams[s].active = active[s];
        let ta: Vec<engine::Aura> = team_auras[s].iter().map(|x| aura_from(&mut b, x)).collect();
        b.teams[s].auras = ta;
        for (i, ps) in pets[s].iter().enumerate() {
            let auras: Vec<engine::Aura> = ps.4.iter().map(|x| aura_from(&mut b, x)).collect();
            let p = &mut b.teams[s].pets[i];
            p.hp = ps.0;
            p.dead = ps.1;
            for j in 0..3 {
                p.cooldowns[j] = ps.2.get(j).copied().unwrap_or(0);
                p.abilities[j] = ps.3.get(j).copied().unwrap_or(0);
            }
            p.auras = auras;
            p.states = engine::States(ps.5.clone());
            p.channel = ps.6;
            p.undead_used = (ps.7).0;
            p.mech_revived = (ps.7).1;
            p.swapped_in = (ps.7).2;
            p.last_ability = (ps.7).3;
            p.revealed = (ps.7).4.iter().fold(0u8, |m, &x| m | (1 << x));
            p.family = ps.8;
        }
    }
    b.weather = weather.as_ref().map(|w| aura_from(&mut b, w));
    b
}

fn batch_dict<'py>(py: Python<'py>, ob: search::ObsBatch) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyAny>)> {
    use features::*;
    let n = ob.n;
    let d = PyDict::new(py);
    d.set_item("glob", PyArray1::from_vec(py, ob.glob).reshape(vec![n, N_GLOB])?)?;
    d.set_item("pet_int", PyArray1::from_vec(py, ob.pet_int).reshape(vec![n, 6])?)?;
    d.set_item("pet_float", PyArray1::from_vec(py, ob.pet_float).reshape(vec![n, 6, N_PET])?)?;
    d.set_item("ab_int", PyArray1::from_vec(py, ob.ab_int).reshape(vec![n, N_ABIL, 3])?)?;
    d.set_item("ab_float", PyArray1::from_vec(py, ob.ab_float).reshape(vec![n, N_ABIL, N_ABF])?)?;
    d.set_item("ab_mask", PyArray1::from_vec(py, ob.ab_mask).reshape(vec![n, N_ABIL])?)?;
    d.set_item("aura_int", PyArray1::from_vec(py, ob.aura_int).reshape(vec![n, MAX_AURAS, 2])?)?;
    d.set_item("aura_float", PyArray1::from_vec(py, ob.aura_float).reshape(vec![n, MAX_AURAS, N_AUF])?)?;
    d.set_item("aura_mask", PyArray1::from_vec(py, ob.aura_mask).reshape(vec![n, MAX_AURAS])?)?;
    let mask = PyArray1::from_vec(py, ob.mask).reshape(vec![n, 7])?.into_any();
    Ok((d, mask))
}

fn read_replies(pol: PyReadonlyArray2<f32>, val: PyReadonlyArray1<f32>, lo: PyReadonlyArray2<f32>) -> PyResult<Vec<search::Reply>> {
    let (p, v, l) = (pol.as_array(), val.as_array(), lo.as_array());
    let n = v.len();
    if p.shape() != [n, 7] || l.shape() != [n, 9] {
        return Err(PyValueError::new_err("reply shapes must be (n,7), (n,), (n,9)"));
    }
    Ok((0..n)
        .map(|i| {
            let mut r = search::Reply { pol: [0.0; 7], val: v[i], lo: [0.0; 9] };
            for j in 0..7 {
                r.pol[j] = p[[i, j]];
            }
            for j in 0..9 {
                r.lo[j] = l[[i, j]];
            }
            r
        })
        .collect())
}

/// Self-play actor: many battles advanced in parallel threads; Python only runs the network.
///   pool = ActorPool(n_envs, seed, k_deep=8, k_cheap=4, depth_deep=2, deep_frac=0.25, tau=0.2, gamma=0.98)
///   pool.add_starts([(team_a, team_b, state or None), ...])
///   pool.kick(); then repeat: obs, mask = pool.requests(); pool.reply(policy, value, loadout)
///   pool.take_samples(); pool.take_games()
#[pyclass(name = "ActorPool", module = "bpsim")]
struct PyActorPool {
    pool: search::Pool,
    seed_rng: rng::PyRandom,
}

#[pymethods]
impl PyActorPool {
    #[new]
    #[pyo3(signature = (n_envs, seed=0, k_deep=8, k_cheap=4, depth_deep=2, deep_frac=0.25, tau=0.2, gamma=0.98))]
    #[allow(clippy::too_many_arguments)]
    fn new(n_envs: usize, seed: u64, k_deep: usize, k_cheap: usize, depth_deep: usize, deep_frac: f64, tau: f32, gamma: f64) -> Self {
        let cfg = search::Cfg { k_deep, k_cheap, depth_deep, deep_frac, tau, gamma };
        PyActorPool { pool: search::Pool::new(n_envs, seed, cfg), seed_rng: rng::PyRandom::new(seed ^ 0xabcdef) }
    }

    /// starts: [(team_a, team_b, state or None)]; None = a fresh battle with the lead pick
    fn add_starts(&mut self, starts: Vec<(Vec<SpecTuple>, Vec<SpecTuple>, Option<BattleState>)>) {
        let mut q = self.pool.starts.lock().unwrap();
        for (a, b, st) in starts {
            let seed = self.seed_rng.getrandbits(32);
            let (battle, lead) = match st {
                Some(state) => (battle_from_state(&a, &b, state, seed), false),
                None => {
                    let sa: Vec<Spec> = a.iter().map(to_spec).collect();
                    let sb: Vec<Spec> = b.iter().map(to_spec).collect();
                    (engine::Battle::new(&sa, &sb, seed, false), true)
                }
            };
            q.push_back(search::Start { battle, lead });
        }
    }

    fn n_starts(&self) -> usize {
        self.pool.starts.lock().unwrap().len()
    }

    /// CPU work that needs no network reply (start battles, first searches)
    fn kick(&mut self, py: Python<'_>) {
        let pool = &mut self.pool;
        py.allow_threads(|| pool.work(&[]));
    }

    fn n_requests(&self) -> usize {
        self.pool.total_reqs()
    }

    fn requests<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyAny>)> {
        let n = self.pool.total_reqs();
        let ob = search::stack_obs(self.pool.envs.iter().flat_map(|e| e.reqs()), n);
        batch_dict(py, ob)
    }

    fn reply(&mut self, py: Python<'_>, pol: PyReadonlyArray2<f32>, val: PyReadonlyArray1<f32>, lo: PyReadonlyArray2<f32>) -> PyResult<()> {
        let replies = read_replies(pol, val, lo)?;
        if replies.len() != self.pool.total_reqs() {
            return Err(PyValueError::new_err(format!("{} replies for {} requests", replies.len(), self.pool.total_reqs())));
        }
        let pool = &mut self.pool;
        py.allow_threads(|| pool.work(&replies));
        Ok(())
    }

    fn n_samples(&self) -> usize {
        self.pool.samples.len()
    }

    /// Finished training samples as numpy arrays (obs keys, mask, pi, q, z, zd, opp_act, lo_y, pw).
    fn take_samples<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = std::mem::take(&mut self.pool.samples);
        let n = samples.len();
        let mut ob = search::ObsBatch::with_capacity(n);
        let mut pi = Vec::with_capacity(n * 7);
        let mut q = Vec::with_capacity(n * 7);
        let mut z = Vec::with_capacity(n);
        let mut zd = Vec::with_capacity(n);
        let mut opp = Vec::with_capacity(n);
        let mut lo = Vec::with_capacity(n * 9);
        let mut pw = Vec::with_capacity(n);
        for s in &samples {
            ob.push(&s.obs, &s.mask);
            pi.extend_from_slice(&s.pi);
            q.extend_from_slice(&s.q);
            z.push(s.z);
            zd.push(s.zd);
            opp.push(s.opp_act);
            lo.extend_from_slice(&s.lo_y);
            pw.push(s.pw);
        }
        let (d, mask) = batch_dict(py, ob)?;
        d.set_item("mask", mask)?;
        d.set_item("pi", PyArray1::from_vec(py, pi).reshape(vec![n, 7])?)?;
        d.set_item("q", PyArray1::from_vec(py, q).reshape(vec![n, 7])?)?;
        d.set_item("z", PyArray1::from_vec(py, z))?;
        d.set_item("zd", PyArray1::from_vec(py, zd))?;
        d.set_item("opp_act", PyArray1::from_vec(py, opp))?;
        d.set_item("lo_y", PyArray1::from_vec(py, lo).reshape(vec![n, 9])?)?;
        d.set_item("pw", PyArray1::from_vec(py, pw))?;
        Ok(d)
    }

    /// [(winner or -1, rounds)] of the battles finished since the last call
    fn take_games(&mut self) -> Vec<(i64, i64)> {
        std::mem::take(&mut self.pool.games)
    }
}

/// One look-ahead for the phone advisor (any depth), driven by Python like the pool.
#[pyclass(name = "Search", module = "bpsim")]
struct PySearch {
    s: search::Search,
    rng: rng::PyRandom,
}

#[pymethods]
impl PySearch {
    #[new]
    #[pyo3(signature = (battle, side, phase=1, k=16, depth=1, tau=0.2, seed=0, lo=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(battle: &PyBattle, side: usize, phase: usize, k: usize, depth: usize, tau: f32, seed: u64, lo: Option<Vec<f32>>) -> Self {
        let mut rng = rng::PyRandom::new(seed);
        let lo = lo.map(|v| {
            let mut a = [0.5f32; 9];
            for (i, x) in v.iter().take(9).enumerate() {
                a[i] = *x;
            }
            a
        });
        let s = search::Search::new(&battle.b, side, phase, k, depth, tau, lo, &mut rng, true);
        PySearch { s, rng }
    }

    #[getter]
    fn done(&self) -> bool {
        self.s.done()
    }

    fn requests<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyAny>)> {
        batch_dict(py, search::stack_obs(self.s.reqs.iter(), self.s.reqs.len()))
    }

    fn reply(&mut self, py: Python<'_>, pol: PyReadonlyArray2<f32>, val: PyReadonlyArray1<f32>, lo: PyReadonlyArray2<f32>) -> PyResult<()> {
        let replies = read_replies(pol, val, lo)?;
        if replies.len() != self.s.reqs.len() {
            return Err(PyValueError::new_err("reply count does not match the requests"));
        }
        let (s, rng) = (&mut self.s, &mut self.rng);
        py.allow_threads(|| s.advance(&replies, rng));
        Ok(())
    }

    /// {prior, value, lo, pi, q: {action: (mean, stderr, n)}}
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("prior", self.s.prior.to_vec())?;
        d.set_item("value", self.s.value)?;
        d.set_item("lo", self.s.lo.to_vec())?;
        d.set_item("pi", self.s.pi.to_vec())?;
        let q = PyDict::new(py);
        for &a in &self.s.legal {
            q.set_item(a, (self.s.q[a], self.s.se[a], self.s.n[a]))?;
        }
        d.set_item("q", q)?;
        Ok(d)
    }
}

/// Numpy dict of one observation (same keys / dtypes as bp.pvp.features.encode).
fn obs_dict<'py>(py: Python<'py>, o: &features::Obs) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let f32a = |v: Vec<f32>, shape: Vec<usize>| PyArray1::from_vec(py, v).reshape(shape);
    let i64a = |v: Vec<i64>, shape: Vec<usize>| PyArray1::from_vec(py, v).reshape(shape);
    d.set_item("glob", f32a(o.glob.to_vec(), vec![features::N_GLOB])?)?;
    d.set_item("pet_int", i64a(o.pet_int.to_vec(), vec![6])?)?;
    d.set_item("pet_float", f32a(o.pet_float.iter().flatten().copied().collect(), vec![6, features::N_PET])?)?;
    d.set_item("ab_int", i64a(o.ab_int.iter().flatten().copied().collect(), vec![features::N_ABIL, 3])?)?;
    d.set_item("ab_float", f32a(o.ab_float.iter().flatten().copied().collect(), vec![features::N_ABIL, features::N_ABF])?)?;
    d.set_item("ab_mask", f32a(o.ab_mask.to_vec(), vec![features::N_ABIL])?)?;
    d.set_item("aura_int", i64a(o.aura_int.iter().flatten().copied().collect(), vec![features::MAX_AURAS, 2])?)?;
    d.set_item("aura_float", f32a(o.aura_float.iter().flatten().copied().collect(), vec![features::MAX_AURAS, features::N_AUF])?)?;
    d.set_item("aura_mask", f32a(o.aura_mask.to_vec(), vec![features::MAX_AURAS])?)?;
    Ok(d)
}

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

    /// A battle in any state: specs + the state tuple of bp.sim.rust.export_state(python battle).
    #[staticmethod]
    #[pyo3(signature = (team_a, team_b, state, seed=0))]
    fn from_state(team_a: Vec<SpecTuple>, team_b: Vec<SpecTuple>, state: BattleState, seed: u64) -> Self {
        PyBattle { b: battle_from_state(&team_a, &team_b, state, seed) }
    }

    #[pyo3(signature = (side, phase=1, hide=true))]
    fn encode<'py>(&self, py: Python<'py>, side: usize, phase: usize, hide: bool) -> PyResult<Bound<'py, PyDict>> {
        obs_dict(py, &features::encode(&self.b, side, phase, hide))
    }

    #[pyo3(signature = (side, phase=1))]
    fn action_mask<'py>(&self, py: Python<'py>, side: usize, phase: usize) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, features::action_mask(&self.b, side, phase).to_vec())
    }

    fn probe(&self, side: usize) -> Vec<Vec<f32>> {
        let legal = self.b.legal_actions(side);
        self.b.probe(side, &legal).iter().map(|r| r.to_vec()).collect()
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
    m.add_class::<PyActorPool>()?;
    m.add_class::<PySearch>()?;
    Ok(())
}
