//! Look-ahead search and the self-play actor loop in Rust (port of bp/pvp/actor.py + search.py).
//!
//! A Search belongs to one side of one position and advances in stages; each stage asks the network
//! for a batch of evaluations (requests) and continues with the replies:
//!   Root         root position (prior, value, hidden-ability guess) + the opponent in K worlds
//!   ChildPolicy  (deep search only) both sides' policy after the first round, to play a second one
//!   Values       value of every resulting position
//! Q(a) = mean value after playing a;  pi_t(a) ~ prior(a) * exp(Q(a) / tau).
//!
//! Playout-cap randomisation (KataGo): a share of the decisions is searched deep (more worlds, two
//! rounds) and only those train the policy (weight pw = 1); the cheap ones still give value targets.

use crate::data::gd;
use crate::engine::*;
use crate::features::{self, action_mask, encode, Obs, PHASE_LEAD, PHASE_REPLACE, PHASE_ROUND};
use crate::rng::PyRandom;
use std::collections::VecDeque;
use std::sync::Mutex;

pub const MAX_ENV_ROUNDS: i64 = 100;

#[derive(Clone)]
pub struct Req {
    pub obs: Obs,
    pub mask: [f32; 7],
}

#[derive(Clone, Copy)]
pub struct Reply {
    pub pol: [f32; 7],
    pub val: f32,
    pub lo: [f32; 9],
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Root,
    ChildPolicy,
    Values,
    Done,
}

pub fn phase_of(b: &Battle) -> usize {
    if b.needs_replacement(0) || b.needs_replacement(1) { PHASE_REPLACE } else { PHASE_ROUND }
}

fn pending_sides(b: &Battle) -> Vec<usize> {
    if b.over() {
        return vec![];
    }
    if phase_of(b) == PHASE_REPLACE {
        return (0..2).filter(|&s| b.needs_replacement(s)).collect();
    }
    vec![0, 1]
}

/// search._apply
pub fn apply(b: &mut Battle, phase: usize, acts: &[Option<usize>; 2]) {
    if phase == PHASE_LEAD {
        for s in 0..2 {
            if let Some(a) = acts[s] {
                b.teams[s].active = a - 3;
            }
        }
        return;
    }
    let mut full = [A_PASS; 2];
    for s in 0..2 {
        let a = acts[s].unwrap_or(A_PASS);
        let legal = b.legal_actions(s);
        full[s] = if legal.contains(&a) { a } else if b.needs_replacement(s) { legal[0] } else { A_PASS };
    }
    b.step(full[0], full[1]);
}

/// search.determinize: the opponent's unrevealed ability choices sampled from lo (P(second option)).
fn determinize(b: &mut Battle, side: usize, lo: &[f32; 9], rng: &mut PyRandom) {
    let es = 1 - side;
    for pi in 0..b.teams[es].pets.len() {
        let species = b.teams[es].pets[pi].species;
        let opts = &gd().species[&species].abilities;
        for sl in 0..3 {
            let p = &b.teams[es].pets[pi];
            if p.revealed & (1 << sl) != 0 || opts.get(sl).map(|o| o.len() < 2).unwrap_or(true) {
                continue;
            }
            let second = rng.random() < lo[pi * 3 + sl] as f64;
            b.teams[es].pets[pi].abilities[sl] = if second { opts[sl][1] } else { opts[sl][0] };
        }
    }
}

fn sample(p: &[f32; 7], rng: &mut PyRandom) -> usize {
    let total: f64 = p.iter().map(|&x| x.max(0.0) as f64).sum();
    if total <= 0.0 {
        return A_PASS;
    }
    let mut r = rng.random() * total;
    for (i, &x) in p.iter().enumerate() {
        r -= x.max(0.0) as f64;
        if r < 0.0 {
            return i;
        }
    }
    (0..7).rev().find(|&i| p[i] > 0.0).unwrap_or(A_PASS)
}

pub struct Search {
    pub side: usize,
    pub phase: usize,
    pub root: Req,
    pub legal: Vec<usize>,
    pub depth: usize,
    tau: f32,
    worlds: Vec<Battle>,
    opp_acts: bool,
    states: Vec<Battle>,
    tags: Vec<usize>,
    step_reqs: Vec<(usize, usize)>, // (state index, side) of a ChildPolicy stage
    value_idx: Vec<usize>,
    stage: Stage,
    rounds_left: usize,
    pub prior: [f32; 7],
    pub value: f32,
    pub lo: [f32; 9],
    pub pi: [f32; 7],
    pub q: [f32; 7],
    pub n: [u32; 7],
    pub se: [f32; 7],
    pub chosen: usize,
    pub deep: bool,
    pub reqs: Vec<Req>,
}

impl Search {
    pub fn new(b: &Battle, side: usize, phase: usize, k: usize, depth: usize, tau: f32, lo_prev: Option<[f32; 9]>, rng: &mut PyRandom, deep: bool) -> Search {
        let root = Req { obs: encode(b, side, phase, true), mask: action_mask(b, side, phase) };
        let legal: Vec<usize> = (0..7).filter(|&a| root.mask[a] > 0.0).collect();
        let opp_acts = phase != PHASE_REPLACE || b.needs_replacement(1 - side);
        let mut worlds = Vec::new();
        let mut reqs = vec![root.clone()];
        if legal.len() > 1 {
            let lo = lo_prev.unwrap_or([0.5; 9]);
            for _ in 0..k {
                let mut w = b.clone_with_seed(rng.getrandbits(32));
                determinize(&mut w, side, &lo, rng);
                if opp_acts {
                    reqs.push(Req { obs: encode(&w, 1 - side, phase, true), mask: action_mask(&w, 1 - side, phase) });
                }
                worlds.push(w);
            }
        }
        Search {
            side,
            phase,
            root,
            legal,
            depth,
            tau,
            worlds,
            opp_acts,
            states: Vec::new(),
            tags: Vec::new(),
            step_reqs: Vec::new(),
            value_idx: Vec::new(),
            stage: Stage::Root,
            rounds_left: depth.saturating_sub(1),
            prior: [0.0; 7],
            value: 0.0,
            lo: [0.5; 9],
            pi: [0.0; 7],
            q: [0.0; 7],
            n: [0; 7],
            se: [0.0; 7],
            chosen: A_PASS,
            deep,
            reqs,
        }
    }

    pub fn done(&self) -> bool {
        self.stage == Stage::Done
    }

    /// Feed the replies to the current requests (same order) and move on.
    pub fn advance(&mut self, replies: &[Reply], rng: &mut PyRandom) {
        match self.stage {
            Stage::Root => {
                let r0 = replies[0];
                self.prior = r0.pol;
                self.value = r0.val;
                self.lo = r0.lo;
                if self.legal.len() > 1 && self.worlds.is_empty() {
                    // no look-ahead (k = 0, the cheap decisions): play the network's own policy
                    self.pi = self.prior;
                    self.chosen = sample(&self.prior, rng);
                    self.finish_done();
                    return;
                }
                if self.legal.len() <= 1 {
                    self.pi = [0.0; 7];
                    self.chosen = *self.legal.first().unwrap_or(&A_PASS);
                    self.pi[self.chosen] = 1.0;
                    self.finish_done();
                    return;
                }
                let worlds = std::mem::take(&mut self.worlds);
                for (j, w) in worlds.iter().enumerate() {
                    let b_j = if self.opp_acts { Some(sample(&replies[1 + j].pol, rng)) } else { None };
                    // common random numbers: every action of a world gets the same reply and dice
                    let seed_j = rng.getrandbits(32);
                    for &a in &self.legal {
                        let mut c = w.clone_with_seed(seed_j);
                        let mut acts = [None, None];
                        acts[self.side] = Some(a);
                        acts[1 - self.side] = b_j;
                        apply(&mut c, self.phase, &acts);
                        self.states.push(c);
                        self.tags.push(a);
                    }
                }
                self.next_stage(rng);
            }
            Stage::ChildPolicy => {
                let mut acts: Vec<[Option<usize>; 2]> = vec![[None, None]; self.states.len()];
                for (&(i, s), r) in self.step_reqs.iter().zip(replies) {
                    acts[i][s] = Some(sample(&r.pol, rng));
                }
                let reqs = std::mem::take(&mut self.step_reqs);
                let mut touched = vec![false; self.states.len()];
                for &(i, _) in &reqs {
                    touched[i] = true;
                }
                for (i, st) in self.states.iter_mut().enumerate() {
                    if touched[i] {
                        let ph = phase_of(st);
                        apply(st, ph, &acts[i]);
                    }
                }
                self.rounds_left -= 1;
                self.next_stage(rng);
            }
            Stage::Values => {
                let mut vals = vec![0f32; self.states.len()];
                for (i, st) in self.states.iter().enumerate() {
                    if st.over() {
                        vals[i] = match st.winner {
                            Some(w) if w == self.side as i64 => 1.0,
                            Some(w) if w == 1 - self.side as i64 => -1.0,
                            _ => 0.0,
                        };
                    }
                }
                for (&i, r) in self.value_idx.iter().zip(replies) {
                    vals[i] = r.val;
                }
                let mut sum = [0f64; 7];
                let mut sq = [0f64; 7];
                let mut n = [0u32; 7];
                for (&a, &v) in self.tags.iter().zip(&vals) {
                    sum[a] += v as f64;
                    sq[a] += (v as f64) * (v as f64);
                    n[a] += 1;
                }
                let mut logits = [f64::NEG_INFINITY; 7];
                for &a in &self.legal {
                    if n[a] > 0 {
                        let m = sum[a] / n[a] as f64;
                        self.q[a] = m as f32;
                        let var = (sq[a] / n[a] as f64 - m * m).max(0.0);
                        self.se[a] = (var.sqrt() / (n[a] as f64).sqrt()) as f32;
                    }
                    self.n[a] = n[a];
                    logits[a] = (self.prior[a].max(1e-8) as f64).ln() + self.q[a] as f64 / self.tau as f64;
                }
                let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut z = 0.0;
                let mut p = [0f64; 7];
                for a in 0..7 {
                    if logits[a].is_finite() {
                        p[a] = (logits[a] - mx).exp();
                        z += p[a];
                    }
                }
                for a in 0..7 {
                    self.pi[a] = (p[a] / z) as f32;
                }
                self.chosen = sample(&self.pi, rng);
                self.finish_done();
            }
            Stage::Done => {}
        }
    }

    fn finish_done(&mut self) {
        self.stage = Stage::Done;
        self.reqs.clear();
        self.states.clear();
        self.worlds.clear();
    }

    fn next_stage(&mut self, rng: &mut PyRandom) {
        if self.rounds_left > 0 {
            self.step_reqs.clear();
            self.reqs.clear();
            for (i, st) in self.states.iter().enumerate() {
                if st.round >= MAX_ENV_ROUNDS {
                    continue;
                }
                let ph = phase_of(st);
                for s in pending_sides(st) {
                    self.reqs.push(Req { obs: encode(st, s, ph, true), mask: action_mask(st, s, ph) });
                    self.step_reqs.push((i, s));
                }
            }
            if !self.reqs.is_empty() {
                self.stage = Stage::ChildPolicy;
                return;
            }
            self.rounds_left = 0;
        }
        self.value_idx.clear();
        self.reqs.clear();
        for (i, st) in self.states.iter().enumerate() {
            if !st.over() {
                let ph = phase_of(st);
                self.reqs.push(Req { obs: encode(st, self.side, ph, true), mask: action_mask(st, self.side, ph) });
                self.value_idx.push(i);
            }
        }
        self.stage = Stage::Values;
        if self.reqs.is_empty() {
            self.advance(&[], rng);
        }
    }
}

// ------------------------------------------------------------------------------------ actor pool
#[derive(Clone)]
pub struct Sample {
    pub obs: Obs,
    pub mask: [f32; 7],
    pub pi: [f32; 7],
    pub q: [f32; 7],
    pub rnd: i64,
    pub opp_act: i64,
    pub pw: f32,
    pub z: f32,
    pub zd: f32,
    pub lo_y: [f32; 9],
}

pub struct Start {
    pub battle: Battle,
    pub lead: bool,
}

#[derive(Clone, Copy)]
pub struct Cfg {
    pub k_deep: usize,
    pub k_cheap: usize,
    pub depth_deep: usize,
    pub deep_frac: f64,
    pub tau: f32,
    pub gamma: f64,
}

pub struct Env {
    b: Option<Battle>,
    phase: usize,
    searches: Vec<Search>,
    traj: [Vec<Sample>; 2],
    lo_prev: [Option<[f32; 9]>; 2],
    rng: PyRandom,
}

fn loadout_truth(b: &Battle, side: usize) -> [f32; 9] {
    let mut y = [0f32; 9];
    let es = 1 - side;
    for (pi, p) in b.teams[es].pets.iter().enumerate() {
        let opts = &gd().species[&p.species].abilities;
        for sl in 0..3 {
            let aid = p.abilities[sl];
            if aid != 0 && opts.get(sl).map(|o| o.len() >= 2).unwrap_or(false) {
                y[pi * 3 + sl] = (aid == opts[sl][1]) as i32 as f32;
            }
        }
    }
    y
}

impl Env {
    pub fn new(seed: u64) -> Env {
        Env { b: None, phase: PHASE_LEAD, searches: Vec::new(), traj: [Vec::new(), Vec::new()], lo_prev: [None, None], rng: PyRandom::new(seed) }
    }

    fn settle(&mut self) -> bool {
        let b = self.b.as_ref().unwrap();
        if b.over() || b.round >= MAX_ENV_ROUNDS {
            return true;
        }
        self.phase = phase_of(b);
        false
    }

    fn pending(&self) -> Vec<usize> {
        let b = self.b.as_ref().unwrap();
        if self.phase == PHASE_REPLACE {
            return (0..2).filter(|&s| b.needs_replacement(s)).collect();
        }
        vec![0, 1]
    }

    fn start_searches(&mut self, cfg: &Cfg) {
        let b = self.b.as_ref().unwrap();
        let sides = self.pending();
        self.searches.clear();
        for s in sides {
            let deep = self.rng.random() < cfg.deep_frac;
            let (k, d) = if deep { (cfg.k_deep, cfg.depth_deep) } else { (cfg.k_cheap, 1) };
            let sr = Search::new(b, s, self.phase, k, d, cfg.tau, self.lo_prev[s], &mut self.rng, deep);
            self.searches.push(sr);
        }
    }

    /// PvPEnv.act
    fn act(&mut self, acts: &[Option<usize>; 2]) {
        let b = self.b.as_mut().unwrap();
        if self.phase == PHASE_LEAD {
            for s in 0..2 {
                if let Some(a) = acts[s] {
                    let idx = a as i64 - 3;
                    if 0 <= idx && (idx as usize) < b.teams[s].pets.len() {
                        b.teams[s].active = idx as usize;
                    }
                }
            }
            self.phase = PHASE_ROUND;
            return;
        }
        apply(b, self.phase, acts);
    }

    fn end_game(&mut self, cfg: &Cfg, out: &mut Vec<Sample>, games: &mut Vec<(i64, i64)>) {
        let b = self.b.take().unwrap();
        let w = match b.winner {
            Some(0) => 0,
            Some(1) => 1,
            _ => -1,
        };
        let end = b.round;
        for side in 0..2 {
            let y = loadout_truth(&b, side);
            let z: f32 = if w == -1 { 0.0 } else if w == side as i64 { 1.0 } else { -1.0 };
            for mut s in self.traj[side].drain(..) {
                s.z = z;
                s.lo_y = y;
                s.zd = (z as f64 * cfg.gamma.powi((end - s.rnd).max(0) as i32)) as f32;
                out.push(s);
            }
        }
        games.push((w, end));
        self.lo_prev = [None, None];
        self.searches.clear();
    }

    /// One step of work after the network replied (or at start). Returns when the env waits for
    /// the network again, or has nothing to do (no start available).
    fn work(&mut self, replies: &[Reply], cfg: &Cfg, starts: &Mutex<VecDeque<Start>>, out: &mut Vec<Sample>, games: &mut Vec<(i64, i64)>) {
        // hand the replies to the searches, in request order
        let mut off = 0;
        for sr in self.searches.iter_mut() {
            let n = sr.reqs.len();
            if n > 0 && off + n <= replies.len() {
                let rs = &replies[off..off + n];
                off += n;
                sr.advance(rs, &mut self.rng);
            }
        }
        loop {
            if self.b.is_none() {
                let st = starts.lock().unwrap().pop_front();
                match st {
                    None => return,
                    Some(st) => {
                        self.b = Some(st.battle);
                        self.phase = if st.lead { PHASE_LEAD } else { PHASE_ROUND };
                        if !st.lead && self.settle() {
                            self.end_game(cfg, out, games);
                            continue;
                        }
                        self.start_searches(cfg);
                        return;
                    }
                }
            }
            if self.searches.is_empty() {
                self.start_searches(cfg);
                return;
            }
            if !self.searches.iter().all(|s| s.done()) {
                return;
            }
            // every side decided: record and act
            let mut acts = [None, None];
            for sr in &self.searches {
                acts[sr.side] = Some(sr.chosen);
            }
            let rnd = self.b.as_ref().unwrap().round;
            let round_phase = self.phase == PHASE_ROUND;
            let srs = std::mem::take(&mut self.searches);
            for sr in srs {
                self.lo_prev[sr.side] = Some(sr.lo);
                let opp = if round_phase { acts[1 - sr.side].map(|a| a as i64).unwrap_or(-1) } else { -1 };
                self.traj[sr.side].push(Sample {
                    obs: sr.root.obs,
                    mask: sr.root.mask,
                    pi: sr.pi,
                    q: sr.q,
                    rnd,
                    opp_act: opp,
                    pw: sr.deep as i32 as f32,
                    z: 0.0,
                    zd: 0.0,
                    lo_y: [0.0; 9],
                });
            }
            self.act(&acts);
            if self.phase != PHASE_LEAD && self.settle() {
                self.end_game(cfg, out, games);
                continue;
            }
            self.start_searches(cfg);
            return;
        }
    }

    pub fn n_reqs(&self) -> usize {
        self.searches.iter().map(|s| s.reqs.len()).sum()
    }

    pub fn reqs(&self) -> impl Iterator<Item = &Req> {
        self.searches.iter().flat_map(|s| s.reqs.iter())
    }
}

pub struct Pool {
    pub envs: Vec<Env>,
    pub cfg: Cfg,
    pub starts: Mutex<VecDeque<Start>>,
    pub samples: Vec<Sample>,
    pub games: Vec<(i64, i64)>,
    last_counts: Vec<usize>,
}

impl Pool {
    pub fn new(n_envs: usize, seed: u64, cfg: Cfg) -> Pool {
        Pool {
            envs: (0..n_envs).map(|i| Env::new(seed.wrapping_mul(1_000_003).wrapping_add(i as u64))).collect(),
            cfg,
            starts: Mutex::new(VecDeque::new()),
            samples: Vec::new(),
            games: Vec::new(),
            last_counts: vec![0; n_envs],
        }
    }

    /// Advance every env with its share of `replies` (empty at the very start), in parallel.
    pub fn work(&mut self, replies: &[Reply]) {
        use rayon::prelude::*;
        let mut chunks: Vec<&[Reply]> = Vec::with_capacity(self.envs.len());
        let mut off = 0;
        for &n in &self.last_counts {
            let end = (off + n).min(replies.len());
            chunks.push(&replies[off..end]);
            off = end;
        }
        let cfg = self.cfg;
        let starts = &self.starts;
        let results: Vec<(Vec<Sample>, Vec<(i64, i64)>)> = self
            .envs
            .par_iter_mut()
            .zip(chunks.into_par_iter())
            .map(|(e, rs)| {
                let mut out = Vec::new();
                let mut games = Vec::new();
                e.work(rs, &cfg, starts, &mut out, &mut games);
                (out, games)
            })
            .collect();
        for (s, g) in results {
            self.samples.extend(s);
            self.games.extend(g);
        }
        self.last_counts = self.envs.iter().map(|e| e.n_reqs()).collect();
    }

    pub fn total_reqs(&self) -> usize {
        self.last_counts.iter().sum()
    }
}

/// Identical requests (same position seen by the same side: worlds without hidden abilities,
/// rounds without dice) are sent to the network once. -> (unique requests, index of each request
/// into the unique list)
pub fn dedup<'a>(reqs: impl Iterator<Item = &'a Req>) -> (Vec<&'a Req>, Vec<u32>) {
    use std::collections::HashMap;
    let mut uniq: Vec<&'a Req> = Vec::new();
    let mut by_hash: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut map = Vec::new();
    for r in reqs {
        let h = r.obs.hash(&r.mask);
        let cands = by_hash.entry(h).or_default();
        let found = cands.iter().copied().find(|&i| {
            let u = uniq[i as usize];
            u.mask == r.mask && u.obs == r.obs
        });
        let idx = match found {
            Some(i) => i,
            None => {
                uniq.push(r);
                let i = (uniq.len() - 1) as u32;
                cands.push(i);
                i
            }
        };
        map.push(idx);
    }
    (uniq, map)
}

// ------------------------------------------------------------------------------------ helpers for python
pub fn stack_obs<'a>(reqs: impl Iterator<Item = &'a Req>, n: usize) -> ObsBatch {
    let mut ob = ObsBatch::with_capacity(n);
    for r in reqs {
        ob.push(&r.obs, &r.mask);
    }
    ob
}

pub struct ObsBatch {
    pub n: usize,
    pub glob: Vec<f32>,
    pub pet_int: Vec<i64>,
    pub pet_float: Vec<f32>,
    pub ab_int: Vec<i64>,
    pub ab_float: Vec<f32>,
    pub ab_mask: Vec<f32>,
    pub aura_int: Vec<i64>,
    pub aura_float: Vec<f32>,
    pub aura_mask: Vec<f32>,
    pub mask: Vec<f32>,
}

impl ObsBatch {
    pub fn with_capacity(n: usize) -> ObsBatch {
        ObsBatch {
            n: 0,
            glob: Vec::with_capacity(n * features::N_GLOB),
            pet_int: Vec::with_capacity(n * 6),
            pet_float: Vec::with_capacity(n * 6 * features::N_PET),
            ab_int: Vec::with_capacity(n * features::N_ABIL * 3),
            ab_float: Vec::with_capacity(n * features::N_ABIL * features::N_ABF),
            ab_mask: Vec::with_capacity(n * features::N_ABIL),
            aura_int: Vec::with_capacity(n * features::MAX_AURAS * 2),
            aura_float: Vec::with_capacity(n * features::MAX_AURAS * features::N_AUF),
            aura_mask: Vec::with_capacity(n * features::MAX_AURAS),
            mask: Vec::with_capacity(n * 7),
        }
    }
    pub fn push(&mut self, o: &Obs, m: &[f32; 7]) {
        self.n += 1;
        self.glob.extend_from_slice(&o.glob);
        self.pet_int.extend_from_slice(&o.pet_int);
        for r in &o.pet_float {
            self.pet_float.extend_from_slice(r);
        }
        for r in &o.ab_int {
            self.ab_int.extend_from_slice(r);
        }
        for r in &o.ab_float {
            self.ab_float.extend_from_slice(r);
        }
        self.ab_mask.extend_from_slice(&o.ab_mask);
        for r in &o.aura_int {
            self.aura_int.extend_from_slice(r);
        }
        for r in &o.aura_float {
            self.aura_float.extend_from_slice(r);
        }
        self.aura_mask.extend_from_slice(&o.aura_mask);
        self.mask.extend_from_slice(m);
    }
}
