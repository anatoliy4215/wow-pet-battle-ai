//! Battle engine: a line-by-line port of bp/sim/battle.py + ops.py. Every rule, order of events and
//! dice draw follows the Python engine, so both give identical battles for the same seed
//! (checked by tests/test_rust_sim.py). Comments name the Python function each part mirrors.

use crate::data::{gd, Ability, Code, Effect, Params, K};
use crate::rng::PyRandom;

// ------------------------------------------------------------------------------------ constants
pub const HUMANOID: i64 = 0;
pub const DRAGONKIN: i64 = 1;
pub const FLYING: i64 = 2;
pub const UNDEAD: i64 = 3;
pub const CRITTER: i64 = 4;
pub const MAGIC: i64 = 5;
pub const ELEMENTAL: i64 = 6;
pub const BEAST: i64 = 7;
pub const AQUATIC: i64 = 8;
pub const MECHANICAL: i64 = 9;
const STRONG: [i64; 10] = [1, 5, 8, 0, 3, 2, 9, 4, 6, 7];
const WEAK: [i64; 10] = [7, 3, 1, 8, 0, 9, 4, 2, 5, 6];
const STRONG_MULT: f64 = 1.5;
const WEAK_MULT: f64 = 0.66;
const PASSIVE_ABILITY: [u32; 10] = [238, 245, 239, 242, 236, 243, 241, 237, 240, 244];
const HUMANOID_RECOVERY_AURA: u32 = 726;
const DRAGONKIN_AURA: u32 = 245;
pub const STUNNED_AURA: u32 = 927;
const DAMNED_ABILITY: u32 = 724;
const FAILSAFE_ABILITY: u32 = 723;
const BOSS_AURA: u32 = 1089;
const ELITE_AURA: u32 = 956;

pub const S_IS_DEAD: i64 = 1;
const S_MAX_HEALTH_BONUS: i64 = 2;
const S_POWER: i64 = 18;
const S_STAMINA: i64 = 19;
const S_SPEED: i64 = 20;
pub const S_STUNNED: i64 = 22;
const S_DMG_DEALT_PCT: i64 = 23;
const S_DMG_TAKEN_PCT: i64 = 24;
const S_SPEED_PCT: i64 = 25;
const S_RAMPING_ID: i64 = 26;
const S_RAMPING_USES: i64 = 27;
const S_WAS_DAMAGED: i64 = 28;
pub const S_UNTARGETTABLE: i64 = 29;
const S_UNDERGROUND: i64 = 30;
const S_LAST_HIT_TAKEN: i64 = 31;
const S_LAST_HIT_DEALT: i64 = 32;
const S_FLYING: i64 = 33;
pub const S_TURN_LOCK: i64 = 35;
pub const S_SWAP_OUT_LOCK: i64 = 36;
const S_CRIT: i64 = 40;
const S_ACCURACY: i64 = 41;
const S_HEAL_DEALT_PCT: i64 = 65;
const S_HEAL_TAKEN_PCT: i64 = 66;
pub const S_UNKILLABLE: i64 = 68;
const S_FLAT_DMG_TAKEN: i64 = 71;
const S_FLAT_DMG_DEALT: i64 = 72;
const S_DODGE: i64 = 73;
const S_PETTYPE_DMG_DEALT: i64 = 87;
const S_PETTYPE_ID: i64 = 89;
const S_SWAP_IN_LOCK: i64 = 98;
const S_MAX_HEALTH_PCT: i64 = 99;
const S_DID_DAMAGE: i64 = 127;
pub const S_RESILIENT: i64 = 149;
const S_BLOCKED_FRIENDLY: i64 = 123;
pub const S_TYPE_OVERRIDE: i64 = 124;
const S_CONSUMED_CORPSE: i64 = 120;
const S_PLANT: i64 = 70;
const S_HEALTH_BEFORE_INSTAKILL: i64 = 145;
const S_ELITE: i64 = 153;
const S_BOSS: i64 = 162;
const S_IGNORE_DMG_BELOW: i64 = 191;
const S_ADD_PERIODIC_DMG_TAKEN: i64 = 202;
const S_CRIT_DMG_BONUS: i64 = 314;
const PASSIVE_STATES: [i64; 10] = [44, 46, 45, 50, 42, 49, 47, 43, 51, 48];
const WEATHER_STATES: [i64; 12] = [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 316];

const EV_APPLY: i64 = 0;
const EV_DAMAGE_TAKEN: i64 = 1;
const EV_DAMAGE_DEALT: i64 = 2;
const EV_REMOVE: i64 = 5;
const EV_ROUND_START: i64 = 6;
const EV_ROUND_END: i64 = 7;
const EV_ABILITY: i64 = 9;
const EV_SWAP_IN: i64 = 10;
const EV_SWAP_OUT: i64 = 11;
const EV_POST_TICK: i64 = 12;

const BASE_CRIT: f64 = 5.0;
const CRIT_MULT: f64 = 1.5;
const LEVEL_HIT_PENALTY: i64 = 2;
const MAGIC_CAP: f64 = 0.35;
const MAX_DMG_REDUCTION: f64 = 0.75;
const BEAST_BONUS: i64 = 25;
const FLYING_SPEED_PCT: i64 = 50;
const AQUATIC_DOT_TAKEN: f64 = 0.5;
const AQUATIC_DOT_DEALT: f64 = 1.25;
pub const MAX_ROUNDS: i64 = 150;
pub const A_PASS: usize = 6;

// ------------------------------------------------------------------------------------ small maps
/// `pet.states` (a dict in Python: key presence matters, values are ints).
#[derive(Clone, Default)]
pub struct States(pub Vec<(i64, i64)>);

impl States {
    #[inline]
    pub fn get(&self, k: i64) -> Option<i64> {
        self.0.iter().find(|(a, _)| *a == k).map(|(_, v)| *v)
    }
    #[inline]
    pub fn g0(&self, k: i64) -> i64 {
        self.get(k).unwrap_or(0)
    }
    #[inline]
    pub fn set(&mut self, k: i64, v: i64) {
        if let Some(e) = self.0.iter_mut().find(|(a, _)| *a == k) {
            e.1 = v;
        } else {
            self.0.push((k, v));
        }
    }
    #[inline]
    pub fn pop(&mut self, k: i64) -> Option<i64> {
        let i = self.0.iter().position(|(a, _)| *a == k)?;
        Some(self.0.remove(i).1)
    }
}

// ------------------------------------------------------------------------------------ battle state
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PetRef {
    pub side: usize,
    pub idx: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Holder {
    Pet(PetRef),
    Team(usize),
    Weather,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tgt {
    Pet(PetRef),
    Team(usize),
}

#[derive(Clone)]
pub struct Aura {
    pub aid: u32,
    pub caster: Option<PetRef>,
    pub duration: i64,
    pub fresh: bool,
    pub stacks: i64,
    pub uid: u64,
    pub age: i64,
    pub next_attack: bool,
    pub blocks_left: i64,
}

#[derive(Clone)]
pub struct Pet {
    pub species: u32,
    pub level: i64,
    pub family: i64,
    pub base_max_hp: i64,
    pub hp: i64,
    pub base_power: i64,
    pub base_speed: i64,
    pub abilities: [u32; 3],
    pub cooldowns: [i64; 3],
    pub auras: Vec<Aura>,
    pub states: States,
    pub dead: bool,
    pub mech_revived: bool,
    pub undead_used: bool,
    pub channel: Option<(usize, usize)>,
    pub dealt_damage_round: bool,
    pub dragon_trigger: bool,
    pub swapped_in: i64,
    pub last_ability: Option<u32>,
    pub revealed: u8,
    pub using: Option<u32>,
}

#[derive(Clone)]
pub struct Team {
    pub side: usize,
    pub pets: Vec<Pet>,
    pub active: usize,
    pub auras: Vec<Aura>,
}

/// What Python passes in: species, abilities (0 = none), level, health / power / speed, extra states.
#[derive(Clone)]
pub struct Spec {
    pub species: u32,
    pub abilities: [u32; 3],
    pub level: i64,
    pub stats: (i64, i64, i64),
    pub extra_states: Vec<(i64, i64)>,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Ability,
    Channel,
    Swap,
    Pass,
}

#[derive(Clone)]
pub struct Battle {
    pub rng: PyRandom,
    pub teams: [Team; 2],
    pub weather: Option<Aura>,
    pub round: i64,
    pub winner: Option<i64>,
    pub logging: bool,
    pub log: Vec<String>,
    pub uid: u64,
    pub unknown_ops: Vec<(i64, i64)>,
    pub force_hit: Option<bool>,
    pub force_crit: Option<bool>,
    pub force_variance: Option<f64>,
    pub force_first: Option<usize>,
    pub first_side: Option<usize>,
    actors: Option<[PetRef; 2]>,
    fire_depth: u32,
    // bp.pvp.probe support
    pub probe_counts: Option<Vec<(u8, usize, i64)>>,
    pub probe_expect: bool,
    hit_scale: Option<f64>,
}

#[derive(Clone)]
struct Ctx {
    me: PetRef,
    target: Option<PetRef>,
    caster: PetRef,
    family: i64,
    ability_id: u32,
    aura: Option<u64>,
    periodic: bool,
    failed: bool,
    missed: bool,
    stop: bool,
    first: bool,
    last: bool,
    in_trigger: bool,
    row_target: Option<Tgt>,
    blocked: bool,
}

impl Ctx {
    fn new(me: PetRef, target: Option<PetRef>, caster: PetRef, family: i64, ability_id: u32) -> Ctx {
        Ctx {
            me,
            target,
            caster,
            family,
            ability_id,
            aura: None,
            periodic: false,
            failed: false,
            missed: false,
            stop: false,
            first: false,
            last: false,
            in_trigger: false,
            row_target: None,
            blocked: false,
        }
    }
}

pub const PROBE_UNREACHABLE: u8 = 0;
pub const PROBE_BLOCKED: u8 = 1;
pub const PROBE_IMMUNE: u8 = 2;
pub const PROBE_ROLLED: u8 = 3;

macro_rules! emit {
    ($b:expr, $($arg:tt)*) => {
        if $b.logging {
            let m = format!($($arg)*);
            let line = format!("[{}] {}", $b.round, m);
            $b.log.push(line);
        }
    };
}

#[inline]
fn ab(aid: u32) -> &'static Ability {
    gd().ability(aid)
}

#[inline]
fn aname(aid: u32) -> &'static str {
    gd().ab(aid).map(|a| a.name.as_str()).unwrap_or("?")
}

fn family_mult(att: i64, def: i64) -> f64 {
    if att < 0 {
        return 1.0;
    }
    if STRONG[att as usize] == def {
        return STRONG_MULT;
    }
    if WEAK[att as usize] == def {
        return WEAK_MULT;
    }
    1.0
}

fn duration_of(p: &Params, op: i64) -> i64 {
    let d = p.gd(K::Duration, 0);
    if (op == 28 || op == 150) && d == 0 {
        return -1;
    }
    d
}

/// Python `int(x)` for a float.
#[inline]
fn pint(x: f64) -> i64 {
    x as i64
}

impl Battle {
    // ---------------------------------------------------------------------------- construction
    pub fn new(team_a: &[Spec], team_b: &[Spec], seed: u64, log: bool) -> Battle {
        let make = |side: usize, specs: &[Spec]| Team {
            side,
            pets: specs
                .iter()
                .map(|s| {
                    let sp = &gd().species[&s.species];
                    let mut st = States::default();
                    for &(k, v) in &s.extra_states {
                        st.set(k, v);
                    }
                    Pet {
                        species: s.species,
                        level: s.level,
                        family: sp.family,
                        base_max_hp: s.stats.0,
                        hp: s.stats.0,
                        base_power: s.stats.1,
                        base_speed: s.stats.2,
                        abilities: s.abilities,
                        cooldowns: [0; 3],
                        auras: Vec::new(),
                        states: st,
                        dead: false,
                        mech_revived: false,
                        undead_used: false,
                        channel: None,
                        dealt_damage_round: false,
                        dragon_trigger: false,
                        swapped_in: 0,
                        last_ability: None,
                        revealed: 0,
                        using: None,
                    }
                })
                .collect(),
            active: 0,
            auras: Vec::new(),
        };
        let mut b = Battle {
            rng: PyRandom::new(seed),
            teams: [make(0, team_a), make(1, team_b)],
            weather: None,
            round: 0,
            winner: None,
            logging: log,
            log: Vec::new(),
            uid: 0,
            unknown_ops: Vec::new(),
            force_hit: None,
            force_crit: None,
            force_variance: None,
            force_first: None,
            first_side: None,
            actors: None,
            fire_depth: 0,
            probe_counts: None,
            probe_expect: false,
            hit_scale: None,
        };
        for r in b.all_refs() {
            let sp = &gd().species[&b.pet(r).species];
            let has = |sid: i64| sp.states.iter().any(|(k, v)| *k == sid && *v != 0);
            if has(S_BOSS) {
                b.add_aura(Holder::Pet(r), BOSS_AURA, Some(r), -1, 0, false);
            } else if has(S_ELITE) {
                b.add_aura(Holder::Pet(r), ELITE_AURA, Some(r), -1, 0, false);
            }
        }
        b
    }

    /// fast_clone: independent copy with fresh dice.
    pub fn clone_with_seed(&self, seed: u64) -> Battle {
        let mut c = self.clone();
        c.rng = PyRandom::new(seed);
        if !c.logging {
            c.log.clear();
        }
        c
    }

    // ---------------------------------------------------------------------------- accessors
    #[inline]
    pub fn pet(&self, r: PetRef) -> &Pet {
        &self.teams[r.side].pets[r.idx]
    }
    #[inline]
    pub fn pet_mut(&mut self, r: PetRef) -> &mut Pet {
        &mut self.teams[r.side].pets[r.idx]
    }
    #[inline]
    pub fn active(&self, side: usize) -> PetRef {
        PetRef { side, idx: self.teams[side].active }
    }
    pub fn all_refs(&self) -> Vec<PetRef> {
        let mut v = Vec::with_capacity(6);
        for side in 0..2 {
            for idx in 0..self.teams[side].pets.len() {
                v.push(PetRef { side, idx });
            }
        }
        v
    }
    #[inline]
    pub fn over(&self) -> bool {
        self.winner.is_some()
    }
    #[inline]
    pub fn name(&self, r: PetRef) -> &'static str {
        gd().species.get(&self.pet(r).species).map(|s| s.name.as_str()).unwrap_or("?")
    }
    fn holder_name(&self, h: Holder) -> String {
        match h {
            Holder::Pet(r) => self.name(r).to_string(),
            Holder::Team(s) => format!("team {}", s),
            Holder::Weather => "?".to_string(),
        }
    }
    fn next_uid(&mut self) -> u64 {
        self.uid += 1;
        self.uid
    }
    fn probe_count(&mut self, kind: u8, r: PetRef) {
        if let Some(c) = self.probe_counts.as_mut() {
            match c.iter_mut().find(|(k, s, _)| *k == kind && *s == r.side) {
                Some(x) => x.2 += 1,
                None => c.push((kind, r.side, 1)),
            }
        }
    }

    /// Pet.state
    pub fn state(&self, r: PetRef, sid: i64) -> i64 {
        let p = self.pet(r);
        let mut v = p.states.g0(sid);
        if sid != S_POWER && sid != S_SPEED && sid != S_STAMINA {
            if let Some(sp) = gd().species.get(&p.species) {
                for &(k, x) in &sp.states {
                    if k == sid {
                        v += x;
                        break;
                    }
                }
            }
        }
        for a in &p.auras {
            v += ab(a.aid).state(sid) * a.stacks;
        }
        if self.teams[r.side].active == r.idx {
            for a in &self.teams[r.side].auras {
                v += ab(a.aid).state(sid);
            }
        }
        if let Some(w) = &self.weather {
            if let Some(x) = ab(w.aid).has_state(sid) {
                if !(p.family == ELEMENTAL && x < 0 && !WEATHER_STATES.contains(&sid)) {
                    v += x;
                }
            }
        }
        if p.family == BEAST && sid == S_DMG_DEALT_PCT && p.hp * 2 < self.max_hp(r) {
            v += BEAST_BONUS;
        }
        if p.family == FLYING && sid == S_SPEED_PCT && p.hp * 2 > self.max_hp(r) {
            v += FLYING_SPEED_PCT;
        }
        v
    }
    pub fn max_hp(&self, r: PetRef) -> i64 {
        let pct = self.state(r, S_MAX_HEALTH_PCT);
        let base = self.pet(r).base_max_hp;
        (pint((base * (100 + pct)) as f64 / 100.0) + self.state(r, S_MAX_HEALTH_BONUS)).max(1)
    }
    pub fn power(&self, r: PetRef) -> i64 {
        (self.pet(r).base_power + self.state(r, S_POWER)).max(0)
    }
    pub fn speed(&self, r: PetRef) -> i64 {
        let s = self.pet(r).base_speed + self.state(r, S_SPEED);
        pint((s * (100 + self.state(r, S_SPEED_PCT))) as f64 / 100.0).max(0)
    }
    fn has_aura(&self, r: PetRef, aid: u32) -> bool {
        self.pet(r).auras.iter().any(|a| a.aid == aid)
    }
    fn alive_count(&self, side: usize) -> usize {
        self.teams[side].pets.iter().filter(|p| !p.dead).count()
    }

    // ---------------------------------------------------------------------------- legality
    pub fn needs_replacement(&self, side: usize) -> bool {
        let t = &self.teams[side];
        t.pets[t.active].dead && self.alive_count(side) > 0
    }

    fn free_corpse(&self, r: PetRef) -> Option<usize> {
        self.teams[r.side]
            .pets
            .iter()
            .enumerate()
            .find(|(i, p)| p.dead && *i != r.idx && p.states.g0(S_CONSUMED_CORPSE) == 0)
            .map(|(i, _)| i)
    }

    fn ability_usable(&self, r: PetRef, aid: u32) -> bool {
        !(ab(aid).needs_corpse && self.free_corpse(r).is_none())
    }

    pub fn legal_actions(&self, side: usize) -> Vec<usize> {
        let t = &self.teams[side];
        let r = self.active(side);
        if self.needs_replacement(side) {
            return (0..t.pets.len()).filter(|&i| !t.pets[i].dead).map(|i| 3 + i).collect();
        }
        let pet = self.pet(r);
        if pet.channel.is_some() {
            return vec![A_PASS];
        }
        let locked = self.state(r, S_TURN_LOCK) > 0;
        let mut acts = Vec::with_capacity(7);
        for i in 0..3 {
            let aid = pet.abilities[i];
            if !locked && aid != 0 && pet.cooldowns[i] == 0 && self.ability_usable(r, aid) {
                acts.push(i);
            }
        }
        if self.state(r, S_SWAP_OUT_LOCK) <= 0 {
            for i in 0..t.pets.len() {
                if i != t.active && !t.pets[i].dead && self.state(PetRef { side, idx: i }, S_SWAP_IN_LOCK) <= 0 {
                    acts.push(3 + i);
                }
            }
        }
        acts.push(A_PASS);
        acts
    }

    // ---------------------------------------------------------------------------- swapping
    pub fn swap(&mut self, side: usize, idx: usize, forced: bool) {
        let old = self.active(side);
        if idx == self.teams[side].active {
            return;
        }
        if !self.pet(old).dead {
            self.fire(EV_SWAP_OUT, old, None, None, None);
            let p = self.pet_mut(old);
            p.channel = None;
            p.states.pop(S_RAMPING_ID);
            p.states.pop(S_RAMPING_USES);
        }
        self.teams[side].active = idx;
        let new = self.active(side);
        let round = self.round;
        self.pet_mut(new).swapped_in = round;
        emit!(self, "{} swaps to {}{}", if side == 0 { "A" } else { "B" }, self.name(new), if forced { " (forced)" } else { "" });
        self.fire(EV_SWAP_IN, new, None, None, None);
    }

    // ---------------------------------------------------------------------------- round
    pub fn step(&mut self, action_a: usize, action_b: usize) {
        let actions = [action_a, action_b];
        let mut replaced = false;
        for side in 0..2 {
            if self.needs_replacement(side) {
                let mut idx = actions[side] as i64 - 3;
                let n = self.teams[side].pets.len() as i64;
                if !(0 <= idx && idx < n) || self.teams[side].pets[idx as usize].dead {
                    idx = self.teams[side].pets.iter().position(|p| !p.dead).unwrap() as i64;
                }
                self.swap(side, idx as usize, true);
                replaced = true;
            }
        }
        if replaced {
            return;
        }
        self.round += 1;
        for side in 0..2 {
            for p in self.teams[side].pets.iter_mut() {
                p.dealt_damage_round = false;
                p.states.set(S_WAS_DAMAGED, 0);
                p.states.set(S_DID_DAMAGE, 0);
            }
        }
        // 2. swaps
        let mut chosen = [(Kind::Pass, 0usize); 2];
        for side in 0..2 {
            let a = actions[side];
            let r = self.active(side);
            if let Some((slot, _)) = self.pet(r).channel {
                chosen[side] = (Kind::Channel, slot);
            } else if (3..6).contains(&a) {
                self.swap(side, a - 3, false);
                chosen[side] = (Kind::Swap, 0);
            } else if a < 3 && self.state(r, S_TURN_LOCK) <= 0 && self.pet(r).abilities[a] != 0 {
                chosen[side] = (Kind::Ability, a);
            } else {
                chosen[side] = (Kind::Pass, 0);
            }
        }
        let actors = [self.active(0), self.active(1)];
        self.actors = Some(actors);
        // 3. round start triggers
        for side in 0..2 {
            let r = self.active(side);
            let (kind, slot) = chosen[side];
            let ab_id = if kind == Kind::Ability || kind == Kind::Channel { Some(self.pet(r).abilities[slot]) } else { None };
            self.fire(EV_ROUND_START, r, None, ab_id.filter(|&x| x != 0), None);
        }
        // 4. order
        let (pa, pb) = (self.active(0), self.active(1));
        let (sa, sb) = (self.speed(pa), self.speed(pb));
        let order = if sa == sb && self.force_first.is_some() {
            let f = self.force_first.unwrap();
            [f, 1 - f]
        } else if sa != sb {
            if sa > sb { [0, 1] } else { [1, 0] }
        } else if self.rng.random() < 0.5 {
            [0, 1]
        } else {
            [1, 0]
        };
        self.first_side = Some(order[0]);
        for &side in &order {
            let r = self.active(side);
            if self.pet(r).dead || self.over() || r != actors[side] {
                continue;
            }
            let (kind, slot) = chosen[side];
            if kind != Kind::Ability && kind != Kind::Channel {
                continue;
            }
            if self.state(r, S_TURN_LOCK) > 0 && kind == Kind::Ability {
                continue;
            }
            self.pet_mut(r).revealed |= 1 << slot;
            let other = chosen[1 - side].0;
            let idle = other != Kind::Ability && other != Kind::Channel;
            self.use_ability(r, slot, side == order[0] || idle, side == order[1]);
            self.check_end();
        }
        // 5. end of round
        if !self.over() {
            self.end_of_round(&chosen);
        }
        self.check_end();
        self.clear_dead_auras(false);
        if self.round >= MAX_ROUNDS && !self.over() {
            self.winner = Some(-1);
        }
    }

    fn end_of_round(&mut self, chosen: &[(Kind, usize); 2]) {
        let actors = self.actors.unwrap();
        for side in 0..2 {
            let r = self.active(side);
            let (kind, slot) = chosen[side];
            let ab_id = if (kind == Kind::Ability || kind == Kind::Channel) && r == actors[side] { Some(self.pet(r).abilities[slot]) } else { None };
            self.fire(EV_ROUND_END, r, None, ab_id.filter(|&x| x != 0), None);
        }
        if self.over() {
            return;
        }
        self.clear_dead_auras(true);
        self.tick_auras();
        if self.over() {
            return;
        }
        for r in self.all_refs() {
            let (dead, fam, trig) = {
                let p = self.pet(r);
                (p.dead, p.family, p.dragon_trigger)
            };
            if !dead && fam == DRAGONKIN && trig {
                self.add_aura(Holder::Pet(r), DRAGONKIN_AURA, Some(r), 1, 0, false);
            }
            self.pet_mut(r).dragon_trigger = false;
        }
        for side in 0..2 {
            let r = self.active(side);
            let (kind, slot) = chosen[side];
            let ab_id = if (kind == Kind::Ability || kind == Kind::Channel) && r == actors[side] { Some(self.pet(r).abilities[slot]) } else { None };
            self.fire(EV_POST_TICK, r, None, ab_id.filter(|&x| x != 0), None);
        }
        self.countdown_auras();
        for r in self.all_refs() {
            let p = self.pet(r);
            if !p.dead && p.family == HUMANOID && p.dealt_damage_round {
                self.run_passive_ability(r, HUMANOID_RECOVERY_AURA);
            }
        }
        self.clear_dead_auras(false);
        for side in 0..2 {
            for p in self.teams[side].pets.iter_mut() {
                for i in 0..3 {
                    if p.cooldowns[i] > 0 {
                        p.cooldowns[i] -= 1;
                    }
                }
            }
        }
    }

    pub fn check_end(&mut self) {
        if self.winner.is_some() {
            return;
        }
        let a = self.alive_count(0) > 0;
        let b = self.alive_count(1) > 0;
        if !a && !b {
            self.winner = Some(-1);
        } else if !a {
            self.winner = Some(1);
        } else if !b {
            self.winner = Some(0);
        }
    }

    // ---------------------------------------------------------------------------- abilities
    fn use_ability(&mut self, r: PetRef, slot_in: usize, first: bool, last: bool) {
        let aid = self.pet(r).abilities[slot_in];
        let a = ab(aid);
        let (slot, idx) = match self.pet(r).channel {
            Some(ch) => ch,
            None => {
                if a.cooldown != 0 {
                    self.pet_mut(r).cooldowns[slot_in] = a.cooldown + 1;
                }
                (slot_in, 0)
            }
        };
        if a.main.is_empty() {
            return;
        }
        let ti = a.main[idx.min(a.main.len() - 1)];
        self.pet_mut(r).channel = if idx + 1 < a.main.len() { Some((slot, idx + 1)) } else { None };
        self.pet_mut(r).last_ability = Some(aid);
        emit!(self, "{} uses {}", self.name(r), a.name);
        let enemy = self.active(1 - r.side);
        let mut ctx = Ctx::new(r, Some(enemy), r, a.family, aid);
        ctx.first = first;
        ctx.last = last;
        self.pet_mut(r).using = Some(aid);
        self.run_effects(&a.turns[ti].effects, &mut ctx, true);
        self.pet_mut(r).using = None;
        if !self.over() {
            self.fire(EV_ABILITY, r, Some(enemy), None, None);
        }
    }

    fn resolve_target(&self, ctx: &Ctx, code: Option<Code>) -> Option<Tgt> {
        let code = code?;
        let me = ctx.me;
        match code {
            Code::C => Some(Tgt::Pet(ctx.caster)),
            Code::T => ctx.target.map(Tgt::Pet),
            Code::AP => Some(Tgt::Team(me.side)),
            Code::EP => Some(Tgt::Team(1 - me.side)),
            Code::A(n) | Code::E(n) => {
                let side = if matches!(code, Code::A(_)) { me.side } else { 1 - me.side };
                let t = &self.teams[side];
                if n == 0 {
                    return Some(Tgt::Pet(PetRef { side, idx: t.active }));
                }
                let backups: Vec<usize> = (0..t.pets.len()).filter(|&i| i != t.active).collect();
                backups.get(n as usize - 1).map(|&i| Tgt::Pet(PetRef { side, idx: i }))
            }
        }
    }

    /// run_effects; `plan`: use the rows' target plan / chain rows (an ability or aura turn)
    fn run_effects(&mut self, effects: &'static [Effect], ctx: &mut Ctx, plan: bool) {
        for e in effects.iter() {
            if ctx.stop || self.over() {
                break;
            }
            if ctx.failed && (e.p.t(K::ChainFailure) || (plan && e.chain)) {
                continue;
            }
            if !has_handler(e.op) {
                match self.unknown_ops.iter_mut().find(|(k, _)| *k == e.op) { Some(x) => x.1 += 1, None => self.unknown_ops.push((e.op, 1)) }
                continue;
            }
            let code = if plan { e.code } else { None };
            let tgt = self.resolve_target(ctx, code);
            ctx.row_target = tgt;
            let ok = if code.is_some() && tgt.is_none() { Some(false) } else { self.op(e, ctx) };
            if let Some(ok) = ok {
                ctx.failed = !ok;
                ctx.missed = !ok;
            }
        }
    }

    #[inline]
    fn tgt(&self, ctx: &Ctx) -> Option<PetRef> {
        match ctx.row_target {
            Some(Tgt::Team(s)) => Some(self.active(s)),
            Some(Tgt::Pet(r)) => Some(r),
            None => ctx.target,
        }
    }

    // ---------------------------------------------------------------------------- triggers
    fn fire(&mut self, event: i64, pet: PetRef, other: Option<PetRef>, ability_id: Option<u32>, aura_filter: Option<(u32, PetRef, u64)>) {
        if self.over() && event != EV_DAMAGE_TAKEN {
            return;
        }
        if self.fire_depth >= 3 {
            return;
        }
        self.fire_depth += 1;
        self.fire_inner(event, pet, other, ability_id, aura_filter);
        self.fire_depth -= 1;
    }

    fn aura_alive(&self, pet: PetRef, uid: u64) -> bool {
        self.pet(pet).auras.iter().any(|a| a.uid == uid)
            || self.teams[pet.side].auras.iter().any(|a| a.uid == uid)
            || self.weather.as_ref().map(|w| w.uid == uid).unwrap_or(false)
    }

    fn fire_inner(&mut self, event: i64, pet: PetRef, other: Option<PetRef>, ability_id: Option<u32>, aura_filter: Option<(u32, PetRef, u64)>) {
        // (ability, aura uid, caster)
        let mut sources: Vec<(u32, Option<u64>, PetRef)> = Vec::new();
        if let Some((aid, caster, uid)) = aura_filter {
            sources.push((aid, Some(uid), caster));
        } else {
            for a in &self.pet(pet).auras {
                sources.push((a.aid, Some(a.uid), a.caster.unwrap_or(pet)));
            }
            if self.teams[pet.side].active == pet.idx {
                for a in &self.teams[pet.side].auras {
                    sources.push((a.aid, Some(a.uid), a.caster.unwrap_or(pet)));
                }
            }
            if let Some(w) = &self.weather {
                if event != EV_REMOVE {
                    sources.push((w.aid, Some(w.uid), w.caster.unwrap_or(pet)));
                }
            }
            if let Some(aid) = ability_id {
                sources.push((aid, None, pet));
            } else if let Some(u) = self.pet(pet).using {
                if event == EV_DAMAGE_DEALT || event == EV_DAMAGE_TAKEN {
                    sources.push((u, None, pet));
                }
            }
        }
        let first = self.first_side == Some(pet.side);
        for (aid, aura, caster) in sources {
            let a = ab(aid);
            for t in a.turns.iter() {
                if t.typ != 1 || t.event != event {
                    continue;
                }
                if let Some(uid) = aura {
                    if aura_filter.is_none() && !self.aura_alive(pet, uid) {
                        break;
                    }
                }
                let mut tgt = other.unwrap_or_else(|| self.active(1 - pet.side));
                if event == EV_REMOVE && other.is_none() {
                    tgt = pet;
                }
                let mut ctx = Ctx::new(pet, Some(tgt), if aura.is_some() { caster } else { pet }, a.family, aid);
                ctx.aura = aura;
                ctx.in_trigger = true;
                ctx.first = first;
                self.run_effects(&t.effects, &mut ctx, true);
            }
        }
    }

    // ---------------------------------------------------------------------------- auras
    fn aura_list(&self, h: Holder) -> &Vec<Aura> {
        match h {
            Holder::Pet(r) => &self.pet(r).auras,
            Holder::Team(s) => &self.teams[s].auras,
            Holder::Weather => unreachable!(),
        }
    }
    fn aura_list_mut(&mut self, h: Holder) -> &mut Vec<Aura> {
        match h {
            Holder::Pet(r) => &mut self.teams[r.side].pets[r.idx].auras,
            Holder::Team(s) => &mut self.teams[s].auras,
            Holder::Weather => unreachable!(),
        }
    }
    fn aura_mut(&mut self, h: Holder, uid: u64) -> Option<&mut Aura> {
        match h {
            Holder::Weather => self.weather.as_mut().filter(|w| w.uid == uid),
            _ => self.aura_list_mut(h).iter_mut().find(|a| a.uid == uid),
        }
    }
    fn holder_pet(&self, h: Holder) -> PetRef {
        match h {
            Holder::Pet(r) => r,
            Holder::Team(s) => self.active(s),
            Holder::Weather => unreachable!(),
        }
    }

    fn add_aura(&mut self, h: Holder, aid: u32, caster: Option<PetRef>, duration: i64, max_allowed: i64, next_attack: bool) -> Option<u64> {
        let a = gd().ab(aid)?;
        let cap = (if max_allowed != 0 { max_allowed } else { 1 }).max(1) as usize;
        let existing: Vec<usize> = self.aura_list(h).iter().enumerate().filter(|(_, x)| x.aid == aid).map(|(i, _)| i).collect();
        if !existing.is_empty() && existing.len() >= cap {
            if cap == 1 {
                // a refresh keeps the first caster (logs 2026-09-27: Poisoned, Swarm of Flies)
                let x = &mut self.aura_list_mut(h)[existing[0]];
                x.duration = duration;
                x.age = 0;
                x.fresh = true;
                return Some(x.uid);
            }
            self.aura_list_mut(h).remove(existing[0]);
        }
        let uid = self.next_uid();
        self.aura_list_mut(h).push(Aura {
            aid,
            caster,
            duration,
            fresh: true,
            stacks: 1,
            uid,
            age: 0,
            next_attack,
            blocks_left: a.block_count,
        });
        emit!(self, "{} applied to {}", a.name, self.holder_name(h));
        let pet = self.holder_pet(h);
        self.fire(EV_APPLY, pet, None, None, Some((aid, caster.unwrap_or(pet), uid)));
        Some(uid)
    }

    fn remove_aura(&mut self, h: Holder, uid: u64) {
        if h == Holder::Weather {
            if self.weather.as_ref().map(|w| w.uid == uid).unwrap_or(false) {
                let w = self.weather.take().unwrap();
                emit!(self, "{} is no longer the weather", aname(w.aid));
            }
            return;
        }
        let pos = match self.aura_list(h).iter().position(|a| a.uid == uid) {
            Some(p) => p,
            None => return,
        };
        let a = self.aura_list_mut(h).remove(pos);
        emit!(self, "{} fades from {}", aname(a.aid), self.holder_name(h));
        let pet = self.holder_pet(h);
        if !self.pet(pet).dead {
            let x = ab(a.aid);
            for t in x.turns.iter() {
                if t.typ == 1 && t.event == EV_REMOVE {
                    let mut ctx = Ctx::new(pet, Some(pet), a.caster.unwrap_or(pet), x.family, a.aid);
                    ctx.aura = Some(uid);
                    ctx.in_trigger = true;
                    self.run_effects(&t.effects, &mut ctx, true);
                }
            }
        }
    }

    fn set_weather(&mut self, aid: u32, caster: PetRef, duration: i64) {
        if let Some(w) = &self.weather {
            emit!(self, "{} is no longer the weather", aname(w.aid));
        }
        let uid = self.next_uid();
        self.weather = Some(Aura { aid, caster: Some(caster), duration, fresh: true, stacks: 1, uid, age: 0, next_attack: false, blocks_left: 0 });
        emit!(self, "weather changed to {}", aname(aid));
    }

    fn tick_auras(&mut self) {
        let first = self.first_side.unwrap_or(0);
        let order = [first, 1 - first];
        let mut holders: Vec<(Holder, u64)> = Vec::new();
        if let Some(w) = &self.weather {
            holders.push((Holder::Weather, w.uid));
        }
        for &side in &order {
            for (i, p) in self.teams[side].pets.iter().enumerate() {
                for a in &p.auras {
                    holders.push((Holder::Pet(PetRef { side, idx: i }), a.uid));
                }
            }
        }
        for &side in &order {
            for a in &self.teams[side].auras {
                holders.push((Holder::Team(side), a.uid));
            }
        }
        for (h, uid) in holders {
            if self.over() {
                return;
            }
            let (aid, age, caster) = match self.aura_mut(h, uid) {
                Some(a) => {
                    let age = a.age;
                    a.age += 1;
                    (a.aid, age, a.caster)
                }
                None => continue,
            };
            let x = ab(aid);
            if x.main.is_empty() {
                continue;
            }
            let ti = if (age as usize) < x.main.len() {
                x.main[age as usize]
            } else if x.main.len() == 1 {
                x.main[0]
            } else {
                continue;
            };
            let effects = &x.turns[ti].effects;
            if effects.is_empty() {
                continue;
            }
            match h {
                Holder::Pet(r) => {
                    if self.pet(r).dead {
                        continue;
                    }
                    let mut ctx = Ctx::new(r, Some(r), caster.unwrap_or(r), x.family, aid);
                    ctx.aura = Some(uid);
                    self.run_effects(effects, &mut ctx, true);
                }
                Holder::Team(side) => {
                    let me = self.active(side);
                    let own = caster.map(|c| c.side == side).unwrap_or(false);
                    let tgt = if own { self.active(1 - side) } else { me };
                    let mut ctx = Ctx::new(me, Some(tgt), caster.unwrap_or(me), x.family, aid);
                    ctx.aura = Some(uid);
                    self.run_effects(effects, &mut ctx, true);
                }
                Holder::Weather => {
                    let mut heads = [self.active(0), self.active(1)];
                    if caster.map(|c| c.side == 1).unwrap_or(false) {
                        heads.swap(0, 1);
                    }
                    for (k, _) in effects.iter().enumerate() {
                        let pet = heads[k % 2];
                        if self.pet(pet).dead {
                            continue;
                        }
                        let mut ctx = Ctx::new(pet, Some(pet), caster.unwrap_or(pet), x.family, aid);
                        ctx.aura = Some(uid);
                        ctx.row_target = Some(Tgt::Pet(pet));
                        self.run_effects(&effects[k..k + 1], &mut ctx, false);
                    }
                }
            }
        }
    }

    fn clear_dead_auras(&mut self, untimed_only: bool) {
        for side in 0..2 {
            for p in self.teams[side].pets.iter_mut() {
                if p.dead && !p.auras.is_empty() {
                    if untimed_only {
                        p.auras.retain(|a| a.duration >= 0);
                    } else {
                        p.auras.clear();
                    }
                }
            }
        }
    }

    fn countdown_auras(&mut self) {
        let mut items: Vec<(Holder, u64)> = Vec::new();
        if let Some(w) = &self.weather {
            items.push((Holder::Weather, w.uid));
        }
        for side in 0..2 {
            for a in &self.teams[side].auras {
                items.push((Holder::Team(side), a.uid));
            }
            for (i, p) in self.teams[side].pets.iter().enumerate() {
                for a in &p.auras {
                    items.push((Holder::Pet(PetRef { side, idx: i }), a.uid));
                }
            }
        }
        for (h, uid) in items {
            let dealt = match h {
                Holder::Pet(r) => self.pet(r).dealt_damage_round,
                _ => false,
            };
            let a = match self.aura_mut(h, uid) {
                Some(a) => a,
                None => continue,
            };
            if a.duration < 0 {
                let rm = a.next_attack && !a.fresh && matches!(h, Holder::Pet(_)) && dealt;
                a.fresh = false;
                if rm {
                    self.remove_aura(h, uid);
                }
                continue;
            }
            if a.fresh {
                a.fresh = false;
                if a.duration == 0 {
                    self.remove_aura(h, uid);
                }
                continue;
            }
            a.duration -= 1;
            if a.duration <= 0 {
                self.remove_aura(h, uid);
            }
        }
    }

    // ---------------------------------------------------------------------------- hit / damage / heal
    fn roll_hit(&mut self, ctx: &Ctx, target: PetRef, accuracy: i64) -> bool {
        if self.pet(target).dead {
            return false;
        }
        let enemy = target.side != ctx.me.side;
        if enemy && self.state(target, S_UNTARGETTABLE) > 0 && !ctx.periodic {
            let reach = (self.state(target, S_UNDERGROUND) > 0 && self.state(ctx.caster, S_UNDERGROUND) > 0)
                || (self.state(target, S_FLYING) > 0 && self.state(ctx.caster, S_FLYING) > 0);
            if !reach {
                self.probe_count(PROBE_UNREACHABLE, target);
                return false;
            }
        }
        let mut acc = accuracy + self.state(ctx.caster, S_ACCURACY);
        if enemy {
            acc -= self.state(target, S_DODGE);
            let lvl = self.pet(target).level - self.pet(ctx.caster).level;
            if lvl > 0 {
                acc -= lvl * LEVEL_HIT_PENALTY;
            }
        }
        if self.probe_expect {
            let s = (acc as f64 / 100.0).clamp(0.0, 1.0);
            self.hit_scale = Some(s);
            if s < 1.0 {
                if let Some(c) = self.probe_counts.as_mut() {
                    match c.iter_mut().find(|(k, s, _)| *k == PROBE_ROLLED && *s == target.side) {
                        Some(x) => x.2 = 1,
                        None => c.push((PROBE_ROLLED, target.side, 1)),
                    }
                }
            }
            return true;
        }
        if acc >= 100 {
            return true;
        }
        if let Some(f) = self.force_hit {
            return f;
        }
        self.rng.random() * 100.0 < acc as f64
    }

    fn vary(&mut self, x: f64, variance: i64) -> f64 {
        if variance == 0 {
            return x;
        }
        let half = x * variance as f64 / 200.0;
        if let Some(f) = self.force_variance {
            return x + half * f;
        }
        let mut r = self.rng.random();
        for _ in 0..2 {
            let v = self.rng.random();
            if v > r {
                r = v;
            }
        }
        if self.rng.random() < 0.5 { x + half * r } else { x - half * r }
    }

    fn scaled_flat(&self, pet: PetRef, sid: i64) -> f64 {
        let mut total = 0.0f64;
        let p = self.pet(pet);
        let mut add = |a: &Aura| {
            let v = ab(a.aid).state(sid);
            if v != 0 {
                let c = a.caster.unwrap_or(pet);
                total += v as f64 * (1.0 + self.power(c) as f64 / 20.0) * a.stacks as f64;
            }
        };
        for a in &p.auras {
            add(a);
        }
        if self.teams[pet.side].active == pet.idx {
            for a in &self.teams[pet.side].auras {
                add(a);
            }
        }
        if let Some(w) = &self.weather {
            if !(p.family == ELEMENTAL && sid == S_FLAT_DMG_DEALT) {
                add(w);
            }
        }
        total + p.states.g0(sid) as f64
    }

    fn calc_damage(&mut self, ctx: &Ctx, target: PetRef, points: f64, variance: i64, can_crit: bool, raw: Option<f64>) -> (i64, bool) {
        let caster = ctx.caster;
        let mut dmg = match raw {
            Some(r) => r,
            None => points * (1.0 + self.power(caster) as f64 / 20.0),
        };
        let fam = ctx.family;
        let to = self.state(target, S_TYPE_OVERRIDE);
        let target_fam = if to > 0 { to - 1 } else { self.pet(target).family };
        let mut dealt = self.state(caster, S_DMG_DEALT_PCT);
        let taken = self.state(target, S_DMG_TAKEN_PCT);
        let pt_id = self.state(caster, S_PETTYPE_ID);
        if pt_id != 0 && fam >= 0 && pt_id - 1 == fam {
            dealt += self.state(caster, S_PETTYPE_DMG_DEALT);
        }
        dmg = self.vary(dmg, variance);
        if raw.is_none() {
            dmg *= family_mult(fam, target_fam);
            dmg *= (100 + dealt) as f64 / 100.0;
            dmg *= (1.0 - MAX_DMG_REDUCTION).max((100 + taken) as f64 / 100.0);
        }
        if ctx.periodic {
            if self.pet(target).family == AQUATIC {
                dmg *= AQUATIC_DOT_TAKEN;
            }
            if self.pet(caster).family == AQUATIC {
                dmg *= AQUATIC_DOT_DEALT;
            }
            dmg += self.scaled_flat(target, S_ADD_PERIODIC_DMG_TAKEN);
        }
        if raw.is_none() {
            dmg += self.scaled_flat(target, S_FLAT_DMG_TAKEN) + self.scaled_flat(caster, S_FLAT_DMG_DEALT);
        }
        let mut d = pint(dmg).max(0);
        let mut crit = false;
        if can_crit && raw.is_none() && d > 0 {
            let roll = match self.force_crit {
                Some(f) => f,
                None => self.rng.random() * 100.0 < BASE_CRIT + self.state(caster, S_CRIT) as f64,
            };
            if roll {
                crit = true;
                d = pint(d as f64 * (CRIT_MULT + self.state(caster, S_CRIT_DMG_BONUS) as f64 / 100.0));
            }
        }
        if self.pet(target).family == MAGIC || self.has_aura(target, BOSS_AURA) {
            d = d.min(pint(self.max_hp(target) as f64 * MAGIC_CAP + 0.5));
        }
        (d, crit)
    }

    fn deal_damage(&mut self, ctx: &mut Ctx, target: PetRef, amount: i64, crit: bool, periodic: bool, source: &str, nonlethal: bool) -> i64 {
        if self.pet(target).dead || amount <= 0 {
            return 0;
        }
        let mut amount = amount;
        let caster = ctx.caster;
        let enemy_hit = target.side != caster.side;
        if !periodic && enemy_hit {
            let found = self.teams[caster.side]
                .auras
                .iter()
                .find(|a| ab(a.aid).state(S_BLOCKED_FRIENDLY) != 0 && a.blocks_left > 0)
                .map(|a| a.uid);
            if let Some(uid) = found {
                let h = Holder::Team(caster.side);
                let a = self.aura_mut(h, uid).unwrap();
                a.blocks_left -= 1;
                let (left, aid) = (a.blocks_left, a.aid);
                emit!(self, "{} was blocked by {}", source, aname(aid));
                self.probe_count(PROBE_BLOCKED, target);
                if left <= 0 {
                    self.remove_aura(h, uid);
                }
                return 0;
            }
            for h in [Holder::Pet(target), Holder::Team(target.side)] {
                let found = self.aura_list(h).iter().find(|a| a.blocks_left > 0).map(|a| a.uid);
                if let Some(uid) = found {
                    let a = self.aura_mut(h, uid).unwrap();
                    a.blocks_left -= 1;
                    let (left, aid) = (a.blocks_left, a.aid);
                    emit!(self, "{} was blocked by {}", source, aname(aid));
                    self.probe_count(PROBE_BLOCKED, target);
                    if left <= 0 {
                        self.remove_aura(h, uid);
                    }
                    ctx.blocked = true;
                    return 0;
                }
            }
        }
        let low = self.state(target, S_IGNORE_DMG_BELOW);
        if low != 0 && amount < low {
            return 0;
        }
        let full = amount;
        if nonlethal || self.state(target, S_UNKILLABLE) > 0 {
            let hp = self.pet(target).hp;
            amount = if hp > 1 { amount.min(hp - 1) } else { 0 };
            if amount <= 0 && nonlethal {
                return 0;
            }
        }
        let before = self.pet(target).hp;
        self.pet_mut(target).hp -= amount;
        emit!(self, "{} dealt {} damage to {}{}", source, amount, self.name(target), if crit { " (Critical)" } else { "" });
        {
            let t = self.pet_mut(target);
            t.states.set(S_WAS_DAMAGED, 1);
            t.states.set(S_LAST_HIT_TAKEN, full);
        }
        {
            let c = self.pet_mut(caster);
            c.states.set(S_DID_DAMAGE, 1);
            c.states.set(S_LAST_HIT_DEALT, full);
        }
        if enemy_hit {
            self.pet_mut(caster).dealt_damage_round = true;
            if self.pet(caster).family == DRAGONKIN {
                let mhp = self.max_hp(target);
                let now = self.pet(target).hp;
                if before * 2 >= mhp && mhp > now * 2 {
                    self.pet_mut(caster).dragon_trigger = true;
                }
            }
        }
        if self.pet(target).hp <= 0 {
            self.handle_death(target);
        }
        if !periodic {
            let other = if caster != target { Some(caster) } else { None };
            self.fire(EV_DAMAGE_TAKEN, target, other, None, None);
        }
        if self.over() {
            return amount;
        }
        if enemy_hit && !ctx.in_trigger {
            self.fire(EV_DAMAGE_DEALT, caster, Some(target), None, None);
        }
        amount
    }

    fn run_passive_ability(&mut self, r: PetRef, aid: u32) {
        let a = ab(aid);
        for t in a.turns.iter() {
            if t.typ == 0 {
                let mut ctx = Ctx::new(r, Some(r), r, -1, aid);
                self.run_effects(&t.effects, &mut ctx, true);
            }
        }
    }

    fn handle_death(&mut self, r: PetRef) {
        if self.state(r, S_UNKILLABLE) > 0 {
            let p = self.pet_mut(r);
            p.hp = p.hp.max(1);
            return;
        }
        let (fam, und, mech) = {
            let p = self.pet(r);
            (p.family, p.undead_used, p.mech_revived)
        };
        if fam == UNDEAD && !und && self.state(r, S_TYPE_OVERRIDE) == 0 {
            let p = self.pet_mut(r);
            p.undead_used = true;
            p.hp = 1;
            emit!(self, "{} returns to life (Undead)", self.name(r));
            self.run_passive_ability(r, DAMNED_ABILITY);
            return;
        }
        if fam == MECHANICAL && !mech && self.state(r, S_TYPE_OVERRIDE) == 0 {
            let p = self.pet_mut(r);
            p.mech_revived = true;
            p.hp = 1;
            emit!(self, "{} comes back to life (Failsafe)", self.name(r));
            self.run_passive_ability(r, FAILSAFE_ABILITY);
            return;
        }
        self.kill(r);
    }

    fn kill(&mut self, r: PetRef) {
        if self.pet(r).dead {
            return;
        }
        {
            let p = self.pet_mut(r);
            p.hp = 0;
            p.dead = true;
            p.channel = None;
        }
        emit!(self, "{} dies", self.name(r));
        self.check_end();
    }

    fn heal(&mut self, target: PetRef, amount: f64, source: &str) -> i64 {
        if self.pet(target).dead || amount <= 0.0 {
            return 0;
        }
        let amount = pint(amount);
        let mhp = self.max_hp(target);
        let hp = self.pet(target).hp;
        let new = mhp.min(hp + amount);
        let got = new - hp;
        self.pet_mut(target).hp = new;
        emit!(self, "{} healed {} damage from {}", source, got, self.name(target));
        got
    }

    fn calc_heal(&mut self, ctx: &Ctx, target: PetRef, points: f64, variance: i64) -> i64 {
        let mut h = points * (1.0 + self.power(ctx.caster) as f64 / 20.0);
        h = self.vary(h, variance);
        h *= (100 + self.state(ctx.caster, S_HEAL_DEALT_PCT)) as f64 / 100.0;
        h *= 0.0f64.max((100 + self.state(target, S_HEAL_TAKEN_PCT)) as f64 / 100.0);
        pint(h)
    }

    fn heal_mods(&self, ctx: &Ctx, target: PetRef, amount: f64) -> i64 {
        let mut a = amount;
        a *= (100 + self.state(ctx.me, S_HEAL_DEALT_PCT)) as f64 / 100.0;
        a *= 0.0f64.max((100 + self.state(target, S_HEAL_TAKEN_PCT)) as f64 / 100.0);
        pint(a)
    }

    fn state_ok(&self, pet: Option<PetRef>, sid: Option<i64>) -> bool {
        match sid {
            None | Some(0) => true,
            Some(s) => pet.map(|p| self.state(p, s) > 0).unwrap_or(false),
        }
    }

    fn hit_and_damage(&mut self, ctx: &Ctx, p: &Params, points: Option<f64>, raw: Option<f64>, nonlethal: bool, autohit: bool) -> bool {
        let tgt = match self.tgt(ctx) {
            Some(t) if !self.pet(t).dead => t,
            _ => return false,
        };
        let periodic = p.t(K::IsPeriodic) || ctx.periodic;
        let mut sub = ctx.clone();
        sub.periodic = periodic;
        self.hit_scale = None;
        let src = aname(ctx.ability_id);
        if !autohit && !self.roll_hit(&sub, tgt, p.gd(K::Accuracy, 100)) {
            emit!(self, "{} missed {}", src, self.name(tgt));
            return false;
        }
        let pts = points.unwrap_or(p.gd(K::Points, 0) as f64);
        let (mut dmg, crit) = self.calc_damage(&sub, tgt, pts, p.gd(K::Variance, 0), true, raw);
        if let Some(s) = self.hit_scale.take() {
            if s < 1.0 {
                dmg = pint(dmg as f64 * s);
            }
        }
        self.deal_damage(&mut sub, tgt, dmg, crit, periodic, src, nonlethal);
        !sub.blocked
    }

    fn immune(&self, target: PetRef, aura_id: u32) -> bool {
        let a = match gd().ab(aura_id) {
            Some(a) => a,
            None => return false,
        };
        let cc = a.state(S_STUNNED) != 0 || a.state(S_TURN_LOCK) != 0 || a.state(S_SWAP_OUT_LOCK) != 0;
        if cc && self.pet(target).family == CRITTER {
            return true;
        }
        (a.state(S_STUNNED) != 0 || a.state(S_TURN_LOCK) != 0) && self.state(target, S_RESILIENT) > 0
    }

    fn aura_target(&mut self, e: &Effect, ctx: &Ctx, check_states: bool, chance: Option<i64>) -> Option<bool> {
        let p = &e.p;
        let holder: Option<Tgt> = match ctx.row_target {
            Some(t) => Some(t),
            None => match ctx.target {
                Some(t) => {
                    if gd().ab(e.aura).map(|a| a.is_object).unwrap_or(false) {
                        Some(Tgt::Team(t.side))
                    } else {
                        Some(Tgt::Pet(t))
                    }
                }
                None => None,
            },
        };
        let holder = match holder {
            Some(h) => h,
            None => return Some(false),
        };
        let pet = match holder {
            Tgt::Team(s) => self.active(s),
            Tgt::Pet(r) => r,
        };
        if self.pet(pet).dead {
            return Some(false);
        }
        if check_states {
            if p.t(K::CasterState) && self.state(ctx.me, p.gd(K::CasterState, 0)) <= 0 {
                return None;
            }
            if p.t(K::TargetState) && self.state(pet, p.gd(K::TargetState, 0)) <= 0 {
                return None;
            }
        }
        let acc = chance.unwrap_or_else(|| p.gd(K::Accuracy, 100));
        let is_enemy_pet = matches!(holder, Tgt::Pet(r) if r.side != ctx.me.side);
        if is_enemy_pet {
            if !self.roll_hit(ctx, pet, acc) {
                emit!(self, "{} missed {}", aname(ctx.ability_id), self.name(pet));
                return Some(false);
            }
            if self.immune(pet, e.aura) {
                emit!(self, "{} is immune", self.name(pet));
                self.probe_count(PROBE_IMMUNE, pet);
                return Some(false);
            }
        } else if acc < 100 && self.rng.random() * 100.0 >= acc as f64 {
            return Some(false);
        }
        let h = match holder {
            Tgt::Pet(r) => Holder::Pet(r),
            Tgt::Team(s) => Holder::Team(s),
        };
        self.add_aura(h, e.aura, Some(ctx.caster), duration_of(p, e.op), p.gd(K::MaxAllowed, 0), p.gd(K::Duration, 0) == -1);
        Some(true)
    }

    fn aura_self(&mut self, e: &Effect, ctx: &Ctx) -> Option<bool> {
        let p = &e.p;
        let mut holder = match ctx.row_target {
            Some(Tgt::Pet(r)) => Holder::Pet(r),
            Some(Tgt::Team(s)) => Holder::Team(s),
            None => Holder::Pet(ctx.me),
        };
        if let Holder::Pet(r) = holder {
            if e.op == 150 && gd().ab(e.aura).map(|a| a.is_object).unwrap_or(false) {
                holder = Holder::Team(r.side);
            }
        }
        if let Holder::Pet(r) = holder {
            if self.pet(r).dead {
                return Some(false);
            }
        }
        self.add_aura(holder, e.aura, Some(ctx.caster), duration_of(p, e.op), 0, false);
        Some(true)
    }

    fn writes_state(aid: u32, sid: i64) -> bool {
        gd().ab(aid).map(|a| a.writes.contains(&sid)).unwrap_or(false)
    }

    fn gate_pet(&self, ctx: &Ctx, sid: Option<i64>) -> Option<PetRef> {
        if ctx.aura.is_some() || sid.map(|s| Self::writes_state(ctx.ability_id, s)).unwrap_or(false) {
            return Some(ctx.me);
        }
        self.tgt(ctx)
    }

    fn mod_state(&mut self, p: &Params, ctx: &Ctx) {
        let s = p.gd(K::State, 0);
        let mut v = self.pet(ctx.me).states.g0(s) + p.gd(K::StateChange, 0);
        let (lo, hi) = (p.gd(K::StateMin, 0), p.gd(K::StateMax, 0));
        if hi != 0 {
            v = v.min(hi);
        }
        if lo != 0 {
            v = v.max(lo);
        }
        self.pet_mut(ctx.me).states.set(s, v);
    }

    fn heal_target(&self, ctx: &Ctx) -> PetRef {
        match ctx.row_target {
            Some(Tgt::Team(s)) => self.active(s),
            Some(Tgt::Pet(r)) => r,
            None => ctx.me,
        }
    }

    fn op_heal(&mut self, p: &Params, ctx: &Ctx) -> Option<bool> {
        let tgt = self.heal_target(ctx);
        if self.pet(tgt).dead {
            return Some(false);
        }
        let h = self.calc_heal(ctx, tgt, p.gd(K::Points, 0) as f64, p.gd(K::Variance, 0));
        self.heal(tgt, h as f64, aname(ctx.ability_id));
        Some(true)
    }

    // ---------------------------------------------------------------------------- opcodes
    fn op(&mut self, e: &'static Effect, ctx: &mut Ctx) -> Option<bool> {
        let p = &e.p;
        let src = aname(ctx.ability_id);
        match e.op {
            // ---- damage
            24 | 222 | 179 => Some(self.hit_and_damage(ctx, p, None, None, false, false)),
            169 => {
                let ts = p.get(K::RequiredTargetState);
                if let Some(ts) = ts.filter(|t| PASSIVE_STATES.contains(t)) {
                    let fam = PASSIVE_STATES.iter().position(|x| *x == ts).unwrap() as i64;
                    if let Some(t) = self.tgt(ctx) {
                        if self.pet(t).family == fam || self.state(t, ts) > 0 {
                            return None;
                        }
                    }
                    return Some(self.hit_and_damage(ctx, p, None, None, false, false));
                }
                self.op_damage_if_state(p, ctx)
            }
            29 => self.op_damage_if_state(p, ctx),
            96 => {
                let tgt = self.tgt(ctx);
                let auto = (p.t(K::CasterState) && self.state(ctx.me, p.gd(K::CasterState, 0)) > 0)
                    || (p.t(K::TargetState) && tgt.map(|t| self.state(t, p.gd(K::TargetState, 0)) > 0).unwrap_or(false));
                Some(self.hit_and_damage(ctx, p, None, None, false, auto))
            }
            384 => {
                if !ctx.first {
                    return None;
                }
                Some(self.hit_and_damage(ctx, p, None, None, false, false))
            }
            385 => {
                if ctx.first {
                    return None;
                }
                Some(self.hit_and_damage(ctx, p, None, None, false, false))
            }
            141 => {
                let mut pts = p.gd(K::Points, 0);
                if p.t(K::BonusState) && self.state(ctx.me, p.gd(K::BonusState, 0)) > 0 {
                    pts += p.gd(K::BonusPoints, 0);
                }
                let es = 1 - ctx.me.side;
                let enemies: Vec<PetRef> = (0..self.teams[es].pets.len()).map(|i| PetRef { side: es, idx: i }).filter(|r| !self.pet(*r).dead).collect();
                if enemies.is_empty() {
                    return Some(false);
                }
                let n = enemies.len() as f64;
                let mut hits = Vec::new();
                for &x in &enemies {
                    let mut sub = ctx.clone();
                    sub.row_target = Some(Tgt::Pet(x));
                    if self.roll_hit(&sub, x, p.gd(K::Accuracy, 100)) {
                        let (d, c) = self.calc_damage(&sub, x, pts as f64 / n, 0, true, None);
                        hits.push((sub, x, d, c));
                    } else {
                        emit!(self, "{} missed {}", src, self.name(x));
                    }
                }
                let any = !hits.is_empty();
                for (mut sub, x, d, c) in hits {
                    self.deal_damage(&mut sub, x, d, c, false, src, false);
                }
                Some(any)
            }
            226 | 65 => {
                let mut pts = p.gd(K::Points, 0);
                let bs = p.get(K::BonusState);
                let tgt = self.tgt(ctx);
                if let Some(bs) = bs.filter(|b| *b != 0) {
                    let weather = WEATHER_STATES.contains(&bs);
                    if tgt.map(|t| self.state(t, bs) > 0).unwrap_or(false) || (weather && self.state(ctx.me, bs) > 0) {
                        pts += p.gd(K::BonusPoints, 0);
                    }
                }
                Some(self.hit_and_damage(ctx, p, Some(pts as f64), None, false, false))
            }
            170 => {
                if p.t(K::WeatherState) && self.state(ctx.me, p.gd(K::WeatherState, 0)) <= 0 {
                    return None;
                }
                Some(self.hit_and_damage(ctx, p, None, None, false, false))
            }
            27 | 358 => {
                let me = ctx.me;
                let aid = ctx.ability_id as i64;
                if self.pet(me).states.get(S_RAMPING_ID) != Some(aid) {
                    let st = &mut self.pet_mut(me).states;
                    st.set(S_RAMPING_ID, aid);
                    st.set(S_RAMPING_USES, 0);
                }
                let uses = self.pet(me).states.g0(S_RAMPING_USES);
                let cap = match p.get(K::PointsMax) {
                    Some(c) if c != 0 => c,
                    _ => 1_000_000,
                };
                let pts = (p.gd(K::Points, 0) + uses * p.gd(K::PointsIncreasePerUse, 0)).min(cap);
                self.pet_mut(me).states.set(S_RAMPING_USES, uses + 1);
                Some(self.hit_and_damage(ctx, p, Some(pts as f64), None, false, false))
            }
            62 => {
                let t = match self.tgt(ctx) {
                    Some(t) => t,
                    None => return Some(false),
                };
                let raw = (self.max_hp(t) * p.gd(K::Points, 0)) as f64 / 100.0;
                Some(self.hit_and_damage(ctx, p, None, Some(raw), false, false))
            }
            68 => {
                let raw = (self.max_hp(ctx.caster) * p.gd(K::Points, 0)) as f64 / 100.0;
                Some(self.hit_and_damage(ctx, p, None, Some(raw), false, false))
            }
            363 => {
                let t = match self.tgt(ctx) {
                    Some(t) => t,
                    None => return Some(false),
                };
                let raw = (self.pet(t).hp * p.gd(K::Points, 0)) as f64 / 100.0;
                Some(self.hit_and_damage(ctx, p, None, Some(raw), false, false))
            }
            197 => {
                let last = self.pet(ctx.caster).states.g0(S_LAST_HIT_TAKEN);
                Some(self.hit_and_damage(ctx, p, None, Some((last * p.gd(K::Power, 0)) as f64 / 100.0), false, false))
            }
            204 => {
                let last = self.pet(ctx.me).states.g0(S_LAST_HIT_TAKEN);
                Some(self.hit_and_damage(ctx, p, None, Some((last * p.gd(K::Points, 0)) as f64 / 100.0), false, false))
            }
            149 => Some(self.hit_and_damage(ctx, p, None, None, true, false)),
            59 => {
                match self.tgt(ctx) {
                    Some(t) if self.pet(ctx.me).hp < self.pet(t).hp => {}
                    _ => return None,
                }
                Some(self.hit_and_damage(ctx, p, None, None, false, false))
            }
            66 | 45 | 97 => {
                let t = match self.tgt(ctx) {
                    Some(t) => t,
                    None => return Some(false),
                };
                let boost = p.gd(K::Boost, 0);
                let cond = match e.op {
                    66 => self.pet(t).hp * 4 < self.max_hp(t),
                    45 => self.speed(ctx.me) < self.speed(t),
                    _ => self.speed(ctx.me) > self.speed(t),
                };
                let base = p.gd(K::Points, 0) as f64;
                let pts = if cond { base * ((100 + boost) as f64 / 100.0) } else { base };
                Some(self.hit_and_damage(ctx, p, Some(pts), None, false, false))
            }
            370 => {
                let me = ctx.me;
                let sp = self.pet(me).species;
                let same = self.teams[me.side].pets.iter().enumerate().filter(|(i, x)| *i != me.idx && x.species == sp).count() as i64;
                let pts = p.gd(K::Points, 0) + same * p.gd(K::BonusDamage, 0);
                Some(self.hit_and_damage(ctx, p, Some(pts as f64), None, false, false))
            }
            86 => {
                let tgt = self.tgt(ctx);
                let ok = self.hit_and_damage(ctx, p, None, None, false, false);
                if ok {
                    if let Some(t) = tgt {
                        if p.t(K::AuraID) && self.has_aura(t, e.aura) {
                            self.add_aura(Holder::Pet(t), p.gd(K::AuraID, 0) as u32, Some(ctx.caster), p.gd(K::Duration2, 1), 0, false);
                        }
                    }
                }
                Some(ok)
            }
            // ---- heals
            23 => self.op_heal(p, ctx),
            104 | 171 => {
                if !self.state_ok(Some(ctx.me), p.get(K::RequiredCasterState)) {
                    return None;
                }
                if p.t(K::RequiredTargetState) {
                    let rts = p.gd(K::RequiredTargetState, 0);
                    let ok = match ctx.target {
                        Some(t) => self.state(t, rts) > 0 || (rts == S_IS_DEAD && self.pet(t).dead),
                        None => false,
                    };
                    if !ok {
                        return None;
                    }
                }
                self.op_heal(p, ctx)
            }
            61 => {
                if !self.state_ok(Some(ctx.me), p.get(K::RequiredCasterState)) {
                    return None;
                }
                let tgt = self.heal_target(ctx);
                let h = self.calc_heal(ctx, tgt, p.gd(K::Points, 0) as f64, 0);
                self.heal(tgt, h as f64, src);
                Some(true)
            }
            100 => {
                let tgt = self.heal_target(ctx);
                if p.t(K::RequiredTargetPetType) && self.pet(tgt).family + 1 != p.gd(K::RequiredTargetPetType, 0) {
                    return None;
                }
                if p.t(K::RequiredCasterPetType) && self.pet(ctx.me).family + 1 != p.gd(K::RequiredCasterPetType, 0) {
                    return None;
                }
                self.op_heal(p, ctx)
            }
            254 => {
                let tgt = self.heal_target(ctx);
                let amt = self.calc_heal(ctx, tgt, p.gd(K::Points, 0) as f64, p.gd(K::Variance, 0));
                let mut a = amt as f64;
                if self.pet(tgt).hp * 100 < self.max_hp(tgt) * p.gd(K::Threshold, 0) {
                    a = (amt * (100 + p.gd(K::Modifier, 0))) as f64 / 100.0;
                }
                self.heal(tgt, a, src);
                Some(true)
            }
            53 => {
                let tgt = self.heal_target(ctx);
                let amt = (self.max_hp(tgt) * p.gd(K::Percentage, 0)) as f64 / 100.0;
                let h = self.heal_mods(ctx, tgt, amt);
                self.heal(tgt, h as f64, src);
                Some(true)
            }
            32 => {
                let me = ctx.me;
                let last = self.pet(me).states.g0(S_LAST_HIT_DEALT);
                let h = self.heal_mods(ctx, me, (last * p.gd(K::Points, 0)) as f64 / 100.0);
                self.heal(me, h as f64, src);
                Some(true)
            }
            44 => {
                let me = ctx.me;
                let last = self.pet(me).states.g0(S_LAST_HIT_TAKEN);
                let h = self.heal_mods(ctx, me, (last * p.gd(K::Points, 0)) as f64 / 100.0);
                self.heal(me, h as f64, src);
                Some(true)
            }
            67 => {
                let t = match self.tgt(ctx) {
                    Some(t) => t,
                    None => return Some(false),
                };
                let me = ctx.me;
                let avg = (self.pet(me).hp + self.pet(t).hp).div_euclid(2);
                let (new_me, new_t) = (self.max_hp(me).min(avg), self.max_hp(t).min(avg));
                for (pet, new) in [(t, new_t), (me, new_me)] {
                    let d = new - self.pet(pet).hp;
                    if d < 0 {
                        emit!(self, "{} dealt {} damage to {}", src, -d, self.name(pet));
                    } else if d > 0 {
                        emit!(self, "{} healed {} damage from {}", src, d, self.name(pet));
                    }
                    self.pet_mut(pet).hp = new;
                }
                Some(true)
            }
            111 => {
                let me = ctx.me;
                self.pet_mut(me).dead = false;
                let hp = pint((self.max_hp(me) * p.gd(K::Percentage, 0)) as f64 / 100.0 + 0.5).max(1);
                self.pet_mut(me).hp = hp;
                emit!(self, "{} healed {} damage from {}", src, hp, self.name(me));
                Some(true)
            }
            112 => {
                let pct = p.gd(K::Percentage, 0);
                let me = ctx.me;
                for i in 0..self.teams[me.side].pets.len() {
                    let x = PetRef { side: me.side, idx: i };
                    if self.pet(x).dead && x != me {
                        self.pet_mut(x).dead = false;
                        let hp = pint((self.max_hp(x) * pct) as f64 / 100.0).max(1);
                        self.pet_mut(x).hp = hp;
                    }
                }
                Some(true)
            }
            128 => {
                let t = match self.tgt(ctx) {
                    Some(t) if !self.pet(t).dead => t,
                    _ => return Some(false),
                };
                let hp = pint((self.max_hp(t) * p.gd(K::Percentage, 0)) as f64 / 100.0).max(1);
                self.pet_mut(t).hp = hp;
                Some(true)
            }
            // ---- auras
            26 | 28 | 63 | 150 => self.aura_self(e, ctx),
            248 => {
                if !self.state_ok(Some(ctx.me), p.get(K::RequiredCasterState)) {
                    return None;
                }
                self.aura_self(e, ctx)
            }
            137 => {
                if p.t(K::CasterState) && self.state(ctx.me, p.gd(K::CasterState, 0)) != p.gd(K::CasterStateValue, 1) {
                    return None;
                }
                if p.t(K::TargetState) {
                    match ctx.target {
                        Some(t) if self.state(t, p.gd(K::TargetState, 0)) == p.gd(K::TargetStateValue, 1) => {}
                        _ => return None,
                    }
                }
                self.aura_self(e, ctx)
            }
            223 => {
                let me = ctx.me;
                let want = p.get(K::RequiredPetType);
                for i in 0..self.teams[me.side].pets.len() {
                    let x = PetRef { side: me.side, idx: i };
                    if !self.pet(x).dead && Some(self.pet(x).family + 1) == want {
                        self.add_aura(Holder::Pet(x), e.aura, Some(ctx.caster), duration_of(p, 26), 0, false);
                    }
                }
                Some(true)
            }
            50 | 52 | 54 | 168 | 229 => self.aura_target(e, ctx, true, None),
            230 => self.aura_target(e, ctx, false, None),
            131 => {
                if !ctx.first {
                    return None;
                }
                self.aura_target(e, ctx, true, None)
            }
            172 => {
                let sure = (p.t(K::CasterState) && self.state(ctx.me, p.gd(K::CasterState, 0)) > 0)
                    || (p.t(K::TargetState) && ctx.target.map(|t| self.state(t, p.gd(K::TargetState, 0)) > 0).unwrap_or(false));
                self.aura_target(e, ctx, false, Some(if sure { 200 } else { p.gd(K::Accuracy, 0) }))
            }
            178 => {
                let t = match self.tgt(ctx) {
                    Some(t) if !self.pet(t).dead => t,
                    _ => return Some(false),
                };
                if !self.roll_hit(ctx, t, p.gd(K::Accuracy, 100)) {
                    return Some(false);
                }
                let ts = p.gd(K::TargetState, 0);
                if (ts != 0 && self.state(t, ts) == p.gd(K::StatePoints, 1)) || self.immune(t, e.aura) {
                    emit!(self, "{} is immune", self.name(t));
                    self.probe_count(PROBE_IMMUNE, t);
                    return Some(false);
                }
                let aura = if e.aura != 0 { e.aura } else { STUNNED_AURA };
                self.add_aura(Holder::Pet(t), aura, Some(ctx.caster), p.gd(K::Duration, 1), 0, false);
                Some(true)
            }
            76 | 77 | 75 => {
                let holder = match e.op {
                    76 => Holder::Pet(ctx.me),
                    77 => match self.tgt(ctx) {
                        Some(t) => Holder::Pet(t),
                        None => return Some(false),
                    },
                    _ => Holder::Team(ctx.me.side),
                };
                let existing = self.aura_list(holder).iter().find(|x| x.aid == e.aura).map(|x| x.uid);
                match existing {
                    None => {
                        let mut d = p.gd(K::Duration, -1);
                        if d == 0 {
                            d = -1;
                        }
                        self.add_aura(holder, e.aura, Some(ctx.caster), d, 0, false);
                        Some(true)
                    }
                    Some(uid) => {
                        let ok = self.hit_and_damage(ctx, p, None, None, false, false);
                        if self.aura_list(holder).iter().any(|x| x.uid == uid) {
                            self.remove_aura(holder, uid);
                        }
                        Some(ok)
                    }
                }
            }
            140 => {
                let t = self.tgt(ctx)?;
                let nd = p.get(K::NewDuration);
                for a in self.pet_mut(t).auras.iter_mut() {
                    if a.aid == e.aura {
                        a.duration = nd.unwrap_or(a.duration);
                    }
                }
                None
            }
            49 => {
                let ch = p.gd(K::Chance, 0);
                if 0 < ch && ch < 100 && self.rng.random() * 100.0 >= ch as f64 {
                    return None;
                }
                for h in [Holder::Pet(ctx.me), Holder::Team(ctx.me.side)] {
                    let found = self.aura_list(h).iter().find(|x| x.aid == e.aura).map(|x| x.uid);
                    if let Some(uid) = found {
                        self.remove_aura(h, uid);
                        break;
                    }
                }
                None
            }
            85 => {
                let sid = p.get(K::State);
                let t = ctx.row_target.or(ctx.target.map(Tgt::Pet));
                let holders: Vec<Holder> = match t {
                    Some(Tgt::Team(s)) => vec![Holder::Team(s)],
                    Some(Tgt::Pet(r)) => vec![Holder::Pet(r), Holder::Team(r.side)],
                    None => vec![],
                };
                for h in holders {
                    let uids: Vec<(u64, u32)> = self.aura_list(h).iter().map(|a| (a.uid, a.aid)).collect();
                    for (uid, aid) in uids {
                        let has = sid.map(|s| ab(aid).state(s) != 0).unwrap_or(false);
                        if has {
                            self.remove_aura(h, uid);
                        }
                    }
                }
                None
            }
            136 => {
                let t = match ctx.row_target.or(ctx.target.map(Tgt::Pet)) {
                    Some(Tgt::Team(s)) => self.active(s),
                    Some(Tgt::Pet(r)) => r,
                    None => return None,
                };
                let uids: Vec<(u64, u32)> = self.pet(t).auras.iter().map(|a| (a.uid, a.aid)).collect();
                for (uid, aid) in uids {
                    if !PASSIVE_ABILITY.contains(&aid) {
                        self.remove_aura(Holder::Pet(t), uid);
                    }
                }
                Some(true)
            }
            // ---- states and gates
            31 | 360 => {
                let (s, v) = (p.gd(K::State, 0), p.gd(K::StateValue, 0));
                self.pet_mut(ctx.me).states.set(s, v);
                None
            }
            79 => {
                self.mod_state(p, ctx);
                None
            }
            138 => {
                if let Some(t) = ctx.target {
                    if self.state(t, p.gd(K::TargetTestState, 0)) >= p.gd(K::TargetTestStateValue, 1).max(1) {
                        self.mod_state(p, ctx);
                    }
                }
                None
            }
            157 => {
                let s = p.gd(K::State, 0);
                let pet = self.gate_pet(ctx, Some(s));
                let mut val = pet.map(|r| self.state(r, s)).unwrap_or(0);
                if s == S_IS_DEAD {
                    if let Some(r) = pet {
                        val = self.pet(r).dead as i64;
                    }
                }
                if val != p.gd(K::StatePoints, 0) {
                    ctx.stop = true;
                }
                None
            }
            156 => {
                let pet = self.gate_pet(ctx, p.get(K::State));
                if let Some(r) = pet {
                    if self.state(r, p.gd(K::State, 0)) > 0 {
                        ctx.stop = true;
                    }
                }
                None
            }
            177 => {
                if let Some(t) = self.tgt(ctx) {
                    if self.state(t, p.gd(K::State, S_RESILIENT)) > 0 {
                        self.probe_count(PROBE_IMMUNE, t);
                        return Some(false);
                    }
                }
                Some(true)
            }
            158 => {
                if ctx.failed {
                    ctx.stop = true;
                }
                None
            }
            194 => {
                if !ctx.missed {
                    ctx.stop = true;
                }
                None
            }
            159 => {
                ctx.failed = false;
                ctx.missed = false;
                None
            }
            // ---- weather, kill, swaps, misc
            80 => {
                self.set_weather(e.aura, ctx.caster, p.gd(K::Duration, 0));
                Some(true)
            }
            135 => {
                let v = ctx.me;
                if p.t(K::CasterImmuneState) && self.state(v, p.gd(K::CasterImmuneState, 0)) > 0 {
                    return None;
                }
                if p.t(K::TargetImmuneState) && self.state(v, p.gd(K::TargetImmuneState, 0)) > 0 {
                    return None;
                }
                if !self.pet(v).dead {
                    let hp = self.pet(v).hp;
                    if hp > 0 {
                        emit!(self, "{} dealt {} damage to {}", src, hp, self.name(v));
                    }
                    if p.t(K::EnableReverse) {
                        self.pet_mut(v).states.set(S_HEALTH_BEFORE_INSTAKILL, hp);
                    }
                    self.kill(v);
                }
                Some(true)
            }
            165 => {
                let c = ctx.caster;
                if self.pet(c).dead && self.pet(c).states.g0(S_HEALTH_BEFORE_INSTAKILL) != 0 {
                    let before = self.pet_mut(c).states.pop(S_HEALTH_BEFORE_INSTAKILL).unwrap();
                    let hp = ((before + 1).div_euclid(2)).max(1);
                    let pm = self.pet_mut(c);
                    pm.dead = false;
                    pm.hp = hp;
                    emit!(self, "{} healed {} damage from {}", src, hp, self.name(c));
                }
                Some(true)
            }
            145 | 139 | 107 => {
                let side = match ctx.row_target {
                    Some(Tgt::Pet(r)) => r.side,
                    Some(Tgt::Team(s)) => s,
                    None => {
                        if ctx.in_trigger { ctx.me.side } else { 1 - ctx.me.side }
                    }
                };
                let t = &self.teams[side];
                let cands: Vec<usize> = (0..t.pets.len()).filter(|&i| i != t.active && !t.pets[i].dead).collect();
                if cands.is_empty() {
                    return Some(false);
                }
                if side != ctx.me.side {
                    let acc = match p.gd(K::Accuracy, 100) {
                        0 => 100,
                        a => a,
                    };
                    let act = self.active(side);
                    if !self.roll_hit(ctx, act, acc) {
                        return Some(false);
                    }
                }
                let hp = |i: usize| self.teams[side].pets[i].hp;
                let idx = match e.op {
                    145 => {
                        let mut best = cands[0];
                        for &i in &cands[1..] {
                            if hp(i) > hp(best) {
                                best = i;
                            }
                        }
                        best
                    }
                    139 => {
                        let mut best = cands[0];
                        for &i in &cands[1..] {
                            if hp(i) < hp(best) {
                                best = i;
                            }
                        }
                        best
                    }
                    _ => cands[self.rng.randbelow(cands.len() as u64) as usize],
                };
                self.swap(side, idx, true);
                Some(true)
            }
            246 => {
                let slot = p.gd(K::Index, 0);
                if (0..3).contains(&slot) {
                    let m = p.gd(K::CooldownModification, 0);
                    let cd = &mut self.pet_mut(ctx.me).cooldowns[slot as usize];
                    *cd = if m != 0 { (*cd + m).max(0) } else { 0 };
                }
                None
            }
            129 => {
                let me = ctx.me;
                if let Some(aid) = self.pet(me).last_ability {
                    if let Some(i) = self.pet(me).abilities.iter().position(|&x| x == aid) {
                        let ld = p.gd(K::LockDuration, 1) + 1;
                        let cd = &mut self.pet_mut(me).cooldowns[i];
                        *cd = (*cd).max(ld);
                    }
                }
                None
            }
            117 => {
                if let Some(t) = self.tgt(ctx) {
                    let i = p.gd(K::Index, 0);
                    if (0..3).contains(&i) {
                        let ld = p.gd(K::LockDuration, 1) + 1;
                        let cd = &mut self.pet_mut(t).cooldowns[i as usize];
                        *cd = (*cd).max(ld);
                    }
                }
                None
            }
            373 | 249 => {
                let slot = p.gd(K::Slot, 0);
                if (0..3).contains(&slot) && p.t(K::Ability) {
                    self.pet_mut(ctx.me).abilities[slot as usize] = p.gd(K::Ability, 0) as u32;
                }
                None
            }
            78 => {
                let me = ctx.me;
                let mp = match p.gd(K::MaxPoints, 10) {
                    0 => 10,
                    x => x,
                };
                let n = self.pet(me).states.g0(p.gd(K::State, S_PLANT)).min(mp);
                if n != 0 {
                    let h = self.calc_heal(ctx, me, (p.gd(K::Points, 0) * n) as f64, 0);
                    self.heal(me, h as f64, src);
                }
                Some(true)
            }
            144 => match self.free_corpse(ctx.me) {
                None => {
                    ctx.stop = true;
                    Some(false)
                }
                Some(i) => {
                    let side = ctx.me.side;
                    self.teams[side].pets[i].states.set(S_CONSUMED_CORPSE, 1);
                    Some(true)
                }
            },
            116 | 22 | 0 | 256 => None,
            _ => None,
        }
    }

    fn op_damage_if_state(&mut self, p: &Params, ctx: &Ctx) -> Option<bool> {
        if !self.state_ok(Some(ctx.me), p.get(K::RequiredCasterState)) {
            return None;
        }
        let t = self.tgt(ctx);
        if !self.state_ok(t, p.get(K::RequiredTargetState)) {
            return None;
        }
        Some(self.hit_and_damage(ctx, p, None, None, false, false))
    }
}

fn has_handler(op: i64) -> bool {
    matches!(
        op,
        24 | 222 | 179 | 169 | 29 | 96 | 384 | 385 | 141 | 226 | 65 | 170 | 27 | 358 | 62 | 68 | 363 | 197 | 204 | 149 | 59 | 66
            | 45 | 97 | 370 | 86 | 23 | 104 | 171 | 61 | 100 | 254 | 53 | 32 | 44 | 67 | 111 | 112 | 128 | 26 | 28 | 63 | 150
            | 248 | 137 | 223 | 50 | 52 | 54 | 168 | 229 | 230 | 131 | 172 | 178 | 76 | 77 | 75 | 140 | 49 | 85 | 136 | 31
            | 360 | 79 | 138 | 157 | 156 | 177 | 158 | 194 | 159 | 80 | 135 | 165 | 145 | 139 | 107 | 246 | 129 | 117 | 373
            | 249 | 78 | 144 | 116 | 22 | 0 | 256
    )
}
