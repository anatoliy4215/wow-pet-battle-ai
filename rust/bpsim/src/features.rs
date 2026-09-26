//! Port of bp/pvp/features.py (encode, action_mask) and bp/pvp/probe.py: the network's input, built
//! from a Rust battle. Must match the Python arrays bit for bit (tests/test_rust_sim.py); where numpy
//! computes in float32 (descriptor arithmetic) the same float32 steps are used here.

use crate::data::gd;
use crate::engine::*;

pub const MAX_AURAS: usize = 20;
pub const N_ABIL: usize = 27;
pub const N_GLOB: usize = 12;
pub const N_PET: usize = 43;
pub const N_PROBE: usize = 8;
pub const N_ABF: usize = 10 + N_PROBE;
pub const N_AUF: usize = 8;
pub const PHASE_LEAD: usize = 0;
pub const PHASE_ROUND: usize = 1;
pub const PHASE_REPLACE: usize = 2;

const PET_STATES: [i64; 25] = [35, 22, 73, 29, 30, 33, 74, 68, 23, 24, 25, 41, 40, 66, 71, 72, 36, 98, 21, 34, 52, 64, 77, 82, 149];
const PCT: [i64; 7] = [73, 23, 24, 25, 41, 40, 66];
const STRONG: [i64; 10] = [1, 5, 8, 0, 3, 2, 9, 4, 6, 7];
const WEAK: [i64; 10] = [7, 3, 1, 8, 0, 9, 4, 2, 5, 6];
const WEAK_MULT: f64 = 0.66;
const S_ACCURACY: i64 = 41;
const S_DODGE: i64 = 73;

#[derive(Clone, PartialEq)]
pub struct Obs {
    pub glob: [f32; N_GLOB],
    pub pet_int: [i64; 6],
    pub pet_float: [[f32; N_PET]; 6],
    pub ab_int: [[i64; 3]; N_ABIL],
    pub ab_float: [[f32; N_ABF]; N_ABIL],
    pub ab_mask: [f32; N_ABIL],
    pub aura_int: [[i64; 2]; MAX_AURAS],
    pub aura_float: [[f32; N_AUF]; MAX_AURAS],
    pub aura_mask: [f32; MAX_AURAS],
}

impl Obs {
    /// Hash of every input value (for merging identical requests before the network).
    pub fn hash(&self, mask: &[f32; 7]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| {
            h ^= x;
            h = h.wrapping_mul(0x0100_0000_01b3);
            h ^= h >> 29;
        };
        for &v in self.glob.iter().chain(mask.iter()) {
            mix(v.to_bits() as u64);
        }
        for &v in &self.pet_int {
            mix(v as u64);
        }
        for r in &self.pet_float {
            for &v in r {
                mix(v.to_bits() as u64);
            }
        }
        for r in &self.ab_int {
            for &v in r {
                mix(v as u64);
            }
        }
        for r in &self.ab_float {
            for &v in r {
                mix(v.to_bits() as u64);
            }
        }
        for r in &self.aura_int {
            for &v in r {
                mix(v as u64);
            }
        }
        for r in &self.aura_float {
            for &v in r {
                mix(v.to_bits() as u64);
            }
        }
        for &v in self.ab_mask.iter().chain(self.aura_mask.iter()) {
            mix(v.to_bits() as u64);
        }
        h
    }

    fn zero() -> Obs {
        Obs {
            glob: [0.0; N_GLOB],
            pet_int: [0; 6],
            pet_float: [[0.0; N_PET]; 6],
            ab_int: [[0; 3]; N_ABIL],
            ab_float: [[0.0; N_ABF]; N_ABIL],
            ab_mask: [0.0; N_ABIL],
            aura_int: [[0; 2]; MAX_AURAS],
            aura_float: [[0.0; N_AUF]; MAX_AURAS],
            aura_mask: [0.0; MAX_AURAS],
        }
    }
}

fn effectiveness(ab_family: i64, target_family: i64) -> f64 {
    if ab_family < 0 || ab_family >= 10 {
        return 0.0;
    }
    if STRONG[ab_family as usize] == target_family {
        return 1.0;
    }
    if WEAK[ab_family as usize] == target_family {
        return -1.0;
    }
    0.0
}

#[inline]
fn clamp3(x: f64) -> f64 {
    if x < -3.0 {
        -3.0
    } else if x > 3.0 {
        3.0
    } else {
        x
    }
}

/// _ability_float. opp: (family, max hp, hp) of the pet the ability would hit.
fn ability_float(f: &mut [f32; N_ABF], pet: &Pet, power: i64, slot: usize, aid: u32, known: bool, opp: Option<(i64, i64, i64)>, legal_now: bool) {
    let a = gd().ab(aid);
    f[0] = 1.0;
    f[1] = known as i32 as f32;
    f[2] = if known { (pet.cooldowns[slot] as f64 / 5.0) as f32 } else { 0.0 };
    f[3] = legal_now as i32 as f32;
    if let Some((ofam, omhp, ohp)) = opp {
        let fam = a.map(|x| x.family).unwrap_or(-1);
        let eff = effectiveness(fam, ofam);
        f[4] = eff as f32;
        let mult: f64 = if eff > 0.0 { 1.5 } else if eff < 0.0 { WEAK_MULT } else { 1.0 };
        let d13 = a.map(|x| x.d13).unwrap_or(0.0);
        // numpy float32: d[13] * 50 * (1 + power / 20) * mult
        let raw: f32 = d13 * 50.0f32 * ((1.0 + power as f64 / 20.0) as f32) * (mult as f32);
        let r5 = raw / (omhp.max(1) as f32);
        let r6 = raw / (ohp.max(1) as f32);
        f[5] = if r5 < 3.0 { r5 } else { 3.0 };
        f[6] = if r6 < 3.0 { r6 } else { 3.0 };
    }
    f[7] = matches!(pet.channel, Some((s, _)) if s == slot) as i32 as f32;
    f[8] = (pet.last_ability == Some(aid)) as i32 as f32;
    f[9] = (pet.revealed & (1 << slot) != 0) as i32 as f32;
}

pub fn action_mask(b: &Battle, side: usize, phase: usize) -> [f32; 7] {
    let mut m = [0f32; 7];
    if phase == PHASE_LEAD {
        for i in 0..b.teams[side].pets.len() {
            m[3 + i] = 1.0;
        }
        return m;
    }
    for a in b.legal_actions(side) {
        m[a] = 1.0;
    }
    m
}

pub fn encode(b: &Battle, side: usize, phase: usize, hide: bool) -> Obs {
    let mut o = Obs::zero();
    let lead = phase == PHASE_LEAD;
    let my_act = if lead { None } else { Some(b.active(side)) };
    let en_act = if lead { None } else { Some(b.active(1 - side)) };
    let legal: Vec<usize> = if phase == PHASE_ROUND { b.legal_actions(side) } else { Vec::new() };

    // (max hp, power, speed) per pet
    let mut stats = [[(0i64, 0i64, 0i64); 3]; 2];
    for s in 0..2 {
        for i in 0..b.teams[s].pets.len() {
            let r = PetRef { side: s, idx: i };
            stats[s][i] = (b.max_hp(r), b.power(r), b.speed(r));
        }
    }
    let st = |r: PetRef| stats[r.side][r.idx];
    let mut auras: Vec<(&Aura, i64)> = Vec::new();
    for (ti, tside) in [side, 1 - side].into_iter().enumerate() {
        let team = &b.teams[tside];
        for (pi, pet) in team.pets.iter().enumerate() {
            let s = ti * 3 + pi;
            let r = PetRef { side: tside, idx: pi };
            let (mhp, pw, spd) = st(r);
            o.pet_int[s] = pet.family + 1;
            let f = &mut o.pet_float[s];
            let active = !lead && team.active == pi;
            f[0] = 1.0;
            f[1] = (pet.hp as f64 / mhp.max(1) as f64) as f32;
            f[2] = (mhp as f64 / 2000.0) as f32;
            f[3] = (pw as f64 / 400.0) as f32;
            f[4] = (spd as f64 / 400.0) as f32;
            f[5] = pet.dead as i32 as f32;
            f[6] = active as i32 as f32;
            f[7] = pet.undead_used as i32 as f32;
            f[8] = pet.mech_revived as i32 as f32;
            f[9] = pet.channel.is_some() as i32 as f32;
            f[10] = (pet.hp as f64 / 2000.0) as f32;
            f[11] = if active { ((b.round - pet.swapped_in).min(10) as f64 / 10.0) as f32 } else { 0.0 };
            let opp = if ti == 0 { en_act } else { my_act };
            if let Some(op) = opp {
                let ospd = st(op).2;
                f[12] = ((spd > ospd) as i32 - (spd < ospd) as i32) as f32;
                let ofam = b.pet(op).family;
                f[13] = effectiveness(pet.family, ofam) as f32;
                f[14] = effectiveness(ofam, pet.family) as f32;
            }
            if !pet.dead {
                for (j, &sid) in PET_STATES.iter().enumerate() {
                    let v = b.state(r, sid);
                    f[15 + j] = clamp3(if PCT.contains(&sid) { v as f64 / 100.0 } else { v as f64 }) as f32;
                }
                let natural = b.state(r, S_TYPE_OVERRIDE) == 0;
                let undead_life = pet.family == UNDEAD && !pet.undead_used && natural;
                let mech_life = pet.family == MECHANICAL && !pet.mech_revived && natural;
                f[40] = (undead_life || mech_life) as i32 as f32;
                let eff_hp = pet.hp as f64 + if mech_life { 0.2 * mhp as f64 } else { 0.0 };
                let x = eff_hp / mhp.max(1) as f64;
                f[41] = (if x < 2.0 { x } else { 2.0 }) as f32;
                f[42] = (pet.family == CRITTER || b.state(r, S_RESILIENT) > 0) as i32 as f32;
            }
            for a in &pet.auras {
                auras.push((a, s as i64));
            }
        }
        for a in &team.auras {
            auras.push((a, 6 + ti as i64));
        }
    }
    if let Some(w) = &b.weather {
        auras.push((w, 8));
    }

    let target = |p: Option<PetRef>| -> Option<(i64, i64, i64)> {
        let p = p?;
        let pet = b.pet(p);
        if pet.dead {
            return None;
        }
        Some((pet.family, st(p).0, pet.hp))
    };
    let (t_en, t_me) = (target(en_act), target(my_act));
    let hints = if phase == PHASE_ROUND && my_act.is_some() { Some(b.probe(side, &legal)) } else { None };
    let mut k = 0;
    for (pi, pet) in b.teams[side].pets.iter().enumerate() {
        let pw = stats[side][pi].1;
        let is_act = my_act.map(|r| r.idx == pi).unwrap_or(false);
        for sl in 0..3 {
            let aid = pet.abilities[sl];
            if aid != 0 {
                o.ab_int[k] = [aid as i64, pi as i64, sl as i64];
                let act_now = is_act && legal.contains(&sl);
                ability_float(&mut o.ab_float[k], pet, pw, sl, aid, true, t_en, act_now);
                if let Some(h) = &hints {
                    if is_act {
                        for j in 0..N_PROBE {
                            o.ab_float[k][10 + j] = h[sl][j];
                        }
                    }
                }
                o.ab_mask[k] = 1.0;
            }
            k += 1;
        }
    }
    let es = 1 - side;
    for (pi, pet) in b.teams[es].pets.iter().enumerate() {
        let opts = &gd().species[&pet.species].abilities;
        let pw = stats[es][pi].1;
        for sl in 0..3 {
            let aid = pet.abilities[sl];
            if aid == 0 {
                k += 2;
                continue;
            }
            if !hide || pet.revealed & (1 << sl) != 0 {
                o.ab_int[k] = [aid as i64, 3 + pi as i64, sl as i64];
                ability_float(&mut o.ab_float[k], pet, pw, sl, aid, true, t_me, false);
                o.ab_mask[k] = 1.0;
            } else if let Some(cands) = opts.get(sl) {
                for (j, &cand) in cands.iter().take(2).enumerate() {
                    o.ab_int[k + j] = [cand as i64, 3 + pi as i64, sl as i64];
                    ability_float(&mut o.ab_float[k + j], pet, pw, sl, cand, false, t_me, false);
                    o.ab_mask[k + j] = 1.0;
                }
            }
            k += 2;
        }
    }

    for (i, (a, holder)) in auras.iter().take(MAX_AURAS).enumerate() {
        o.aura_int[i] = [a.aid as i64, *holder];
        let f = &mut o.aura_float[i];
        f[0] = if a.duration >= 0 { (a.duration as f64 / 5.0) as f32 } else { 0.0 };
        f[1] = (a.duration < 0) as i32 as f32;
        f[2] = (a.stacks as f64 / 3.0) as f32;
        f[3] = a.caster.map(|c| c.side == side).unwrap_or(false) as i32 as f32;
        f[4] = (a.age.min(10) as f64 / 10.0) as f32;
        f[5] = (a.blocks_left as f64 / 3.0) as f32;
        f[6] = a.next_attack as i32 as f32;
        f[7] = 1.0;
        o.aura_mask[i] = 1.0;
    }

    o.glob[phase] = 1.0;
    o.glob[3] = (b.round as f64 / 30.0) as f32;
    o.glob[4] = (b.teams[side].pets.iter().filter(|p| !p.dead).count() as f64 / 3.0) as f32;
    o.glob[5] = (b.teams[es].pets.iter().filter(|p| !p.dead).count() as f64 / 3.0) as f32;
    if let (Some(m), Some(e)) = (my_act, en_act) {
        o.glob[6] = (st(m).2 > st(e).2) as i32 as f32;
        o.glob[7] = (st(m).2 == st(e).2) as i32 as f32;
        o.glob[8] = b.pet(e).channel.is_some() as i32 as f32;
        o.glob[9] = (b.state(m, S_TURN_LOCK) > 0) as i32 as f32;
    }
    o.glob[10] = b.weather.is_some() as i32 as f32;
    o.glob[11] = b.needs_replacement(es) as i32 as f32;
    o
}

// ------------------------------------------------------------------------------------ probe
impl Battle {
    fn probe_run(&self, side: usize, action: usize, expect: bool) -> (i64, bool, i64, Vec<(u8, usize, i64)>) {
        let mut c = self.clone_with_seed(0);
        c.logging = false;
        c.log.clear();
        c.force_hit = Some(true);
        c.force_crit = Some(false);
        c.force_variance = Some(0.0);
        c.probe_expect = expect;
        c.probe_counts = Some(Vec::new());
        let me = c.active(side);
        let en = c.active(1 - side);
        let life = |p: &Pet| (p.hp, p.dead, p.undead_used, p.mech_revived);
        let b_me = life(c.pet(me));
        let b_en = life(c.pet(en));
        let mut acts = [A_PASS, A_PASS];
        acts[side] = action;
        c.step(acts[0], acts[1]);
        let pe = c.pet(en);
        let took = (pe.dead && !b_en.1) || (pe.undead_used && !b_en.2) || (pe.mech_revived && !b_en.3);
        let loss = if took { b_en.0 } else { b_en.0 - pe.hp };
        let pm = c.pet(me);
        let own = if pm.dead { -b_me.0 } else { pm.hp - b_me.0 };
        (loss, took, own, c.probe_counts.take().unwrap_or_default())
    }

    fn hit_chance(&self, pet: PetRef, target: PetRef, aid: u32) -> f64 {
        let acc = match gd().ab(aid).and_then(|a| a.first_acc) {
            Some(a) => a,
            None => return 1.0,
        };
        let mut acc = acc + self.state(pet, S_ACCURACY) - self.state(target, S_DODGE);
        let lvl = self.pet(target).level - self.pet(pet).level;
        if lvl > 0 {
            acc -= lvl * 2;
        }
        (acc as f64 / 100.0).max(0.0).min(1.0)
    }

    /// bp.pvp.probe.probe: (3, 8) hint rows for the active pet's usable abilities.
    pub fn probe(&self, side: usize, legal: &[usize]) -> [[f32; N_PROBE]; 3] {
        let mut out = [[0f32; N_PROBE]; 3];
        let me = self.active(side);
        let en = self.active(1 - side);
        if self.over() || self.pet(me).dead || self.pet(en).dead || self.pet(me).channel.is_some() {
            return out;
        }
        let slots: Vec<usize> = legal.iter().copied().filter(|&a| a < 3).collect();
        if slots.is_empty() {
            return out;
        }
        let (base_loss, base_took, base_own, base_c) = self.probe_run(side, A_PASS, true);
        let mhp_en = self.max_hp(en).max(1) as f64;
        let mhp_me = self.max_hp(me).max(1) as f64;
        let es = 1 - side;
        let count = |c: &Vec<(u8, usize, i64)>, kind: u8| c.iter().filter(|(k, s, _)| *k == kind && *s == es).map(|x| x.2).sum::<i64>();
        let mut base_forced: Option<bool> = None;
        for sl in slots {
            let (loss, took, own, cnt) = self.probe_run(side, sl, true);
            let f = &mut out[sl];
            f[0] = 1.0;
            f[1] = ((loss - base_loss) as f64 / mhp_en).max(-1.0).min(3.0) as f32;
            if count(&cnt, PROBE_ROLLED) != 0 {
                let bf = *base_forced.get_or_insert_with(|| self.probe_run(side, A_PASS, false).1);
                let t2 = self.probe_run(side, sl, false).1;
                f[2] = (t2 && !bf) as i32 as f32;
            } else {
                f[2] = (took && !base_took) as i32 as f32;
            }
            f[3] = ((own - base_own) as f64 / mhp_me).max(-1.0).min(1.0) as f32;
            f[4] = (count(&cnt, PROBE_IMMUNE) > count(&base_c, PROBE_IMMUNE)) as i32 as f32;
            f[5] = (count(&cnt, PROBE_UNREACHABLE) > count(&base_c, PROBE_UNREACHABLE)) as i32 as f32;
            f[6] = (count(&cnt, PROBE_BLOCKED) > count(&base_c, PROBE_BLOCKED)) as i32 as f32;
            f[7] = self.hit_chance(me, en, self.pet(me).abilities[sl]) as f32;
        }
        out
    }
}
