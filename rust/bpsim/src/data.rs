//! Game data (data/processed/sim_rust.json, written by bp.sim.export_rust).

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Effect parameter keys the engine reads (anything else in the DB2 rows is ignored).
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum K {
    Points, Accuracy, Variance, IsPeriodic, ChainFailure, RequiredTargetState, RequiredCasterState,
    CasterState, TargetState, BonusState, BonusPoints, WeatherState, PointsMax, PointsIncreasePerUse,
    Power, Boost, BonusDamage, AuraID, Duration2, Threshold, Modifier, Percentage, RequiredTargetPetType,
    RequiredCasterPetType, Duration, MaxAllowed, RequiredPetType, StatePoints, NewDuration, Chance, State,
    StateValue, StateChange, StateMin, StateMax, TargetTestState, TargetTestStateValue, CasterStateValue,
    TargetStateValue, CasterImmuneState, TargetImmuneState, EnableReverse, Index, CooldownModification,
    LockDuration, Slot, Ability, MaxPoints,
    Count,
}

const KEY_NAMES: [&str; K::Count as usize] = [
    "Points", "Accuracy", "Variance", "IsPeriodic", "ChainFailure", "RequiredTargetState", "RequiredCasterState",
    "CasterState", "TargetState", "BonusState", "BonusPoints", "weatherState", "PointsMax", "PointsIncreasePerUse",
    "Power", "Boost", "BonusDamage", "AuraID", "Duration2", "Threshold", "Modifier", "Percentage", "RequiredTargetPetType",
    "RequiredCasterPetType", "Duration", "MaxAllowed", "RequiredPetType", "StatePoints", "NewDuration", "Chance", "State",
    "StateValue", "StateChange", "StateMin", "StateMax", "TargetTestState", "TargetTestStateValue", "CasterStateValue",
    "TargetStateValue", "CasterImmuneState", "TargetImmuneState", "enableReverse", "Index", "CooldownModification",
    "LockDuration", "Slot", "Ability", "MaxPoints",
];

#[derive(Clone)]
pub struct Params {
    v: [i64; K::Count as usize],
    has: u64,
}

impl Params {
    /// `p.get(k)` (None when absent).
    #[inline]
    pub fn get(&self, k: K) -> Option<i64> {
        let i = k as usize;
        if self.has & (1u64 << i) != 0 { Some(self.v[i]) } else { None }
    }
    /// `p.get(k, default)`.
    #[inline]
    pub fn gd(&self, k: K, d: i64) -> i64 {
        self.get(k).unwrap_or(d)
    }
    /// truthiness of `p.get(k)`.
    #[inline]
    pub fn t(&self, k: K) -> bool {
        self.get(k).unwrap_or(0) != 0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Code {
    A(u8),
    AP,
    E(u8),
    EP,
    C,
    T,
}

pub struct Effect {
    pub op: i64,
    pub aura: u32,
    pub p: Params,
    pub code: Option<Code>, // per-row target (targets.plan)
    pub chain: bool,        // targets.chain_rows
}

pub struct Turn {
    pub typ: i64,
    pub event: i64,
    pub effects: Vec<Effect>,
}

pub struct Ability {
    pub id: u32,
    pub name: String,
    pub family: i64,
    pub cooldown: i64,
    pub states: Vec<(i64, i64)>,
    pub turns: Vec<Turn>,
    pub main: Vec<usize>, // indices of type-0 turns
    pub is_object: bool,
    pub block_count: i64,
    pub needs_corpse: bool,
    pub writes: Vec<i64>, // states written by ops 31 / 79 / 360
}

impl Ability {
    #[inline]
    pub fn state(&self, sid: i64) -> i64 {
        for &(k, v) in &self.states {
            if k == sid {
                return v;
            }
        }
        0
    }
    #[inline]
    pub fn has_state(&self, sid: i64) -> Option<i64> {
        self.states.iter().find(|(k, _)| *k == sid).map(|(_, v)| *v)
    }
}

pub struct Species {
    pub name: String,
    pub family: i64,
    pub states: Vec<(i64, i64)>,
}

pub struct GameData {
    pub abilities: Vec<Option<Ability>>,
    pub species: HashMap<u32, Species>,
}

impl GameData {
    #[inline]
    pub fn ab(&self, id: u32) -> Option<&Ability> {
        self.abilities.get(id as usize).and_then(|a| a.as_ref())
    }
    #[inline]
    pub fn ability(&self, id: u32) -> &Ability {
        self.ab(id).expect("unknown ability id")
    }
}

// ------------------------------------------------------------------------------------------ loading
#[derive(Deserialize)]
struct JEffect {
    op: i64,
    aura: i64,
    params: HashMap<String, i64>,
}
#[derive(Deserialize)]
struct JTurn {
    #[serde(rename = "type")]
    typ: i64,
    event: i64,
    effects: Vec<JEffect>,
}
#[derive(Deserialize)]
struct JAbility {
    name: String,
    family: i64,
    cooldown: i64,
    states: Vec<(i64, i64)>,
    turns: Vec<JTurn>,
    plan: Vec<(usize, usize, String)>,
    chain: Vec<(usize, usize)>,
}
#[derive(Deserialize)]
struct JSpecies {
    name: String,
    family: i64,
    states: Vec<(i64, i64)>,
}
#[derive(Deserialize)]
struct JAll {
    abilities: HashMap<String, JAbility>,
    species: HashMap<String, JSpecies>,
}

fn code(s: &str) -> Code {
    match s {
        "C" => Code::C,
        "T" => Code::T,
        "AP" => Code::AP,
        "EP" => Code::EP,
        _ => {
            let n = s[1..].parse::<u8>().unwrap_or(0);
            if s.starts_with('A') { Code::A(n) } else { Code::E(n) }
        }
    }
}

pub const S_OBJECT: i64 = 69;
pub const S_BOMB: i64 = 136;
pub const S_BLOCK_COUNT: i64 = 74;

pub fn load(path: &str) -> Result<GameData, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let all: JAll = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let max_id = all.abilities.keys().filter_map(|k| k.parse::<usize>().ok()).max().unwrap_or(0);
    let mut abilities: Vec<Option<Ability>> = (0..=max_id).map(|_| None).collect();
    for (k, a) in all.abilities {
        let id: u32 = k.parse().map_err(|_| "bad id")?;
        let plan: HashMap<(usize, usize), Code> = a.plan.iter().map(|(t, e, c)| ((*t, *e), code(c))).collect();
        let chain: std::collections::HashSet<(usize, usize)> = a.chain.iter().cloned().collect();
        let mut writes = Vec::new();
        let turns: Vec<Turn> = a
            .turns
            .into_iter()
            .enumerate()
            .map(|(ti, t)| Turn {
                typ: t.typ,
                event: t.event,
                effects: t
                    .effects
                    .into_iter()
                    .enumerate()
                    .map(|(ei, e)| {
                        let mut p = Params { v: [0; K::Count as usize], has: 0 };
                        for (i, name) in KEY_NAMES.iter().enumerate() {
                            if let Some(v) = e.params.get(*name) {
                                p.v[i] = *v;
                                p.has |= 1u64 << i;
                            }
                        }
                        if matches!(e.op, 31 | 79 | 360) {
                            if let Some(s) = p.get(K::State) {
                                writes.push(s);
                            }
                        }
                        Effect { op: e.op, aura: e.aura.max(0) as u32, p, code: plan.get(&(ti, ei)).copied(), chain: chain.contains(&(ti, ei)) }
                    })
                    .collect(),
            })
            .collect();
        let main: Vec<usize> = turns.iter().enumerate().filter(|(_, t)| t.typ == 0).map(|(i, _)| i).collect();
        let needs_corpse = main.first().map(|&i| turns[i].effects.first().map(|e| e.op == 144).unwrap_or(false)).unwrap_or(false);
        let st = |sid: i64| a.states.iter().find(|(k, _)| *k == sid).map(|(_, v)| *v).unwrap_or(0);
        let is_object = st(S_OBJECT) != 0 || st(S_BOMB) != 0;
        let block_count = st(S_BLOCK_COUNT);
        abilities[id as usize] = Some(Ability {
            id,
            name: a.name,
            family: a.family,
            cooldown: a.cooldown,
            states: a.states,
            turns,
            main,
            is_object,
            block_count,
            needs_corpse,
            writes,
        });
    }
    let species = all
        .species
        .into_iter()
        .map(|(k, s)| (k.parse::<u32>().unwrap_or(0), Species { name: s.name, family: s.family, states: s.states }))
        .collect();
    Ok(GameData { abilities, species })
}

static DATA: OnceLock<GameData> = OnceLock::new();

pub fn init(path: &str) -> Result<(), String> {
    if DATA.get().is_some() {
        return Ok(());
    }
    let d = load(path)?;
    let _ = DATA.set(d);
    Ok(())
}

#[inline]
pub fn gd() -> &'static GameData {
    DATA.get().expect("bpsim: call load_data() first")
}
