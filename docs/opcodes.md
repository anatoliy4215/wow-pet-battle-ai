# Battle-pet effect opcodes (BattlePetEffectProperties) — reference

Scope: every effect opcode used by abilities reachable from battle-pet species (`species.json` ability slots + tamer pets in `tamers.json` + family passives 236–245), plus the auras they apply, followed transitively. That is **1229 abilities/auras, 89 distinct opcodes, 2 700+ effect rows**. Build 12.1.0.69933.

Evidence sources, and the tag used for each in the entries below:
- **[DB2]**: our `abilities.json` (turns/effects/params) and tooltip templates.
- **[TT]**: Blizzard tooltip parser `SharedPetBattleTemplates.lua` (wow-ui-source, `Interface/AddOns/Blizzard_FrameXML/`). Each effect param label is exposed as a tooltip function `label(turn, effect[, abilityID])`, lower-cased, e.g. `points(1,3)` or `duration(1,2)`. `SimpleDamage(t,e) = points(t,e) * (1 + 0.05*Power)`. `StandardDamage(t,e)` = SimpleDamage plus a **±variance/200·dmg** range. `StandardHealing` = `points × (1+0.05·Power)` × healing-done/taken states 65/66. `abilityStateMod(state, aura)` = the aura's `states{}` value. `getProcIndex(PROC_x)` = index of the type-1 turn with that event.
- **[EMU]**: Pandaria 5.4.8 server emulator (`data/raw/refs/emu_BattlePetAbilityEffect.cpp`, `emu_PetBattle.cpp`). It is MoP-era and implements only about 20 handlers, but it gives the per-opcode **default target (CASTER/TARGET)**, hit/miss, aura and round-flow logic. MoP numbers may differ from retail.
- **[PS]**: petsim hand-written rules (`petsim_abilities.json`, `petsim_other.json` buffs keyed by `abilityId`, `petsim_main.js`).

Notation: `op24{Points=20,Accuracy=100}` lists only non-zero params. `@N` = effect's `aura` (AuraBattlePetAbilityID). `T1` = turn order 1 (type 0). `ev6` = type-1 turn with event 6.

---

## 0. Global rules (read first)

### 0.1 Targets are NOT fully encoded in DB2 (important)
- [EMU] Every opcode has a fixed default target: **CASTER** (heals, positive auras, set-state) or **TARGET** = the opponent's active pet (damage, negative/periodic auras). [EMU] also defines `ALL`, `CASTER_TEAM_UNCONDITIONAL` and `HEAD` (both active pets) target kinds.
- **Team-wide effects are written as N copies of the same effect.** Starfall has 3×op24 + 3×op23, Autumn Breeze 3×op23 + 3×op50. The engine hands successive copies to successive team members.
- **Identical data can still mean different targets.** Blessed Hammer (1100) and Spiritfire Beam (1035) are both `3×op24{Points=7|11}`. The first hits the active pet 3 times. The second hits active + 2 backline. So single-target multi-hit vs AoE is **not derivable from the effect rows**.
- The best in-data hint is the **effect's BattlePetVisual.RangeTypeEnum** (raw `BattlePetAbilityEffect.BattlePetVisualID` → `BattlePetVisual.RangeTypeEnum`; not currently copied into `abilities.json`). Measured on the 3+×op24 turns: RangeType 2 → team in 25/26 cases, RangeType 0 → single in 54/57 cases, no visual → ambiguous (≈30 cases).
- **Recommendation:** keep a hand-made `targets` override table per (ability, turn, effect). Seed it from petsim `targets` (`activeEnemy`, `backupEnemy1/2`, `petsEnemy`, `teamEnemy`=pad, `source`, `petsAlly`, `teamAlly`=pad, `battle`=weather, `allActive`, `previous`, `triggerSourcePet`…) and from tooltip wording ("to all enemies", "backline", "your team").
- **Team pad vs pet.** Objects (turrets, bombs, traps, walls, Wish, Tranquility, Decoy) sit on a team "pad" and persist through swaps. DB2 does not say so explicitly. Heuristics: the aura carries state 69 `Mechanic_IsObject` or 136 `Mechanic_IsBomb`, or its tooltip says "Persists through pet swaps". petsim marks these targets `teamAlly`/`teamEnemy`.
- **Weather** = op80 (global, replaces the previous weather). Weather auras carry `Weather_*` states 53–63 and 316.
- Inside **aura ticks and event triggers**, "caster" and "target" are relative to the aura and its trigger. Examples: Thorns op179 hits the attacker. Brittle Webbing's ev2 op24 hits the webbed holder. Plagued Blood's ev1 op23 heals the attacker. Per-aura overrides are needed here too.

### 0.2 Hit / miss / fail model
- [EMU] `acc = Accuracy + caster.State41(Stat_Accuracy) − target.State73(Stat_Dodge) − 2·max(0, lvlT−lvlC)`. The effect rolls only if `acc < 100`. **Accuracy is a plain percentage, and values above 100 are headroom against dodge.**
- **Accuracy 200** (op80 weather ×23, 3rd+ copies of op26/op50 team buffs, self-damage like Body Slam 532 `op24{Points=13,Accuracy=200}`, `op135{Accuracy=200}` kills): *can't miss*. It still passes against 100-dodge auras (Feign Death 804, Portal 1124, Fade 1083 = `Stat_Dodge 100`), but not against Deflection 903 (`Stat_Dodge 200`). [PS] maps `alwaysHit` to accuracy 200.
- **Accuracy 101** appears only on the heal half of Starfall/Moon Tears (op23). Treat it as "always hits" (low confidence on why it isn't 100).
- **Accuracy 0/absent** on control ops (op172, op107 swap) means the effect succeeds only through its state override, or the accuracy is irrelevant.
- **Accuracy −1**: test data only (85 "Apply Stun test").
- The **primary hit** is the first damage effect. Effects are executed in `order`. Each one rolls independently unless a gate says otherwise, so Triple Snap = `24{A100} ; 158 ; 24{A66} ; 24{A33}`.
- **ChainFailure=1** (param on op32/50/52/54/80/131/150/172/178/229/256): **skip this effect if the previous effect failed** (missed, dodged, immune, or blocked by a gate). [EMU] op32/op54: `if (ChainFailure && m_chainFailure) return;`. [PS] `chainFailure: true` = `previous.hasSucced == false → skip`. Example: Poison Fang 152 `24 ; 54@379{ChainFailure=1}` means no poison on a miss. op54 has 2 rows with ChainFailure=10 (DATA ERROR 615 test ability); treat any non-zero value as true. High confidence.
- **Explicit gate op158**: "stop the rest of this turn if the previous effect missed" ([PS] `stopIfMiss`). **op194**: inverse, "continue only if the previous effect missed" (Haymaker, Blingtron).
- **State gates** stop the rest of the turn or the effect: op156, op157, op177. See their entries.
- **Required/Caster/Target State params** (op29/61/96/104/169/171/172/229/230/247/248, op50's CasterState/TargetState): param value = a `states.json` id. Meaning = "the caster (or target) has non-zero state X". Examples: 1 Is_Dead, 21 Poisoned, 28 WasDamagedThisTurn, 34 Burning, 52 Chilled, 55 Moonlight, 56 Darkness, 61 Sunlight, 64 Webbed, 77 Bleeding, 82 Blind, 91 Internal_EffectSucceeded, 124 TypeOverride, 127 DidDamageThisRound, 149 resilient, 302 Dealt_KillingBlow. Weather states are visible on every pet while the weather is up.
- **Pet-type params** (op100 `RequiredCasterPetType/RequiredTargetPetType`, op223 `RequiredPetType`, state 89 `Mod_PetType_ID`, state 124 `Special_TypeOverride`) are **1-based** (family + 1): 2 Dragonkin, 5 Critter, 6 Magic, 7 Elemental, 9 Aquatic, 10 Mechanical. Evidence: Healing Stream 922 `RequiredTargetPetType=9` "aquatic allies"; Moonlight 596 `Mod_PetType_ID 6` "Magic abilities deal 10%"; Undead Mark 968 `TypeOverride 4`.

### 0.3 Aura parameters
- **Duration** is in rounds. [EMU] `AddAura`: `if (!duration && duration != -1) duration++`. So **Duration 0 = lasts only the current round**; this is used by the ev6 dodge/priority auras, which are removed explicitly at ev7 or ev12. **Duration −1 = no expiry**: the aura is removed only by its own logic or by a "next attack" rule. Examples: Soul Ward 751, Dark Rebirth 794/795, Bubble 934, Failsafe marker 723→244, Rake 492 `op50 Duration=-1`, Discharge 1718 `op229 Duration=-1`. [EMU] also expires `Duration==-1` auras once their holder has dealt damage this round (turn > 2), which is the "Attack Reduction … for the next attack" pattern (537).
- When the effect's Duration is 0 on op26/28 but the ability is multi-round, the length comes from the turn layout (Burrow: `T1 26@340{Duration=1}`, `T2 … 49@340`).
- **TickDownFirstRound** (op26/28/248/359 label): **always 0** in the reachable data, so it can be ignored for now. By name it probably means "the application round counts as the first tick" (low confidence).
- **MaxAllowed** (op50/54, emu passes it into `AddAura`): stack cap for **instances of this aura on the same target**. When a new application would exceed the cap, [EMU] expires the oldest instances first. Seen values: 1 (Rake, Chop 943, Wild Winds 514, Gale Slash, Stench, Inebriate), 3 (Build Turret 710 / Raise Ally 2298 / Rapid Turret 2413 = up to 3 turrets; Volcano, Spore Shrooms), 5 (Blistering Cold 785 → Frostbite 787 stacks), 9 (Firebomb 1919), 100 (test). **MaxAllowed 0** is the common case: [EMU] treats it as "no cap", but in retail re-applying a DoT refreshes it rather than stacking. Recommended engine rule: 0 ⇒ 1 (replace/refresh). Medium confidence.
- **IsPeriodic** (op24/29/33/59/62/65/68/149/169/170/179/204/363 …): marks the damage as **periodic (DoT) rather than direct**. [EMU] sets `FLAG_PERIODIC`. [PS] `isPeriodic` ⇒ not "direct damage", so it doesn't proc "each time you take direct damage" (Thorns and similar, ev1 op179/204). It is also what the Aquatic passive (−50% DoT taken, +25% DoT dealt) and Cleansing Rain key on. Set on DoT auras such as Poisoned 379, Bleeding 491, Immolate 177, Haunt 653 and Pheromones 1064. Non-periodic tick damage also exists (Glowing Toxin 271: `op62 IsPeriodic=0`).
- **OverrideIndex** (op24/65/179/201/204/222/257): set to 1 on the **2nd and later hits of multi-hit volleys** (Stampede 163 `24 ; 24* ; 24*`, Rock Barrage, Turret 711, Rapid Turret 2427, Tympanic Tantrum T3, Quake, Omnislash's 4th hit). Likely meaning: "don't start a new attack instance / reuse the previous effect's hit index", so the volley counts as one attack for blocks, dodges and combat-log grouping. Low–medium confidence; it is not a target selector (Quake uses it and is AoE).
- **Variance** (op23/61/100/104/171/222/254/358/362): **total spread in percent**. [TT] `variance/200 * damage` = ±Variance/2. op222 Burn `Variance=40` → ±20% ([PS] `variance: "20%"`). op23 `Variance=20` → ±10% ([PS] `"10%"`). Values seen: 20 (heals), 40/50/60/80/30 (op222). High confidence.
- **Chance** (op49): percent chance for the aura removal to happen. Snap Trap 671 / Magma Trap 810 `ev9: op49@self{Chance=25}` = "25% on each enemy ability"; Table of Discontents 33%. **0 behaves as 100** (Belly Slide 529 / When Elekks Fly 1412 / Haunt 653 remove their auras with Chance 0 and must always do so). **200 = unconditional**, used for self-consumption (Burrow 159 `T2 49@340{Chance=200}`, Supercharged 207, Overpowered 1721, Minefield 635, Dark Rebirth 795). Treat ≥100 and 0 as "always". The `unused` param (100/20 on 588/612/723/2252) has no known effect.
- **ReportFailsAsImmune** (op73/178): when the state gate fails, show "Immune" instead of silently skipping ([EMU] op178).

### 0.4 Round flow (from [EMU] `PetBattle::HandleRound` + event enum)
1. Swaps / move selection. Reset states 28 `WasDamagedThisTurn` and 127 `DidDamageThisRound`.
2. **ev6 ROUND_START** triggers of the chosen ability (Deflection, Quick Attack speed +10000 via op79 → op116 re-sorts initiative).
3. Abilities execute in speed order. A multi-round ability runs turn `order = roundsPassed`. On the first cast, `BattlePetAbilityState` values are added to the caster.
4. **ev7 ROUND_END** triggers of the chosen ability (undo the priority aura, swap-out abilities).
5. Aura processing: each aura runs its type-0 turn (its periodic tick) and ticks down.
6. **ev12** triggers (post-aura-tick; Feign Death/Warning!/Return to Rest swap happens here).
7. Cooldowns −1, expired auras removed; ev5 fires for each removed aura.

---

## 1. Opcodes (most-used first)

Legend: **uses** = effect rows / distinct abilities in the reachable set. **Target** = default target, before any team replication or overrides.

### op24 — damage (standard) · uses 878/510 · target: enemy active · **high**
Params: `Points` (base damage; [TT] `StandardDamage(t,e)`/`SimpleDamage` read it), `Accuracy`, `IsPeriodic` (DoT tick = not direct damage), `OverrideIndex` (follow-up hit of a volley, §0.3). [EMU] `HandleDamage`, TARGET. Damage = Points × (1+0.05·Power) × damage-dealt/taken mods × type modifier; crits and passives apply.
- Bite 110 `op24{Points=20,Accuracy=100}` "Bites … dealing [StandardDamage(1,1)]"; [PS] `damage activeEnemy`.
- Demolish 390 `{Points=40,Accuracy=50}` [PS] acc=50 pow=40.
- Poisoned 379 (aura tick) `{Points=5,IsPeriodic=1}` "every round"; [PS] endTurn damage.
- Blessed Hammer 1100 vs Spiritfire Beam 1035: identical `3×op24`, but single-target vs AoE (§0.1).

### op26 — apply positive aura (caster) · 201/180 · target: caster · **high**
Params: `Unused` (15/25 on 2 rows, no known use), `Accuracy`, `Duration` (0 = this round, −1 = infinite), `TickDownFirstRound` (always 0). `aura` = the aura ability id. [EMU] `HandlePositiveAura`, CASTER.
- Crouch 165 `85{State=161} ; 26@164{Duration=3}` "reducing all damage taken by 50% for [duration(1,2)]". Aura 164 has `Mod_DamageTakenPercent −50`, matching [PS] buff.
- Dazzling Dance 366 `26@365{A100,D9} ; 26@365{A200,D9} ; 26@365{A200,D9}` "Increases your team's speed by [abilityStateMod(25,365)]%": 3 copies = whole team.
- Burrow 159 `T1 26@340{Duration=1}` (Underground: `untargettable`, `IsUnderground`), then `T2 222 ; 49@340`.
- Righteous Inspiration 1378 `ev6: 26@853` (SpeedPriority aura, `Stat_Speed 99999`).

### op50 — apply negative aura (target) · 190/135 · target: enemy active · **high**
Params: `ChainFailure`, `Accuracy`, `Duration`, `MaxAllowed`, `CasterState`, `TargetState`. [EMU] `HandleNegativeAura`, TARGET; MaxAllowed is passed to AddAura. `CasterState` = required caster state (Moth Balls 507 `CasterState=127 DidDamageThisRound`). TargetState is always 0 in the data.
- Weakness 471 `24 ; 50@470{ChainFailure=1,A100,D1}` "reduces speed by [abilityStateMod(25,470)]%"; [PS] buff `previous`, `chainFailure`.
- Flyby 515 `24 ; 50@516{D3}` "[abs(abilityStateMod(24,516))]% for [duration(1,2)]": Duration is effect 2's param.
- Bombing Run 647 `50@819{D3}`. Aura 819 is an object (state 69) → **enemy pad** ([PS] `teamEnemy`); it deals damage on ev5 "After [duration(1,2)] rounds".
- Autumn Breeze 964: `3×23 ; 3×50@963` = heal own team, debuff enemy team.

### op23 — heal · 109/65 · target: caster · **high**
Params: `Points`, `Accuracy`, `Variance` (±V/2 %). [EMU] `HandleHeal`, CASTER. Healing = Points × (1+0.05·Power) × state 65 (healing dealt) / 66 (healing taken).
- Gift of the Naaru 1410 `{Points=40,Variance=20}` "Restores [StandardDamage(1,1)]"; [PS] heal source, 10%.
- Emerald Dream aura 599 `T1 {10} T2 {20} T3 {40}`: an aura's consecutive turns = its consecutive ticks ("increasing health every round").
- Plagued Blood 658 `ev1: op23{Points=4}` "When damaged, **the attacker** is healed": event context changes the target.

### op49 — remove aura · 108/74 · target: holder of `aura` (usually caster/self) · **high**
Params: `Chance` (0 ⇒ 100, 200 = unconditional; §0.3), `unused`. `aura` = aura id to remove. Removal fires that aura's **ev5** turn.
- Burrow 159 `T2: 222 ; 49@340{Chance=200}` ends the Underground aura after the strike.
- Snap Trap 671 `ev9: 49@671{Chance=25}`: 25% on an enemy ability to remove itself, which fires `ev5: 24 ; 178@927 (stun)`. [PS] `removeBuff onAbility 25% → removed: damage + stun`.
- Supercharged 207 `ev2: 49@207{Chance=200}` is consumed by the next damage dealt.
- Undead 242 `T1: 49@822` removes Polymorph when the passive fires (see §4).

### op222 — damage with variance · 88/78 · target: enemy · **high**
Params: `Points`, `Accuracy`, `Variance`, `OverrideIndex`. Same as op24 plus a random ±Variance/2 %.
- Burn 113 `{Points=21,Variance=40}` "[StandardDamage(1, 1)]" (tooltip renders the range); [PS] `variance: "20%"`.
- Cyclone 189 `3×{Accuracy=35,Variance=…}` (aura).
- Feathered Frenzy 1398 `{Points=25,Variance=60}` "[SimpleDamage(1,2)*0.7] to [SimpleDamage(1,2)*1.3]" = ±30%, which confirms V/2.

### op31 — set state (absolute) · 74/34 · target: caster · **high**
Params: `State`, `StateValue` (absent = 0 = clear). [EMU] `HandleSetState`, CASTER. Used for charge/recover flags and per-round memory.
- Counterstrike 158 `ev6: 31{State=28}` (reset "was struck"), `ev1: 31{State=28,StateValue=100}`, then `T1: 24 ; 29{RequiredCasterState=28}` "additional damage if you were struck this round".
- Solar Beam 753 `T2/T3: 31{114,1} ; 31{114}` (IsRecovering pulse = "must recharge").
- Aged Yolk 667 `31{141,1} ; 136 ; 136 ; 31{141}` (IsCleansing wrapper around the dispel).
- Consume 160 `ev6: 31{State=32}` clears Last_HitDealt before `32` heals from it.

### op54 — apply periodic (DoT/object) aura to target · 73/60 · target: enemy (or enemy/own pad) · **high**
Params: `ChainFailure`, `Accuracy`, `Duration`, `MaxAllowed`. [EMU] `HandlePeriodicTrigger` (checks ChainFailure), TARGET.
- Poison Fang 152 `24 ; 54@379{ChainFailure=1,D5}` "Poisons … [StandardDamage(1,1,379)] per round for [duration(1,2)]"; [PS] `chainFailure`.
- Chop 943 `24@491 ; 54@491{ChainFailure=1,D5,MaxAllowed=1}` Bleed, max 1.
- Build Turret 710 `54@711{D4,MaxAllowed=3}`: object on the **caster's** pad ([PS] `teamAlly`), even though op54 defaults to target. Needs an override.
- Blistering Cold 785 (weather-like aura) `54@787{D3,MaxAllowed=5} ; 140@787{NewDuration=3}`.

### op158 — gate: stop turn if previous effect missed · 66/64 · **high**
Params: `Chance` (always 0, unused). No target.
- Death Coil 121 `24 ; 158 ; 32{50}`: no self-heal on a miss.
- Triple Snap 355 `24 ; 158 ; 24{A66} ; 24{A33}` [PS] `stopIfMiss`, `$hasXPercent 66/33`.
- Pounce 535 `24 ; 158 ; 384`.

### op22 — no-op / data holder · 54/37 · **high**
Params: `Points`, `MorePoints`, `EvenMorePoints`. [EMU] HandleNull. Pads empty rounds (bombs/stuns `T1 22{MorePoints=100}`) and stores numbers for tooltips.
- Stunned 174/927 `T1 22{MorePoints=100}`.
- Nevermore 738 `T1 22{Points=5}` "locked out for [points(1,1)] rounds".
- Snap Trap 671 `22{Points=30,EvenMorePoints=3}`.
- Time Bomb 601 `T1–T3 22 ; T4 24 ; 178` = "Detonates after 3 rounds".

### op116 — recalculate stats / initiative · 53/23 · target: caster (one row per pet) · **medium**
No params (`p1=100` on some). Always follows a speed or max-health change: Quick Attack 563 `ev6: 79{State=20,+10000} ; 116` / `ev7: 79{−10000} ; 116`. Sunny Day 403 `ev0: 6×116` (max health +50% for all 6 pets). Perk Up 576 `26@577 ; 3×116` (team max-health aura). Likely "re-evaluate derived stats (max HP) and turn order".

### op52 — apply control aura (stun/root/flee-lock) to target · 43/42 · target: enemy · **medium-high**
Params: `ChainFailure`, `Accuracy` (= proc chance, e.g. 25), `Duration`. [EMU] null.
- Headbutt 376 `24 ; 177{State=149} ; 52@927{ChainFailure=1,Accuracy=25,D1}` "25% chance to stun"; [PS] `stats.accuracy 25`.
- Time Stop 600 `52@927{D1}`.
- Skunky Brew 1070 `52@1069{ChainFailure=1,D5}` "impossible to flee".
- Haymaker 762 `24{A50} ; 194 ; 159 ; 52@174{D1}` stuns **the user** on a miss (target override).

### op177 — gate: target immune if it has State · 28/27 · **medium-high**
Params: `State` (always 149 `resilient` = post-stun immunity from aura 924). Put before stun effects. If the target has the state, the rest fails as "immune", and the following ChainFailure=1 effect is skipped.
- Screeching Gears 754 `24 ; 177{149} ; 52@927{ChainFailure=1,A25}`.
- Kick 307 `24 ; 177{149} ; 131@927`.
- Volcano 175 T3/T4.

### op85 — remove auras that carry State X · 28/21 · target: target pet/pad (dup rows = both sides) · **medium-high**
Params: `State` (`p1=100` on some rows). Dispel by state; it is the "replace the previous one" step before re-applying.
- Jackhammer 2247 `85{State=71} ; 4×24` "shatters any flat damage reduction effects" (71 Add_FlatDamageTaken).
- Magma Wave 319 / Clean-Up 456 / Tidal Wave 419 `2×85{State=69,p1=100}` remove objects from both teams ([PS] `removeBuff teamAlly` + `teamEnemy`).
- Nature's Ward 574 / Discharge 1717 `85{State=124}` remove the old type override.
- Crouch 165 / Extra Plating 392 `85{State=161}`: state 161 is not in BattlePetState, presumably a hidden "damage-reduction family" tag. Engine: remove auras from the same "group".

### op28 — apply aura to caster (variant; 0 = until removed) · 25/23 · target: caster (or team) · **medium**
Params: `Unused`, `Accuracy`, `Duration`, `TickDownFirstRound`. Used for (a) round-scoped dodge/priority auras at ev6 that ev7/ev12 remove explicitly: Portal 1125 `ev6: 28@1124 ; ev7: 145 ; 49@1124`, Belly Slide 529 `ev6: 157{58,1} ; 28@853 ; 116`. (b) charge shields: Shield Block 760 `28@759` "blocking the next attack" ([PS] `buffDuration −1`), Decoy 334 `28@333` (pad object, [PS] `teamAlly`), Shieldstorm 1101 `3×28{D8}` (team). Duration 0 behaves like "until consumed/removed".

### op178 — apply aura unless target already has state value (stun with immunity) · 24/24 · target: enemy · **high**
Params: `StatePoints`, `Accuracy`, `Duration`, `TargetState`, `ChainFailure`, `ReportFailsAsImmune`. [EMU] `HandlePowerlessAura`: if `target.State[TargetState] == StatePoints` → immune (or silent skip when ReportFailsAsImmune=0).
- Blackout Kick 227 `178@927{StatePoints=1,D1,TargetState=149,Report=1}` "Stun the target for [duration(1,1)]".
- Drowsy 496 `T2: 178@926{D2}` falls asleep.
- Whirlwind 2447 `178@927{Accuracy=50}` "[accuracy(1,1)]% chance to stun".

### op79 — modify state by delta (clamped) · 24/14 · target: caster · **high**
Params: `State`, `StateChange` (delta), `StateMin`, `StateMax`.
- Quick Attack 563 `ev6: 79{State=20 Stat_Speed,+10000}` / `ev7: −10000` "always goes first".
- Planted 302 `79{State=70,+1,StateMax=10}` counts planted rounds. Plant 303 reads it with op78.
- Tympanic Tantrum 387 `ev6: 79{30,+1}` "can affect enemies underground" (caster temporarily counts as underground).
- Zoomies 2493 `ev2: 79{25,+10,Max=10000}` +10% speed per hit.

### op80 — set weather · 23/23 · target: battle (global) · **high**
Params: `Points`, `Accuracy` (always 200), `Duration` (9/5/7), `ChainFailure`. Replaces the current weather aura.
- Scorched Earth 172 `24 ; 80@171{A200,D9}` "for [duration(1,2)] rounds"; [PS] buff `battle`.
- Sandstorm 453 `80@454{D5}`.
- Call of Winter 1013 `80@205{D9}` (Blizzard: state 52 Chilled on everyone).

### op29 — conditional bonus damage (state-gated) · 23/16 · target: enemy · **high**
Params: `Points`, `Accuracy`, `RequiredCasterState`, `RequiredTargetState`, `IsPeriodic`. Deals damage only if the caster/target state is non-zero.
- Conflagrate 179 `24 ; 158 ; 29{Points=10,RequiredTargetState=34}` "[StandardDamage(1,3)] additional … if Burning".
- Maul 345 `29{20,TargetState=77}` (Bleeding).
- Blast of Hatred 472 `29{RequiredCasterState=28}` "if you were struck first".
- Howling Blast 120 `138{91,+1,TargetTestState=52} ; 3×29{RequiredCasterState=91}` = enemy team only if the primary target is Chilled.

### op384 — extra damage if caster acts first · 23/23 · target: enemy · **high**
Params: `Points`, `Accuracy`. Modern equivalent of [EMU] op103 `HandleExtraAttackFirst` (`if firstAttackingTeam != caster → return`).
- Gnaw 412 `24 ; 158 ; 384{10}` "[StandardDamage(1,3)] additional damage if the user attacks first".
- Thrash 202 `24 ; 158 ; 24{A50} ; 384`.
- Tail Sweep 122 `385{25} ; 384{18}` "[StandardDamage(1,2)]… or [StandardDamage(1,1)] if the user strikes last".

### op141 — damage split across enemy team (+weather bonus) · 22/18 · target: enemy team · **high**
Params: `Points`, `Accuracy`, `BonusState`, `BonusPoints`. Total damage is divided evenly among living enemies. +BonusPoints if BonusState (weather) is active.
- Thunderbolt 779 `{36}` "split evenly across the enemy team"; [PS] `teamEnemy`.
- Arcane Storm 589 `{24,BonusState=54,BonusPoints=4}` "Does additional damage if Arcane Winds is already active"; [PS] +28 if arcaneStorm.

### op27 — ramping damage · 21/21 · target: enemy · **high**
Params: `Points`, `Accuracy`, `PointsIncreasePerUse`, `PointsMax` (in [EMU] this is the max number of *uses*), `StateToTriggerMaxPoints` (always 0 here). [EMU] `HandleRampingDamage` tracks states 26 `Ramping_DamageID` and 27 `Ramping_DamageUses`. Damage = Points + Inc·uses, uses capped. The counter resets if another ramping ability is used.
- Vicious Fang 156 `{15,Inc=5,Max=25}` "increases by [pointsincreaseperuse(1,1)*AttackBonus()] … Max [pointsmax(1,1)*…]". The tooltip treats PointsMax as a damage cap (25). [PS] `combo max 2 value 5` (2 increments). Use: damage = min(Points + Inc·n, PointsMax).

### op32 — heal caster for % of last damage dealt · 21/15 · target: caster · **high**
Params: `Points` (%), `Accuracy`, `ChainFailure`. [EMU] `HandleHealPctDealt`: heal = Points% of state 32 `Last_HitDealt`. Abilities clear state 32 at ev6 first.
- Death Coil 121 `{50}` "heals the user for [points(1,3)]%".
- Hungry Bite 2423 `{200}`.
- Necrotic Bomb 2301 `ev5: 3×(24 ; 32{50})`: interleaved per enemy.

### op135 — instakill (self/holder) · 20/18 · target: caster/aura holder · **medium**
Params: `Accuracy`, `CasterImmuneState`, `TargetImmuneState`, `enableReverse`, `BypassPetPassives` (always 0).
- Explode 282 `68{40} ; 135{CasterImmuneState=100}` "Instantly kills the caster".
- Apocalypse aura 682 `ev5: 156{141} ; 135{A200,CasterImmuneState=1,TargetImmuneState=93}`: kills when the meteor lands, except Cockroaches (state 93 `Special_IsCockroach`), and not if removed by a cleanse (state 141).
- Undead 242 `ev5: 135` ends the immortal round.
- Haunt 652 `230@653 ; 158 ; 135{enableReverse=1}`: the caster "dies" reversibly and is revived by op165 when the aura ends.

### op145 — swap to highest-health pet · 17/17 · target: enemy (forced) or self (not encoded) · **high** (side needs override)
Params: `Accuracy`.
- Sweep 457 `24 ; 158 ; 145` "forces the opponent's highest health pet to swap in" ([PS] swap previous, highestHealth, chainFailure).
- Portal 1125 `ev7: 145` "The user then swaps … highest health".
- Feign Death 568 `ev12: 145`.

### op62 — damage = Points% of max health · 17/12 · target: enemy (varies) · **high**
Params: `Points` (%), `Accuracy`, `IsPeriodic`.
- Glowing Toxin 271 `{6}` "[points(1,1)]% of maximum health as damage every round".
- BONESTORM! 1761 `3×24 ; 62{30}` "costs [points(1,4)]% of your total health": this row targets **self**.
- Toxic Skin 1086 `ev1: 62@379{5}` hits the attacker.

### op139 — swap to lowest-health pet · 15/15 · target: enemy (forced) or self · **high**
Params: `Accuracy`. Death Grip 780 `24 ; 158 ; 139` "forces the opponent's lowest health pet". Return to Rest 2405 `ev12: 49@2409 ; 139` (self). Terrifying Toxicity 2456 `ev7: 139` (enemy).

### op136 — dispel all auras · 14/6 · target: varies · **medium**
Params: `DontMiss` (always 1). Always wrapped in `31{141 IsCleansing,1} … 31{141}`, so removal procs can tell they were cleansed (Apocalypse `156{141}`). Copies per target: 2 for one pet (probably pet + its pad, low confidence), 4 for a team (High Fiber 941).
- Sear Magic 763 "Removes all buffs and debuffs from the user" (self).
- Disruption 1123 "…from the enemy" (enemy): same data, different target.

### op179 — thorns: damage attacker on direct damage taken · 12/12 · target: attacker · **high**
Params: `Points`, `Accuracy`, `IsPeriodic`, `OverrideIndex`. Only in `ev1` of auras.
- Thorns 317 `ev1: 179{5}` "[FormatDamage(SimpleDamage(1,1),nil,AURAENEMY)] damage each time you take direct damage".
- Spirit Spikes 915 `{20}` + state 71 −10.
- Spiked Shell 2323 `{12,IsPeriodic=1}` ([PS] isPeriodic).

### op104 — heal caster if state requirements met · 11/7 · target: caster · **high**
Params: `Points`, `Accuracy`, `RequiredCasterState`, `RequiredTargetState`, `Variance`. [EMU] `HandleHealState` (returns if the required state is missing, heals caster).
- Devour 538 `24 ; 104{30,RequiredTargetState=1}` "If the user kills an enemy with Devour, they restore [StandardHealing(1,2)]".
- Leech Life 383 `ev2: 157{91,1} ; 23{10} ; 104{10,TargetState=64} ; 31{91}` "Double if Webbed".

### op159 — reset/advance chain context · 10/6 · **low**
No params. Appears between repeated team aura applications (Stench 527 `50,159,50,159,50`; Unholy Ascension 321; Apocalypse 519 `50@683 ; 159 ; 50@682{ChainFailure=1}` = enemy pad + own pad) and after op194 (Haymaker/Blingtron `24 ; 194 ; 159 ; x`). Best guess: it clears the "previous effect" result (and possibly retargets), so the next effect isn't chain-failed by the preceding miss/gate.

### op131 — interrupt: apply aura only if caster acted first · 9/9 · target: enemy · **medium-high**
Params: `ChainFailure`, `Accuracy`, `Duration` (0 = current round). Kick 307 `24 ; 177 ; 131@927` "stuns the opponent if the user goes first" ([PS] `$isFasterThan`, buffDuration 0). Horn Attack 571 `24 ; 156{149} ; 131@927{ChainFailure=1,A50}` "if faster … 50% chance".

### op157 — gate: continue only if target.State == value · 8/8 · **medium-high**
Params: `State`, `StatePoints`. Belly Slide 529 `ev6: 157{58 Blizzard,1} ; 28@853` "goes first if the weather is a Blizzard". Contagion Strike 2353 `24 ; 157{21,1} ; 29{TargetState=21}`. Whirlwind 2447 `178 ; 157{22 Stunned,1} ; 49@self` "Will dissipate if successful". Haunt 653 `ev7: 157{1,1} ; 49@653` (drop when the host dies).

### op107 — swap to random pet · 8/8 · target: enemy or self · **high**
Params: `Accuracy`. Nether Gate 466 `24 ; 158 ; 107` "forces an opponent's pet to swap" ([PS] random). Fade 1084 `ev7: 49@1083 ; 107` (self).

### op150 — create team shield object (blocks N attacks) · 8/8 · target: caster's pad · **medium-high**
Params: `ChainFailure`, `Accuracy`. Beaver Dam 325 `150@326` "blocks two attacks"; Ice Barrier 479; Onyx Wall 439 ([PS] `teamAlly`). The block count lives in the aura logic (hard-code "2 attacks"; allied attacks count for dams/walls).

### op68 — damage = Points% of caster max health · 7/5 · target: enemy · **high**
Explode 282 `{40}` "damage equal to [points(1,1)]% of the user's total health"; Massive Explosion 2429 `3×68{50}` (enemy team); [PS] `40% of source`.

### op156 — gate: stop if target has State (non-zero) · 7/6 · **medium**
Params: `State`. Chew 540 `T2: 156{1 Is_Dead} ; 24`; Torment 2292 `…363 ; 156{1} ; 32…` ([PS] `stop if backupEnemy dead`); Apocalypse 682 `ev5: 156{141} ; 135` (a cleansed meteor doesn't kill).

### op76 — two-stage attack: 1st use applies aura on caster, 2nd consumes it and hits · 7/7 · **high**
Params: `Points`, `Accuracy`. [EMU] `HandleDamageToggleAura`. Pump 297 `76@296{45}` "First use: … Second use: … [StandardDamage(1,1)]"; Wind-Up 459; Cannon Shot 1860 ([PS] `buffBeforeAction needed`). op75 = same with the aura on the **caster's pad** (Launch Rocket 293, [PS] `teamAlly`). op77 = aura on the **target** (Lock-On 301 `77@300{45,D1}`, [PS] `teamEnemy`).

### op96 — damage; always hits if state present · 7/7 · target: enemy · **high**
Params: `Points`, `Accuracy`, `CasterState`, `TargetState`, `IsPeriodic`. [EMU] `HandleDamageHitState`: skip the hit roll when caster.State[CasterState] or target.State[TargetState] is set. Blood in the Water 423 `{40,A50,TargetState=77}` "Always hits if Bleeding"; Spirit Claws 974 `{28,A80,CasterState=55}` "Always hits … Moonlight"; Worst Fear 1555 `TargetState=35 turnLock` (asleep or stunned).

### op229 — apply type-override aura (replace existing) · 7/7 · target: follows previous effect (self or enemy) · **medium**
Params: `ChainFailure`, `Accuracy`, `Duration` (1, or −1), `Unused` (=1), `CasterState`, `TargetState` (=124 TypeOverride). Rot 476 `85{124} ; 24 ; 50@1418 ; 229@968(Undead Mark)` targets the enemy. Nature's Ward 574 `85{124} ; 26@820 ; 229@1417(Elemental Mark)` targets self. Nature's Ward aura 820 re-applies it every tick. Type-override auras carry state 124 = 1-based pet type.

### op363 — damage = Points% of target's current health · 6/2 · **high**
Torment 2292 `{20}` "[points(1,1)]% of their current health"; Mind Games: Health 2388 (own team).

### op170 — bonus damage if weather state active · 6/4 · target: enemy · **high**
Params: `Points`, `Accuracy`, `weatherState`, `Unused`, `IsPeriodic`. Solar Beam 753 `24 ; 158 ; 170{20,weatherState=61}` "plus [StandardDamage(1,3)] if Sunlight"; Mana Surge 489 (54 Arcane Winds); Storm Breath 2508 (62).

### op100 — heal only if pet type matches · 6/3 · target: ally (backline copies) · **high**
Params: `Points`, `Accuracy`, `RequiredCasterPetType`, `RequiredTargetPetType` (1-based), `Variance`. Healing Stream 922 `23{24} ; 2×100{12,TargetPetType=9}` "half that to other aquatic allies" ([PS] backupAlly1/2 isPetType aquatic). Rebuild 533 (10 = Mechanical).

### op226 — damage + bonus if target has state · 6/5 · **high**
Params: `Points`, `Accuracy`, `BonusPoints`, `BonusState`, `IsPeriodic`, `OverrideIndex`. Deep Burn 1407 `{13,Bonus=13,State=34}` "double damage if Burning"; Sneak Attack 1371 `{17,+8,82}`; Exposing Claw 2424.

### op246 — modify cooldown of ability slot · 6/2 · target: triggering pet · **high**
Params: `Index` (slot 1–3), `CooldownModification`. Bend Time 1904 aura `ev10: 246{1} ; 246{2} ; 246{3} ; 49@self` "The next pet you swap in will have all of its cooldowns cleared" (0 = reset). op245/247 are variants (not reachable).

### op53 — heal Percentage% of max health · 5/5 · target: caster (or holder) · **high**
Buried Treasure 578 `{25}` "healing up to [percentage(1,1)]%"; Wish aura 274 `T2: 53{50}` "Next round, your active pet will be healed for 50%"; Recovery 726 (Humanoid passive) `{4}`.

### op373 — replace ability in slot (temporary) · 5/5 · **high**
Params: `Slot`, `Ability`. Mind Games 2393 `ev0: 373{Slot=2,Ability=1411}` "temporarily replaced an ability with Grave Destruction". op249 is the same with a test aura (2285). Slot 0 = ? (2392 `{Ability=2388}` has no Slot; probably slot 0/1).

### op112 — resurrect other dead allies · 4/2 · **high**
Params: `Percentage`, `Unused(100)`, `RequiredCasterState`, `RequiredTargetState(=1 dead)`. Raise Dead 2347 `2×112{5}` "back to life, returning them to 5% health" ([PS] res backupAlly1/2).

### op44 — heal Points% of last hit taken · 4/4 · target: caster · **high**
Healing Flame 168 `23{20} ; 44{50}` "plus half of the last hit taken" (state 31 Last_HitTaken). [PS] `lastDamageTaken × 0.5`.

### op63 — apply positive periodic aura (caster/own pad) · 4/4 · **high**
Params: `Unused`, `Accuracy`, `Duration`. [EMU] `HandlePeriodicPositiveTrigger`, CASTER. Tranquility 254 `63@255{D2}` "Persists through pet swaps" ([PS] `teamAlly`); Photosynthesis 268 `63@267{D5}` ([PS] source); Wish 273 `63@274{D1}`.

### op169 — damage with state requirements (weather/proc) · 4/3 · **medium**
Params: `Points`, `Accuracy`, `RequiredCasterState`, `RequiredTargetState`, `IsPeriodic`. Scorched Earth weather 171 `2×169{4,RequiredTargetState=47,IsPeriodic=1}` "front pet of both teams each round" ([PS] `allActive` = both heads). The 47 (`Passive_Elemental`) looks like an exclusion (elementals ignore weather) despite the "Required" label, so verify in-game. Lightning Storm 203 `ev2: 169{2}` "bonus damage each time they attack".

### op172 — apply aura with chance, guaranteed if state · 4/4 · target: enemy · **high**
Params: `StatePoints`, `CasterState`, `Duration`, `TargetState`, `ChainFailure`, `Accuracy`. Deep Freeze 481 `172@927{A25,TargetState=52}` "25% … becomes 100% if Chilled"; Surge of Light 769 `CasterState=56,TargetState=56` "100% if Darkness"; Fury of 1,000 Fists 226 `{Accuracy=0,TargetState=82}` "stun if Blinded".

### op385 — extra damage if caster acts last · 4/4 · **high**
Tail Sweep 122, Arcane Slash 1353 `384{25} ; 385{18}` (one of the two applies), Steady Snap 2402.

### op59 — bonus damage if caster HP < target HP · 3/3 · **high**
Comeback 253 `24{15} ; 59{10}` "[StandardDamage(1,2)] additional damage if the user has lower health" ([PS] `$hasLessHealth`).

### op66 — damage, +Boost% if target < 25% HP · 3/3 · **high**
Predatory Strike 518 `{30,Boost=100}` "double damage if the target is below 25% health". The 25% threshold is hard-coded.

### op248 — apply positive aura if caster has state · 3/2 · target: caster · **high**
Params: `Unused`, `Accuracy`, `Duration`, `TickDownFirstRound`, `RequiredCasterState`. Evolving Bite 2207 `… ; 248@485{D5,RequiredCasterState=302 Dealt_KillingBlow}` (buff on kill). The Biggest Hammer 2246 `248@2252{D100,State=304}`.

### op138 — modify state if target tests true · 2/2 · target: caster · **medium-high**
Params: `State`, `StateChange`, `StateMin`, `StateMax`, `TargetTestState`, `TargetTestStateValue`. Howling Blast 120 `138{91,+1,TargetTestState=52}` sets EffectSucceeded if the target is Chilled. Dark Rebirth 795 `138{146,+1}`.

### op111 — resurrect self at Percentage% · 2/2 (+723/724) · **high**
Params: `Percentage`, `Unused`. Dark Rebirth 795 `ev7: 138 ; 157{1,1} ; 111{100} ; 26@796{D−1}` "If killed, will be reborn with full health". Failsafe 723 `49@822 ; 111{20} ; 26@244{D−1}` (Mechanical passive, 20%).

### op223 — apply aura to allies of a pet type · 2/1 · **high**
Params: `Unused`, `Accuracy`, `Duration`, `RequiredPetType` (1-based). Refuge 1344 `26@759{D6} ; 2×223@759{RequiredPetType=5}` "the user and each friendly critter".

### op128 — set health to Percentage% · 2/2 (DNT) · **high**
[DNT] TRAP HELP 2468 `{1}` "Sets target to 1% of max health"; INSTADEAD 2502 `{}` → 0%.

### op360 — clear/consume state · 2/2 · **low**
`{State=304 Special_Auto_Rebuilder}` in The Biggest Hammer 2246 and Auto-Rebuilder 2252. Probably resets the state to 0, or the opposite of op157.

### op230 — apply aura (Haunt variant) with caster/target state checks · 2/2 · **medium**
Params: `ChainFailure`, `Accuracy`, `Duration`, `MaxAllowed`, `CasterState(=68 unkillable)`, `TargetState(=1 dead)`. Haunt 652 `230@653{D4,68,1}` "enters the target". The state params most likely mean "fail if the caster is unkillable or the target is dead" (they act as exclusions).

### op197 — damage = Power% of last hit taken · 2/2 · **high**
Vengeance 997 `{Power=100}` "damage equal to the last hit taken by the user"; Dark Fate 2432 `{150}`.

### op149 — non-lethal damage · 2/2 · **high**
[EMU] `HandleDamageNonLethal` (caps damage at HP−1). Weakening Blow 826 "cannot reduce the opponent's health below 1".

### op194 — gate: continue only if previous missed · 2/2 · **medium-high**
Haymaker 762 `24{A50} ; 194 ; 159 ; 52@174` "If the ability misses, the user is stunned"; Blingtron Gift Package 989 "If the ability misses, the user is healed".

### op254 — heal, +Modifier% if caster HP below Threshold · 2/2 · **high**
Params: `Points`, `Accuracy`, `Threshold`, `Modifier`, `Variance`. Eat the Fly 2206 `{10,Threshold=33,Modifier=300}` "increased by [modifier(1,1)]% if health is below [threshold(1,1)]%". Aura of Gonk 2069 (ev2).

### op117 — lock ability slot · 2/1 · **high**
Params: `Index`, `LockDuration`. Mind Games: Ability Lockout 2390 `117@174{Index=1,LockDuration=9} ; 117{Index=2}` "Lock out ability slots 1 and 2 for [lockduration(1,1)] turns".

### Single-use opcodes (1 row each)
| op | name | params → meaning | example | conf. |
|---|---|---|---|---|
| 358 | ramping damage + variance | op27 + Variance | Evolving Bite 2207 `{5,Inc=10,Max=500,Var=50}` | high |
| 77 | two-stage, aura on target | Points, Accuracy, Duration | Lock-On 301 | medium |
| 204 | reflect % of direct damage | Points (%) | Reflective Shield 1104 `ev1: 204{50}` "50% of direct attacks back" | high |
| 137 | apply aura if caster/target has state | CasterState, CasterStateValue, Duration, TargetState… | Spiny Carapace 1073 `137@1075{CasterState=158,D1} ; 26@1074` "tripled … if used while already active" | medium-high |
| 0 | test | p1 | Stun Aura Test 86 | n/a |
| 45 | damage + Boost% if caster slower | Points, Accuracy, Boost | Slow Digestion 2024 `{20,Boost=150}` | high |
| 97 | damage + Boost% if caster faster | same | Rapid Consumption 2026 | high |
| 86 | damage; second hit on the same target applies AuraID | Points, Accuracy, Duration, AuraID, Duration2 | Gravity 323 `86@322{25,D2,AuraID=324,D2=2}` "If used again on the same target, levitated … unable to flee" | medium |
| 75 | two-stage, aura on caster pad | Points, Accuracy | Launch Rocket 293 | medium |
| 249 | replace ability in slot | Slot, Ability | Test Aura 2285 | high |
| 370 | damage + BonusDamage per same-species ally | Points, Accuracy, BonusDamage | Dragon's Call 2283 `{25,+7}` | high |
| 140 | refresh existing aura instances to NewDuration | NewDuration | Blistering Cold 785 `140@787{3}` "Existing stacks of Frostbite will be refreshed" | high |
| 256 | aura/weather apply (unknown variant) | Points, Accuracy, Duration, ChainFailure | DATA ERROR 615 `{A100,D9}` (no aura id) | low |
| 67 | equalize health | Accuracy | Life Exchange 277 | high |
| 165 | restore/revive the aura's caster (undo reversible death) | Accuracy, CasterImmuneState, TargetImmuneState | Haunt aura 653 `ev5: 165` ([PS] `res source`) | medium |
| 78 | plant toggle: heal Points × State counter | Points, State, MaxPoints, StateToMultiplyAgainst | Plant 303 `78@302{12,State=70,Max=10,Mult=70}` "For every round spent planted, you gain [StandardHealing(1,1)]" | medium-high |
| 129 | lock the ability just used by the holder | LockDuration | Nevermore aura 738 `ev9: 129{5} ; 49@self` | high |
| 65 | periodic damage + bonus in weather | Points, Accuracy, BonusPoints, BonusState, IsPeriodic, OverrideIndex | Creeping Fungus 742 `{7,+3,55 Moonlight,Periodic}` | high |
| 171 | heal with state requirement | Points, Accuracy, ReqCaster/TargetState, Variance | Photosynthesis aura 267 `23{6} ; 171{6,61,61}` "Twice as effective in Sunny weather" | high |
| 168 | apply negative aura (Nevermore) | as op50 | Nevermore 522 `168@738{D2}` | medium |
| 144 | select an unused allied corpse | none | Consume Corpse 665 `144 ; 61{50,TargetState=120} ; 31{120}` | low |
| 61 | heal Points% of max HP with state requirements | Points, Accuracy, ReqCaster/TargetState, Variance | same, "restore [points(1,2)]% of total health" | medium |

---

## 2. Event enum (TurnType=1, `EventTypeEnum`)

The names come from Blizzard's tooltip env ([TT]): `PROC_ON_APPLY, PROC_ON_DAMAGE_TAKEN, PROC_ON_DAMAGE_DEALT, PROC_ON_HEAL_TAKEN, PROC_ON_HEAL_DEALT, PROC_ON_AURA_REMOVED, PROC_ON_ROUND_START, PROC_ON_ROUND_END, PROC_ON_TURN, PROC_ON_ABILITY, PROC_ON_SWAP_IN, PROC_ON_SWAP_OUT`. The Lua does not show numeric values. The numbers below are inferred from list order, which matches every data point. [EMU] confirms ROUND_START/ROUND_END with Deflection 490 (ev6/ev7). Tooltips use `getProcIndex(PROC_ON_AURA_REMOVED)` on ev5 turns, `PROC_ON_DAMAGE_DEALT` on ev2 and `PROC_ON_DAMAGE_TAKEN` on ev1.

A trigger turn on an **ability** is live only while that ability is being used this round. A trigger turn on an **aura** is live while the aura exists. Type-0 turns use event 255 (none).

| ev | name | conf. | evidence (reachable count) |
|---|---|---|---|
| 0 | ON_APPLY (aura applied) | high | Sunny Day 403 `6×116` (max-HP refresh on apply); Mind Games 2392–2395 `373` swap abilities on apply; Elementium Bolt 2372 `24` ([PS] `applied`). 9 uses |
| 1 | ON_DAMAGE_TAKEN | high | Thorns 317 `179`; Asleep 498/926 `49@self` "Breaks on damage"; Counterstrike 158 `31{28,100}`; Howl 1725 / Vulnerable 1726 `49@self` (one-hit debuffs). [PS] `damageReceive`/`directDamageReceive`. 30 |
| 2 | ON_DAMAGE_DEALT | high | Supercharged 207 / Dominance 1028 `49@self` (next attack); Brittle Webbing 381 `24` "each time they attack"; Leech Life 383 heal; Lightning Storm 203 `169`. [PS] `damageDeal`. 18 |
| 3 | ON_HEAL_TAKEN | high (name) | not used |
| 4 | ON_HEAL_DEALT | high (name) | not used |
| 5 | ON_AURA_REMOVED (expired, dispelled or consumed) | high | Bomb 819 / Sticky Grenade 637 `24` "After 3 rounds…"; Stunned 927 / Asleep / Polymorphed 822 `26@924 Resilient{D2}` (post-CC immunity); Undead 242 `135`; Snap Trap 671 (fires when ev9 removes it). [PS] `removed`/`expired`. 27 |
| 6 | ON_ROUND_START (before any ability resolves) | high | Quick Attack 563 `79 +10000 ; 116`; Deflection 490 `28@853`; Counterstrike `31{28}=0` reset; Consume 160 `31{32}` reset. [PS] `startTurn`. 45 |
| 7 | ON_ROUND_END (after both abilities, before aura ticks) | high | Quick Attack `−10000`; Portal 1125 `145 ; 49@1124`; Fade 1084 `107`; Terrifying Toxicity 2456 `139` "At the end of the round…"; Dark Rebirth 795 revive. [PS] `endTurn`. 43 |
| 8 | ON_TURN | medium | only PadAuraTest 84 (unreachable) |
| 9 | ON_ABILITY (holder uses an ability) | high | Snap Trap/Magma Trap 671/810 `49@self{25}`; Nevermore 738 `129` "This pet's next ability will be locked out"; Zoomies 2493 `26@831`. [PS] `onAbility`. 5 |
| 10 | ON_SWAP_IN | high | Minefield 635 / Booby-Trapped Presents 1081 `24 ; 49@self` "next pet … to enter"; Mudslide 718 `52@248` (root on entry); Sanctified Ground 1033; Bend Time 1904. [PS] `onSwapIn`. 6 |
| 11 | ON_SWAP_OUT | high | Frostbite 787 `49@self` "Swapping removes this aura". [PS] `onSwapOut`. 1 |
| 12 | post-aura-tick (not in Blizzard's list) | medium | Feign Death 568 / Warning! 1583 / Return to Rest 2405 `ev12: 49@dodge ; 145/139`. The dodge aura is kept through the aura phase, then the pet swaps. [PS] `postAuraTick`. 3 |

---

## 3. Ability `Flags` (BattlePetAbility.Flags), reachable set

| value | bits | count (auras) | examples | probable meaning | conf. |
|---|---|---|---|---|---|
| 0 | — | 766 (31) | Bite 110, most attacks | normal | — |
| 1 | 0x1 | 26 (1) | Poisoned 379, Bleeding 491, Corrosion 446, Frostbite 787, Volcano 175, Geyser 417, Brittle Webbing 381 | harmful/debuff-type aura display. Many are DoT/object auras filed under the attack's family (fam ≠ −1) | medium-low |
| 2 | 0x2 | 335 (100) | Crouch 165, Healing Flame 168, Healing Wave 123, buff auras (Adrenaline 161, Crouch 164), weathers 596/205/229 | **no strong/weak type hints**: tooltip `noStrongWeakHints` from `C_PetBattles.GetAbilityInfoByID` ([TT] `abilityHasHints`). Set on heals, buffs and utility | high |
| 3 | 0x1+0x2 | 63 (43) | Stunned 174/927, Rooted 248, Asleep 498, Attack Reduction 153, Polymorphed 822, Glowing Toxin 271 | debuff aura without hints | medium |
| 6 | 0x2+0x4 | 9 | family passives 236–239, 241–245, Failsafe 723, Damned 724, Recovery 726 | 0x4 = **passive (family racial)** | high |
| 8 | 0x8 | 3 | SpeedPriority "Server Only" 853, Elementium Bolt 2372, Void Crash 2373 | hidden / server-only helper aura | medium |
| 10 | 0x8+0x2 | 7 | Aquatic 240 (passive with hidden flag), Critical Bonus 1128/2445, Auto-Rebuilder 2252, Mind Games 2394/2395, Too Big! 2543 | hidden, no hints | medium |
| 18 | 0x10+0x2 | 17 | Aged Yolk 667, Sear Magic 763, Eggnog 835, Disruption 1123, Wild Rage 970/971, Enrage 1391/1392, Annoying Shield 2225, Coolant Leak 2249 | 0x10 unknown. Candidates: "undispellable" or "not affected by cleanse" (low) | low |
| 32/34/50 | 0x20 | 1/1/1 | Sting 2099, Righteous Inspiration 1103, Aura of Gonk 2070 | unknown | low |

---

## 4. Family passives 236–245 and what must be hard-coded

The DB2 passive rows are mostly descriptive. They carry `states{}` (applied while the passive "aura" is active) but no trigger logic. [EMU] hard-codes Beast and Magic inside `CalculateDamage`.

| id | family | DB2 content | hard-code in engine |
|---|---|---|---|
| 236 Critter | 4 | no states, no turns | immunity to stun/root/sleep, i.e. block auras carrying `Mechanic_IsStunned` 22 / `turnLock` 35 sleep / `swapOutLock` 36 root. (The DB2 Resilient 924 aura has state 149 but Critter doesn't use it.) |
| 237 Beast | 7 | `states {23 Mod_DamageDealtPercent: 25}` | apply +25% only while HP < 50% ([EMU] adds 25 to the type modifier when `hp*100/max < 50`) |
| 238 Humanoid | 0 | tooltip → aura **726 Recovery** `T1 op53{Percentage=4}` | at round end, if the pet dealt damage this round (state 127 `Condition_DidDamageThisRound`), run 726 on it |
| 239 Flying | 2 | `states {25 Mod_SpeedPercent: 50}`, empty T1 | apply only while HP > 50% |
| 240 Aquatic | 8 | flags 10, no states | DoT damage (IsPeriodic) **taken** −50% and DoT **dealt** +25% (per the current tooltip) |
| 241 Elemental | 6 | none | ignore negative weather effects. [TT] `SumStates` skips weather states when `unitPetType == 7`. Which weather effects count as "negative" (Scorched Earth tick, Blizzard chill, Sandstorm accuracy, Mud root …) must be listed by hand |
| 242 Undead | 3 | aura: `states {68 unkillable:1, 23:−25}`, `ev5: op135`, `T1: op49@822` | trigger on lethal damage: run **724 Damned** (`49@822 ; 111{1%} ; 26@242{Duration=1}`). The pet is immortal for 1 round at −25% damage, then ev5 kills it. Undead pets revived this way can't be revived again. The trigger itself is not in DB2 |
| 243 Magic | 5 | none | cap any single hit at 35% of max HP ([EMU] `damage = min(damage, 35% maxHP)`) |
| 244 Mechanical | 9 | marker aura only | trigger on death once per battle: run **723 Failsafe** (`49@822 ; 111{20%} ; 26@244{Duration=−1}`). 244 then acts as the "already used" marker. Explode-type self-kills skip it (tooltip "prevents Failsafe"; `BypassPetPassives` is 0 there, so hard-code it) |
| 245 Dragonkin | 1 | `states {23: 50}` | after the dragon's hit brings a target below 50% HP, apply +50% damage for the next round (a 1-round aura) |

Also hard-code: the type-effectiveness table (strong 1.5 / weak 0.66: [EMU] `gtBattlePetTypeDamageMod`, and the tooltips list it), the level-gap accuracy penalty (−2% per level, [EMU]), crit = ×2 (+100%, state 40 chance), resilient-after-stun (auto via ev5 → 924), and Humanoid/Beast/Flying HP thresholds.

---

## 5. Most important uncertainties

1. **Targets**: team replication, pet vs pad, and self vs enemy for swaps, dispels and control are not in DB2 (§0.1). This is the single biggest source of engine errors; drive it with an override table seeded from petsim.
2. **MaxAllowed 0** semantics: [EMU] reads it as "unlimited stacking", retail behaves like refresh. Use 0 ⇒ 1.
3. op159, op360, op144, op256, OverrideIndex, flags 0x1/0x10/0x20: low-confidence guesses.
4. op169 `RequiredTargetState=47` on Scorched Earth reads as an exclusion, not a requirement.
5. Event 12's placement (after aura ticks, before cooldown tick) comes from petsim only.
