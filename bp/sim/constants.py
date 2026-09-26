"""Game constants. Everything not read from DB2 lives here so it is easy to correct against logs."""

# Families, 0-based as in DB2 PetTypeEnum
HUMANOID, DRAGONKIN, FLYING, UNDEAD, CRITTER, MAGIC, ELEMENTAL, BEAST, AQUATIC, MECHANICAL = range(10)
FAMILY_NAMES = ["Humanoid", "Dragonkin", "Flying", "Undead", "Critter", "Magic", "Elemental", "Beast", "Aquatic", "Mechanical"]

# attacker family -> defender family it is strong / weak against
STRONG = {0: 1, 1: 5, 2: 8, 3: 0, 4: 3, 5: 2, 6: 9, 7: 4, 8: 6, 9: 7}
WEAK = {0: 7, 1: 3, 2: 1, 3: 8, 4: 0, 5: 9, 6: 4, 7: 2, 8: 5, 9: 6}
STRONG_MULT = 1.5
WEAK_MULT = 0.66  # log-verified: Chaos Beam 300.6 -> 198

# Family passive ability ids (DB2), index = family
PASSIVE_ABILITY = [238, 245, 239, 242, 236, 243, 241, 237, 240, 244]
HUMANOID_RECOVERY_AURA = 726
UNDEAD_AURA = 242
DRAGONKIN_AURA = 245  # applied for the round after bringing a target below 50%
STUNNED_AURA = 927
DAMNED_ABILITY = 724   # Undead passive trigger: 1% hp + immortal aura 242 for one round
FAILSAFE_ABILITY = 723  # Mechanical passive trigger: 20% hp + marker aura 244
RESILIENT_AURA = 924
BOSS_AURA = 1089   # species with Passive_Boss: -50% damage taken, max 35% of health per hit (log-verified)
ELITE_AURA = 956   # species with Passive_Elite: -50% damage taken

# State ids (BattlePetState)
S_IS_DEAD = 1
S_MAX_HEALTH_BONUS = 2
S_POWER = 18
S_STAMINA = 19
S_SPEED = 20
S_POISONED = 21
S_STUNNED = 22
S_DMG_DEALT_PCT = 23
S_DMG_TAKEN_PCT = 24
S_SPEED_PCT = 25
S_RAMPING_ID = 26
S_RAMPING_USES = 27
S_WAS_DAMAGED = 28
S_UNTARGETTABLE = 29
S_UNDERGROUND = 30
S_LAST_HIT_TAKEN = 31
S_LAST_HIT_DEALT = 32
S_FLYING = 33
S_BURNING = 34
S_TURN_LOCK = 35
S_SWAP_OUT_LOCK = 36
S_CRIT = 40
S_ACCURACY = 41
S_CHILLED = 52
S_WEBBED = 64
S_HEAL_DEALT_PCT = 65
S_HEAL_TAKEN_PCT = 66
S_INVISIBLE = 67
S_UNKILLABLE = 68
S_OBJECT = 69
S_FLAT_DMG_TAKEN = 71
S_FLAT_DMG_DEALT = 72
S_DODGE = 73
S_BLOCK_COUNT = 74
S_BLEEDING = 77
S_BLIND = 82
S_PETTYPE_DMG_DEALT = 87
S_PETTYPE_DMG_TAKEN = 88
S_PETTYPE_ID = 89
S_EFFECT_SUCCEEDED = 91
S_SWAP_IN_LOCK = 98
S_MAX_HEALTH_PCT = 99
S_DID_DAMAGE = 127
S_RESILIENT = 149
S_BLOCKED_FRIENDLY = 123  # walls/dams also block the owner's own attacks (log: Onyx Wall ate Deep Breath)
S_TYPE_OVERRIDE = 124
S_CONSUMED_CORPSE = 120  # Special_ConsumedCorpse: this dead pet was eaten by Consume Corpse
S_BOMB = 136
S_PLANT = 70
S_HEALTH_BEFORE_INSTAKILL = 145
S_ELITE = 153
S_BOSS = 162
S_IGNORE_DMG_BELOW = 191
S_IGNORE_DMG_ABOVE = 200
S_CHEAT_DEATH = 201
S_ADD_PERIODIC_DMG_TAKEN = 202
S_CRIT_DMG_BONUS = 314

# Passive_<family> states, index = family (Critter 42 ... Aquatic 51)
PASSIVE_STATES = [44, 46, 45, 50, 42, 49, 47, 43, 51, 48]

WEATHER_STATES = {53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 316}

# Events (BattlePetAbilityTurn.EventTypeEnum for TurnTypeEnum=1). Meanings inferred, see docs/opcodes.md
EV_APPLY = 0
EV_DAMAGE_TAKEN = 1
EV_DAMAGE_DEALT = 2
EV_HEAL_TAKEN = 3
EV_HEAL_DEALT = 4
EV_REMOVE = 5
EV_ROUND_START = 6
EV_ROUND_END = 7
EV_ABILITY = 9
EV_SWAP_IN = 10
EV_SWAP_OUT = 11
EV_POST_TICK = 12

# Tunables that need confirmation from logs
BASE_CRIT = 5          # %
CRIT_MULT = 1.5        # confirmed on a log: 93 -> 139
BASE_HIT = 100
LEVEL_HIT_PENALTY = 2  # % hit lost per level below the target (wild pets)
MAGIC_CAP = 0.35
MAX_DMG_REDUCTION = 0.75  # percent damage-taken reduction cap (Wowhead tooltips)
MECH_REVIVE_PCT = 0.20
BEAST_BONUS = 25       # handled via aura states when below 50%
FLYING_SPEED_PCT = 50
AQUATIC_DOT_TAKEN = 0.5
AQUATIC_DOT_DEALT = 1.25
HUMANOID_HEAL_PCT = 4

MAX_ROUNDS = 150

# Actions
A_ABILITY = (0, 1, 2)
A_SWAP = (3, 4, 5)
A_PASS = 6
N_ACTIONS = 7
