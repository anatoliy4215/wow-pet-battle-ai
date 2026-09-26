-- BPLogger: records every pet battle into SavedVariables (BPLoggerDB) for offline simulator validation.
-- It only reads the C_PetBattles API and hooks the player's own button presses. It never acts.
--
-- Per battle:  meta, initial full state of all pets, then per round: player's action, combat log lines,
-- and a full snapshot (health, power, speed, cooldowns, auras with turns remaining, weather, states).

BPLoggerDB = BPLoggerDB or { battles = {}, version = 1 }

local ALLY, ENEMY, WEATHER = 1, 2, 0 -- Enum.BattlePetOwner
local PAD = 0
local cur -- battle being recorded
local round = 0

local KEY_STATES = { 18, 19, 20, 22, 23, 24, 25, 29, 30, 33, 35, 36, 40, 41, 68, 73, 74 }

local function safe(f, ...)
  local ok, a, b, c, d, e, f2, g = pcall(f, ...)
  if ok then return a, b, c, d, e, f2, g end
end

local function auras(owner, idx)
  local out = {}
  local n = safe(C_PetBattles.GetNumAuras, owner, idx) or 0
  for i = 1, n do
    local auraID, instanceID, turnsRemaining, isBuff, casterOwner, casterIndex = safe(C_PetBattles.GetAuraInfo, owner, idx, i)
    if auraID then
      out[#out + 1] = { id = auraID, inst = instanceID, turns = turnsRemaining, buff = isBuff,
                        co = casterOwner, ci = casterIndex }
    end
  end
  return out
end

local function petInfo(owner, i, full)
  local p = {
    hp = safe(C_PetBattles.GetHealth, owner, i),
    maxhp = safe(C_PetBattles.GetMaxHealth, owner, i),
    power = safe(C_PetBattles.GetPower, owner, i),
    speed = safe(C_PetBattles.GetSpeed, owner, i),
    auras = auras(owner, i),
    cd = {},
    st = {},
  }
  for s = 1, 3 do
    local usable, cooldown, lockdown = safe(C_PetBattles.GetAbilityState, owner, i, s)
    p.cd[s] = { usable and 1 or 0, cooldown or 0, lockdown or 0 }
  end
  for _, sid in ipairs(KEY_STATES) do
    local v = safe(C_PetBattles.GetStateValue, owner, i, sid)
    if v and v ~= 0 then p.st[sid] = v end
  end
  if full then
    p.species = safe(C_PetBattles.GetPetSpeciesID, owner, i)
    p.level = safe(C_PetBattles.GetLevel, owner, i)
    p.quality = safe(C_PetBattles.GetBreedQuality, owner, i)
    p.family = safe(C_PetBattles.GetPetType, owner, i)
    p.name = safe(C_PetBattles.GetName, owner, i)
    p.abilities = {}
    for s = 1, 3 do
      local id = safe(C_PetBattles.GetAbilityInfo, owner, i, s)
      p.abilities[s] = id or 0
    end
  end
  return p
end

local function snapshot(full)
  local s = { t = GetTime(), active = {}, pets = { {}, {} } }
  for owner = ALLY, ENEMY do
    s.active[owner] = safe(C_PetBattles.GetActivePet, owner)
    local n = safe(C_PetBattles.GetNumPets, owner) or 0
    for i = 1, n do s.pets[owner][i] = petInfo(owner, i, full) end
    s["pad" .. owner] = auras(owner, PAD)
  end
  s.weather = auras(WEATHER, PAD)
  return s
end

local function newRound(n)
  cur.rounds[#cur.rounds + 1] = { n = n, log = {}, action = nil }
end

local function curRound()
  if not cur then return nil end
  if #cur.rounds == 0 then newRound(0) end
  return cur.rounds[#cur.rounds]
end

local f = CreateFrame("Frame")
f:RegisterEvent("PET_BATTLE_OPENING_START")
f:RegisterEvent("PET_BATTLE_OPENING_DONE")
f:RegisterEvent("PET_BATTLE_PET_ROUND_RESULTS")
f:RegisterEvent("PET_BATTLE_PET_ROUND_PLAYBACK_COMPLETE")
f:RegisterEvent("PET_BATTLE_FINAL_ROUND")
f:RegisterEvent("PET_BATTLE_CLOSE")
f:RegisterEvent("CHAT_MSG_PET_BATTLE_COMBAT_LOG")

f:SetScript("OnEvent", function(_, event, ...)
  if event == "PET_BATTLE_OPENING_START" then
    local npc
    local guid = UnitGUID("target") or UnitGUID("npc")
    if guid then npc = select(6, strsplit("-", guid)) end
    cur = {
      date = date("%Y-%m-%d %H:%M:%S"),
      build = select(2, GetBuildInfo()),
      wild = safe(C_PetBattles.IsWildBattle),
      pvp = not safe(C_PetBattles.IsPlayerNPC, ENEMY),
      npc = npc,
      target = UnitName("target"),
      zone = GetZoneText(),
      rounds = {},
    }
    round = 0
    newRound(0)
  elseif not cur then
    return
  elseif event == "PET_BATTLE_OPENING_DONE" then
    cur.start = snapshot(true)
  elseif event == "PET_BATTLE_PET_ROUND_RESULTS" then
    local r = curRound()
    if r then r.results = ... end
  elseif event == "PET_BATTLE_PET_ROUND_PLAYBACK_COMPLETE" then
    local r = curRound()
    if r then r.after = snapshot(false) end
    round = (...) or (round + 1)
    newRound(round)
  elseif event == "CHAT_MSG_PET_BATTLE_COMBAT_LOG" then
    local r = curRound()
    if r then r.log[#r.log + 1] = ... end
  elseif event == "PET_BATTLE_FINAL_ROUND" then
    cur.winner = ...
    local r = curRound()
    if r then r.after = snapshot(false) end
    -- in PvP the enemy's abilities may only be known once used: read them again at the end
    cur.final = snapshot(true)
  elseif event == "PET_BATTLE_CLOSE" then
    if cur and #cur.rounds > 0 then
      table.insert(BPLoggerDB.battles, cur)
      print(string.format("|cff33ff99BPLogger|r: saved battle #%d (%d rounds)", #BPLoggerDB.battles, #cur.rounds))
    end
    cur = nil
  end
end)

-- record the player's own choices
local function act(kind, arg)
  local r = curRound()
  if r and not r.action then r.action = { kind, arg } end
end
hooksecurefunc(C_PetBattles, "UseAbility", function(slot) act("ability", slot) end)
hooksecurefunc(C_PetBattles, "ChangePet", function(idx) act("swap", idx) end)
hooksecurefunc(C_PetBattles, "SkipTurn", function() act("pass") end)
hooksecurefunc(C_PetBattles, "ForfeitGame", function() act("forfeit") end)

-- export the pet journal (owned pets with stats) so teams can be built from the real collection
local function exportCollection()
  if not C_PetJournal or not C_PetJournal.GetNumPets then return end
  local pets = {}
  local _, owned = C_PetJournal.GetNumPets()
  for i = 1, (C_PetJournal.GetNumPets()) do
    local petID, speciesID, isOwned, customName, level, favorite, isRevoked, speciesName, icon, petType =
      C_PetJournal.GetPetInfoByIndex(i)
    if petID and isOwned then
      local health, maxHealth, power, speed, rarity = C_PetJournal.GetPetStats(petID)
      local canBattle = select(15, C_PetJournal.GetPetInfoByPetID(petID))
      pets[#pets + 1] = { species = speciesID, level = level, quality = rarity, maxhp = maxHealth,
                          power = power, speed = speed, family = petType, name = speciesName,
                          battle = canBattle and 1 or 0 }
    end
  end
  if #pets > 0 then
    BPLoggerDB.collection = { date = date("%Y-%m-%d %H:%M:%S"), pets = pets }
  end
  return #pets, owned
end

local cf = CreateFrame("Frame")
cf:RegisterEvent("PET_JOURNAL_LIST_UPDATE")
cf:SetScript("OnEvent", function()
  cf:UnregisterEvent("PET_JOURNAL_LIST_UPDATE")
  C_Timer.After(5, exportCollection)
end)

SLASH_BPLOGGER1 = "/bplog"
SlashCmdList.BPLOGGER = function(msg)
  if msg == "clear" then
    BPLoggerDB.battles = {}
    print("|cff33ff99BPLogger|r: cleared")
  elseif msg == "pets" then
    local n = exportCollection()
    print(string.format("|cff33ff99BPLogger|r: exported %d pets. /reload to save to disk.", n or 0))
  else
    print(string.format("|cff33ff99BPLogger|r: %d battles stored. /reload to flush to disk, /bplog clear to reset.", #BPLoggerDB.battles))
  end
end
