-- BPAdvisor: shows the current pet battle state as a QR code in a corner of the screen, for the phone
-- advisor page to read. It only reads C_PetBattles and draws a picture; it never acts in the game.
--
--   /bpa          show / hide the code         /bpa size 4   module size in pixels (2-8)
--   /bpa reset    move it back to the corner   /bpa test     check the QR encoder
local ALLY, ENEMY, PAD = 1, 2, 0

BPAdvisorDB = BPAdvisorDB or {}

local frame = CreateFrame("Frame", "BPAdvisorFrame", UIParent)
frame:SetFrameStrata("TOOLTIP")
frame:SetMovable(true)
frame:EnableMouse(true)
frame:RegisterForDrag("LeftButton")
frame:SetScript("OnDragStart", frame.StartMoving)
frame:SetScript("OnDragStop", function(self)
  self:StopMovingOrSizing()
  local p, _, rp, x, y = self:GetPoint()
  BPAdvisorDB.pos = {p, rp, x, y}
end)
frame:Hide()
local bg = frame:CreateTexture(nil, "BACKGROUND")
bg:SetAllPoints()
bg:SetColorTexture(1, 1, 1, 1)
local cells = {}

local function place()
  frame:ClearAllPoints()
  local pos = BPAdvisorDB.pos
  if pos then frame:SetPoint(pos[1], UIParent, pos[2], pos[3], pos[4])
  else frame:SetPoint("BOTTOMLEFT", UIParent, "BOTTOMLEFT", 20, 20) end
end
place()

local function safe(f, ...)
  local ok, a, b, c, d, e, g = pcall(f, ...)
  if ok then return a, b, c, d, e, g end
end

-- enemy abilities are hidden in PvP until used: remember what each enemy pet was seen using
local seen = {}  -- [petIndex] = {[slot] = abilityID}
local round = 0

local function enemySlotOf(abilityID, power, speed)
  for i = 1, (safe(C_PetBattles.GetNumPets, ENEMY) or 0) do
    if safe(C_PetBattles.GetPower, ENEMY, i) == power and safe(C_PetBattles.GetSpeed, ENEMY, i) == speed then
      local species = safe(C_PetBattles.GetPetSpeciesID, ENEMY, i)
      if species and C_PetJournal and C_PetJournal.GetPetAbilityList then
        local ids, levels = C_PetJournal.GetPetAbilityList(species)
        if ids then
          for k, id in ipairs(ids) do
            if id == abilityID then return i, ((k - 1) % 3) + 1 end
          end
        end
      end
    end
  end
end

-- ------------------------------------------------------------------------------------------ payload
local out
local function u8(v) v = math.floor(tonumber(v) or 0); if v < 0 then v = 0 elseif v > 255 then v = 255 end; out[#out + 1] = v end
local function u16(v)
  v = math.floor(tonumber(v) or 0); if v < 0 then v = 0 elseif v > 65535 then v = 65535 end
  out[#out + 1] = v % 256; out[#out + 1] = math.floor(v / 256)
end

local function auras(owner, idx)
  local list = {}
  local n = safe(C_PetBattles.GetNumAuras, owner, idx) or 0
  for i = 1, n do
    local id, _, turns, _, co, ci = safe(C_PetBattles.GetAuraInfo, owner, idx, i)
    if id then list[#list + 1] = {id, turns or -1, co or 0, ci or 0} end
  end
  u8(#list)
  for _, a in ipairs(list) do
    u16(a[1]); u8(a[2] + 1); u8(a[3] * 16 + a[4])
  end
end

local function payload()
  out = {}
  u8(66); u8(1)
  u8(round)
  local select = safe(C_PetBattles.ShouldShowPetSelect) and 1 or 0
  local lead = (select == 1 and round == 0) and 2 or 0
  u8(select + lead)
  u8(safe(C_PetBattles.GetActivePet, ALLY) or 0)
  u8(safe(C_PetBattles.GetActivePet, ENEMY) or 0)
  auras(0, PAD)  -- weather
  for owner = ALLY, ENEMY do
    auras(owner, PAD)
    local n = safe(C_PetBattles.GetNumPets, owner) or 0
    u8(n)
    for i = 1, n do
      u16(safe(C_PetBattles.GetPetSpeciesID, owner, i))
      u8(safe(C_PetBattles.GetLevel, owner, i))
      u8(safe(C_PetBattles.GetBreedQuality, owner, i))
      u16(safe(C_PetBattles.GetHealth, owner, i))
      u16(safe(C_PetBattles.GetMaxHealth, owner, i))
      u16(safe(C_PetBattles.GetPower, owner, i))
      u16(safe(C_PetBattles.GetSpeed, owner, i))
      u8((safe(C_PetBattles.GetStateValue, owner, i, 25) or 0) + 128)
      local usable = 0
      for s = 1, 3 do
        local id = safe(C_PetBattles.GetAbilityInfo, owner, i, s)
        if owner == ENEMY and not id and seen[i] then id = seen[i][s] end
        u16(id or 0)
      end
      for s = 1, 3 do
        local ok, cd, lock = safe(C_PetBattles.GetAbilityState, owner, i, s)
        if ok then usable = usable + 2 ^ (s - 1) end
        u8(math.max(cd or 0, lock or 0))
      end
      u8(usable)
      auras(owner, i)
    end
  end
  return out
end

-- ------------------------------------------------------------------------------------------ drawing
local lastKey
local function draw(bytes)
  local grid, size = BPAdvisor_QR(bytes)
  if not grid then return end
  local px = BPAdvisorDB.size or 4
  local quiet = 4
  frame:SetSize((size + 2 * quiet) * px, (size + 2 * quiet) * px)
  local k = 0
  for r = 0, size - 1 do
    for c = 0, size - 1 do
      if grid[r][c] == 1 then
        k = k + 1
        local t = cells[k]
        if not t then
          t = frame:CreateTexture(nil, "ARTWORK")
          t:SetColorTexture(0, 0, 0, 1)
          cells[k] = t
        end
        t:SetSize(px, px)
        t:SetPoint("TOPLEFT", frame, "TOPLEFT", (c + quiet) * px, -(r + quiet) * px)
        t:Show()
      end
    end
  end
  for i = k + 1, #cells do cells[i]:Hide() end
end

local function update()
  if not C_PetBattles.IsInBattle() or BPAdvisorDB.hidden then
    frame:Hide()
    return
  end
  local bytes = payload()
  local key = table.concat(bytes, ",")
  if key ~= lastKey then
    lastKey = key
    draw(bytes)
  end
  frame:Show()
end

local ev = CreateFrame("Frame")
ev:RegisterEvent("PET_BATTLE_OPENING_START")
ev:RegisterEvent("PET_BATTLE_PET_ROUND_PLAYBACK_COMPLETE")
ev:RegisterEvent("PET_BATTLE_CLOSE")
ev:RegisterEvent("CHAT_MSG_PET_BATTLE_COMBAT_LOG")
local ticker
ev:SetScript("OnEvent", function(_, event, ...)
  if event == "PET_BATTLE_OPENING_START" then
    seen, round, lastKey = {}, 0, nil
    if not ticker then ticker = C_Timer.NewTicker(0.5, update) end
  elseif event == "PET_BATTLE_PET_ROUND_PLAYBACK_COMPLETE" then
    round = (...) or (round + 1)
  elseif event == "CHAT_MSG_PET_BATTLE_COMBAT_LOG" then
    local msg = ...
    for id, _, power, speed in string.gmatch(msg or "", "HbattlePetAbil:(%d+):(%d+):(%d+):(%d+)") do
      local i, slot = enemySlotOf(tonumber(id), tonumber(power), tonumber(speed))
      if i then
        seen[i] = seen[i] or {}
        seen[i][slot] = tonumber(id)
      end
    end
  elseif event == "PET_BATTLE_CLOSE" then
    frame:Hide()
    if ticker then ticker:Cancel(); ticker = nil end
  end
end)

SLASH_BPADVISOR1 = "/bpa"
SlashCmdList.BPADVISOR = function(msg)
  local cmd, arg = strsplit(" ", msg or "")
  if cmd == "size" and tonumber(arg) then
    BPAdvisorDB.size = math.max(2, math.min(8, tonumber(arg)))
    lastKey = nil
    update()
  elseif cmd == "reset" then
    BPAdvisorDB.pos = nil
    place()
  elseif cmd == "test" then
    local data = {}
    for i = 0, 199 do data[i + 1] = (i * 7 + 3) % 256 end
    local g, n = BPAdvisor_QR(data)
    local s, dark = 0, 0
    for r = 0, n - 1 do
      for c = 0, n - 1 do
        if g[r][c] == 1 then
          dark = dark + 1
          s = (s + (r * 131 + c * 7 + 1) * (r + 3)) % 1000003
        end
      end
    end
    local ok = (n == 53 and s == 248222 and dark == 1449)
    print(string.format("|cff33ff99BPAdvisor|r: QR self-test %s (size %d, sum %d, dark %d)", ok and "OK" or "FAILED", n, s, dark))
  else
    BPAdvisorDB.hidden = not BPAdvisorDB.hidden
    print("|cff33ff99BPAdvisor|r: " .. (BPAdvisorDB.hidden and "hidden" or "shown"))
    update()
  end
end
