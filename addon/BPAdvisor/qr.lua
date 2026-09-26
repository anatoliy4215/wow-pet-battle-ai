-- QR encoder (byte mode, error correction L, versions 1-15). Line-by-line port of bp/advisor/qr_ref.py,
-- which is checked against a real decoder. Grid is 0-based: grid[r][c] = 1 (dark) / 0.
local band, bxor, lshift, rshift = bit.band, bit.bxor, bit.lshift, bit.rshift

local BLOCKS_L = {
  [1] = {{{1, 19}}, 7}, [2] = {{{1, 34}}, 10}, [3] = {{{1, 55}}, 15}, [4] = {{{1, 80}}, 20}, [5] = {{{1, 108}}, 26},
  [6] = {{{2, 68}}, 18}, [7] = {{{2, 78}}, 20}, [8] = {{{2, 97}}, 24}, [9] = {{{2, 116}}, 30},
  [10] = {{{2, 68}, {2, 69}}, 18}, [11] = {{{4, 81}}, 20}, [12] = {{{2, 92}, {2, 93}}, 24}, [13] = {{{4, 107}}, 26},
  [14] = {{{3, 115}, {1, 116}}, 30}, [15] = {{{5, 87}, {1, 88}}, 22},
}
local ALIGN = {[1] = {}, [2] = {6, 18}, [3] = {6, 22}, [4] = {6, 26}, [5] = {6, 30}, [6] = {6, 34}, [7] = {6, 22, 38},
  [8] = {6, 24, 42}, [9] = {6, 26, 46}, [10] = {6, 28, 50}, [11] = {6, 30, 54}, [12] = {6, 32, 58},
  [13] = {6, 34, 62}, [14] = {6, 26, 46, 66}, [15] = {6, 26, 48, 70}}
local REMAINDER = {[1] = 0, [2] = 7, [3] = 7, [4] = 7, [5] = 7, [6] = 7, [7] = 0, [8] = 0, [9] = 0, [10] = 0,
  [11] = 0, [12] = 0, [13] = 0, [14] = 3, [15] = 3}

local EXP, LOG = {}, {}
do
  local x = 1
  for i = 0, 254 do
    EXP[i] = x
    LOG[x] = i
    x = lshift(x, 1)
    if band(x, 0x100) ~= 0 then x = bxor(x, 0x11D) end
  end
  for i = 255, 511 do EXP[i] = EXP[i - 255] end
end

local function gfmul(a, b)
  if a == 0 or b == 0 then return 0 end
  return EXP[LOG[a] + LOG[b]]
end

local GEN = {}
local function generator(n)
  if GEN[n] then return GEN[n] end
  local g = {1}
  for i = 0, n - 1 do
    local ng = {}
    for j = 1, #g + 1 do ng[j] = 0 end
    for j = 1, #g do
      ng[j] = bxor(ng[j], g[j])
      ng[j + 1] = bxor(ng[j + 1], gfmul(g[j], EXP[i]))
    end
    g = ng
  end
  GEN[n] = g
  return g
end

local function rsEC(data, n)
  local g = generator(n)
  local res = {}
  for i = 1, #data do res[i] = data[i] end
  for i = 1, n do res[#data + i] = 0 end
  for i = 1, #data do
    local c = res[i]
    if c ~= 0 then
      for j = 2, #g do res[i + j - 1] = bxor(res[i + j - 1], gfmul(g[j], c)) end
    end
  end
  local out = {}
  for i = 1, n do out[i] = res[#data + i] end
  return out
end

local function capacity(v)
  local c = 0
  for _, grp in ipairs(BLOCKS_L[v][1]) do c = c + grp[1] * grp[2] end
  return c
end

local function pickVersion(nbytes)
  for v = 1, 15 do
    local header = 4 + ((v < 10) and 8 or 16)
    if header + 8 * nbytes <= 8 * capacity(v) then return v end
  end
  return nil
end

local function codewords(data, v)
  local cap = capacity(v)
  local bits = {}
  local function put(val, n)
    for i = n - 1, 0, -1 do bits[#bits + 1] = band(rshift(val, i), 1) end
  end
  put(4, 4)
  put(#data, (v < 10) and 8 or 16)
  for i = 1, #data do put(data[i], 8) end
  for _ = 1, math.min(4, cap * 8 - #bits) do bits[#bits + 1] = 0 end
  while #bits % 8 ~= 0 do bits[#bits + 1] = 0 end
  local cw = {}
  for i = 1, #bits, 8 do
    local b = 0
    for k = 0, 7 do b = b * 2 + bits[i + k] end
    cw[#cw + 1] = b
  end
  local pad, pi = {0xEC, 0x11}, 0
  while #cw < cap do
    cw[#cw + 1] = pad[pi % 2 + 1]
    pi = pi + 1
  end
  local groups, ecn = BLOCKS_L[v][1], BLOCKS_L[v][2]
  local blocks, pos = {}, 1
  for _, grp in ipairs(groups) do
    for _ = 1, grp[1] do
      local blk = {}
      for k = 0, grp[2] - 1 do blk[k + 1] = cw[pos + k] end
      pos = pos + grp[2]
      blocks[#blocks + 1] = blk
    end
  end
  local ecs = {}
  for i, blk in ipairs(blocks) do ecs[i] = rsEC(blk, ecn) end
  local out, maxlen = {}, 0
  for _, blk in ipairs(blocks) do if #blk > maxlen then maxlen = #blk end end
  for i = 1, maxlen do
    for _, blk in ipairs(blocks) do if i <= #blk then out[#out + 1] = blk[i] end end
  end
  for i = 1, ecn do
    for _, e in ipairs(ecs) do out[#out + 1] = e[i] end
  end
  return out
end

local function bchFormat(ec, mask)
  local data = bit.bor(lshift(ec, 3), mask)
  local rem = lshift(data, 10)
  for i = 14, 10, -1 do
    if band(rem, lshift(1, i)) ~= 0 then rem = bxor(rem, lshift(0x537, i - 10)) end
  end
  return bxor(bit.bor(lshift(data, 10), rem), 0x5412)
end

local function bchVersion(v)
  local rem = lshift(v, 12)
  for i = 17, 12, -1 do
    if band(rem, lshift(1, i)) ~= 0 then rem = bxor(rem, lshift(0x1F25, i - 12)) end
  end
  return bit.bor(lshift(v, 12), rem)
end

local function maskCond(k, r, c)
  if k == 0 then return (r + c) % 2 == 0
  elseif k == 1 then return r % 2 == 0
  elseif k == 2 then return c % 3 == 0
  elseif k == 3 then return (r + c) % 3 == 0
  elseif k == 4 then return (math.floor(r / 2) + math.floor(c / 3)) % 2 == 0
  elseif k == 5 then return (r * c) % 2 + (r * c) % 3 == 0
  elseif k == 6 then return ((r * c) % 2 + (r * c) % 3) % 2 == 0
  else return ((r + c) % 2 + (r * c) % 3) % 2 == 0 end
end

-- data: array of byte values (1-based). Returns grid (0-based rows/cols) and size, or nil if too long.
function BPAdvisor_QR(data, maskChoice)
  local v = pickVersion(#data)
  if not v then return nil end
  local size = 17 + 4 * v
  local m, fn = {}, {}
  for r = 0, size - 1 do
    m[r], fn[r] = {}, {}
    for c = 0, size - 1 do m[r][c] = 0; fn[r][c] = false end
  end
  local function setf(r, c, val)
    if r >= 0 and r < size and c >= 0 and c < size then m[r][c] = val; fn[r][c] = true end
  end
  for _, rc in ipairs({{0, 0}, {0, size - 7}, {size - 7, 0}}) do
    for r = -1, 7 do
      for c = -1, 7 do
        local inside = r >= 0 and r <= 6 and c >= 0 and c <= 6
        local dark = inside and (r == 0 or r == 6 or c == 0 or c == 6 or (r >= 2 and r <= 4 and c >= 2 and c <= 4))
        setf(rc[1] + r, rc[2] + c, dark and 1 or 0)
      end
    end
  end
  for i = 8, size - 9 do
    setf(6, i, (i % 2 == 0) and 1 or 0)
    setf(i, 6, (i % 2 == 0) and 1 or 0)
  end
  local al = ALIGN[v]
  for ai = 1, #al do
    for aj = 1, #al do
      if not ((ai == 1 and aj == 1) or (ai == 1 and aj == #al) or (ai == #al and aj == 1)) then
        for dr = -2, 2 do
          for dc = -2, 2 do
            setf(al[ai] + dr, al[aj] + dc, (math.max(math.abs(dr), math.abs(dc)) ~= 1) and 1 or 0)
          end
        end
      end
    end
  end
  for i = 0, 8 do
    if not fn[8][i] then setf(8, i, 0) end
    if not fn[i][8] then setf(i, 8, 0) end
  end
  for i = 0, 7 do
    setf(8, size - 1 - i, 0)
    setf(size - 1 - i, 8, 0)
  end
  if v >= 7 then
    for i = 0, 5 do
      for j = 0, 2 do
        setf(i, size - 11 + j, 0)
        setf(size - 11 + j, i, 0)
      end
    end
  end
  setf(size - 8, 8, 1)
  local cw = codewords(data, v)
  local bits = {}
  for _, b in ipairs(cw) do
    for i = 7, 0, -1 do bits[#bits + 1] = band(rshift(b, i), 1) end
  end
  for _ = 1, REMAINDER[v] do bits[#bits + 1] = 0 end
  local idx, upward, c = 1, true, size - 1
  while c > 0 do
    if c == 6 then c = c - 1 end
    local r0, r1, step = 0, size - 1, 1
    if upward then r0, r1, step = size - 1, 0, -1 end
    for r = r0, r1, step do
      for _, cc in ipairs({c, c - 1}) do
        if not fn[r][cc] then
          m[r][cc] = bits[idx] or 0
          idx = idx + 1
        end
      end
    end
    upward = not upward
    c = c - 2
  end

  local function masked(k)
    local out = {}
    for r = 0, size - 1 do
      out[r] = {}
      for cc = 0, size - 1 do
        local val = m[r][cc]
        if not fn[r][cc] and maskCond(k, r, cc) then val = 1 - val end
        out[r][cc] = val
      end
    end
    local fmt = bchFormat(1, k)
    local cols = {7, 5, 4, 3, 2, 1, 0}
    for i = 0, 14 do
      local b = band(rshift(fmt, i), 1)
      if i < 8 then out[8][size - 1 - i] = b else out[8][cols[i - 7]] = b end
      if i < 6 then out[i][8] = b
      elseif i < 8 then out[i + 1][8] = b
      else out[size - 15 + i][8] = b end
    end
    if v >= 7 then
      local vb = bchVersion(v)
      for i = 0, 17 do
        local b = band(rshift(vb, i), 1)
        local a, bb = math.floor(i / 3), i % 3
        out[a][size - 11 + bb] = b
        out[size - 11 + bb][a] = b
      end
    end
    return out
  end

  local function penalty(g)
    local p = 0
    for pass = 1, 2 do
      for i = 0, size - 1 do
        local run, prev = 0, -1
        for j = 0, size - 1 do
          local val = (pass == 1) and g[i][j] or g[j][i]
          if val == prev then run = run + 1
          else
            if run >= 5 then p = p + run - 2 end
            run, prev = 1, val
          end
        end
        if run >= 5 then p = p + run - 2 end
      end
    end
    local dark = 0
    for r = 0, size - 2 do
      for cc = 0, size - 2 do
        local s = g[r][cc] + g[r + 1][cc] + g[r][cc + 1] + g[r + 1][cc + 1]
        if s == 0 or s == 4 then p = p + 3 end
      end
    end
    for r = 0, size - 1 do for cc = 0, size - 1 do dark = dark + g[r][cc] end end
    p = p + math.abs(math.floor(dark * 20 / (size * size)) - 10) * 10
    return p
  end

  if maskChoice then return masked(maskChoice), size end
  local best, bestP
  for k = 0, 7 do
    local g = masked(k)
    local pp = penalty(g)
    if not bestP or pp < bestP then best, bestP = g, pp end
  end
  return best, size
end
