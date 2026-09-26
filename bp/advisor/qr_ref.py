"""Reference QR encoder (byte mode, error correction L, versions 1-15).

The in-game addon (addon/BPAdvisor/qr.lua) is a line-by-line port of this file; this version exists
so the algorithm can be checked with a real QR decoder outside the game.
"""
from __future__ import annotations

# data codewords per block groups and EC codewords per block, level L
BLOCKS_L = {
    1: ([(1, 19)], 7), 2: ([(1, 34)], 10), 3: ([(1, 55)], 15), 4: ([(1, 80)], 20), 5: ([(1, 108)], 26),
    6: ([(2, 68)], 18), 7: ([(2, 78)], 20), 8: ([(2, 97)], 24), 9: ([(2, 116)], 30),
    10: ([(2, 68), (2, 69)], 18), 11: ([(4, 81)], 20), 12: ([(2, 92), (2, 93)], 24), 13: ([(4, 107)], 26),
    14: ([(3, 115), (1, 116)], 30), 15: ([(5, 87), (1, 88)], 22),
}
ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42],
         9: [6, 26, 46], 10: [6, 28, 50], 11: [6, 30, 54], 12: [6, 32, 58], 13: [6, 34, 62],
         14: [6, 26, 46, 66], 15: [6, 26, 48, 70]}
REMAINDER = {1: 0, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 3, 15: 3}

EXP = [0] * 512
LOG = [0] * 256
x = 1
for i in range(255):
    EXP[i] = x
    LOG[x] = i
    x <<= 1
    if x & 0x100:
        x ^= 0x11D
for i in range(255, 512):
    EXP[i] = EXP[i - 255]


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def rs_generator(n):
    g = [1]
    for i in range(n):
        ng = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            ng[j] ^= c
            ng[j + 1] ^= gf_mul(c, EXP[i])
        g = ng
    return g


def rs_ec(data, n):
    g = rs_generator(n)
    res = list(data) + [0] * n
    for i in range(len(data)):
        c = res[i]
        if c:
            for j in range(1, len(g)):
                res[i + j] ^= gf_mul(g[j], c)
    return res[len(data):]


def capacity(v):
    groups, _ = BLOCKS_L[v]
    return sum(n * k for n, k in groups)


def pick_version(nbytes):
    for v in range(1, 16):
        header_bits = 4 + (8 if v < 10 else 16)
        if header_bits + 8 * nbytes <= 8 * capacity(v):
            return v
    raise ValueError("payload too large")


def codewords(data, v):
    cap = capacity(v)
    bits = []

    def put(val, n):
        for i in range(n - 1, -1, -1):
            bits.append((val >> i) & 1)

    put(4, 4)
    put(len(data), 8 if v < 10 else 16)
    for b in data:
        put(b, 8)
    for _ in range(min(4, cap * 8 - len(bits))):
        bits.append(0)
    while len(bits) % 8:
        bits.append(0)
    cw = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = [0xEC, 0x11]
    i = 0
    while len(cw) < cap:
        cw.append(pad[i % 2])
        i += 1
    groups, ecn = BLOCKS_L[v]
    blocks, pos = [], 0
    for n, k in groups:
        for _ in range(n):
            blocks.append(cw[pos:pos + k])
            pos += k
    ecs = [rs_ec(b, ecn) for b in blocks]
    out = []
    for i in range(max(len(b) for b in blocks)):
        for b in blocks:
            if i < len(b):
                out.append(b[i])
    for i in range(ecn):
        for e in ecs:
            out.append(e[i])
    return out


def bch_format(ec_bits, mask):
    data = (ec_bits << 3) | mask
    rem = data << 10
    for i in range(14, 9, -1):
        if rem & (1 << i):
            rem ^= 0x537 << (i - 10)
    return ((data << 10) | rem) ^ 0x5412


def bch_version(v):
    rem = v << 12
    for i in range(17, 11, -1):
        if rem & (1 << i):
            rem ^= 0x1F25 << (i - 12)
    return (v << 12) | rem


def encode(data: bytes, mask=None):
    """-> list of rows (1 = dark)."""
    v = pick_version(len(data))
    size = 17 + 4 * v
    m = [[None] * size for _ in range(size)]
    fn = [[False] * size for _ in range(size)]

    def setf(r, c, val):
        if 0 <= r < size and 0 <= c < size:
            m[r][c] = val
            fn[r][c] = True

    for (r0, c0) in ((0, 0), (0, size - 7), (size - 7, 0)):
        for r in range(-1, 8):
            for c in range(-1, 8):
                inside = 0 <= r <= 6 and 0 <= c <= 6
                dark = inside and (r in (0, 6) or c in (0, 6) or (2 <= r <= 4 and 2 <= c <= 4))
                setf(r0 + r, c0 + c, 1 if dark else 0)
    for i in range(8, size - 8):
        setf(6, i, 1 if i % 2 == 0 else 0)
        setf(i, 6, 1 if i % 2 == 0 else 0)
    al = ALIGN[v]
    last = len(al) - 1
    for ai, r in enumerate(al):
        for aj, c in enumerate(al):
            if (ai, aj) in ((0, 0), (0, last), (last, 0)):
                continue  # these would overlap the finder patterns
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    setf(r + dr, c + dc, 1 if max(abs(dr), abs(dc)) != 1 else 0)
    # reserve format / version areas
    for i in range(9):
        if not fn[8][i]:
            setf(8, i, 0)
        if not fn[i][8]:
            setf(i, 8, 0)
    for i in range(8):
        setf(8, size - 1 - i, 0)
        setf(size - 1 - i, 8, 0)
    if v >= 7:
        for i in range(6):
            for j in range(3):
                setf(i, size - 11 + j, 0)
                setf(size - 11 + j, i, 0)
    setf(size - 8, 8, 1)  # dark module
    # data
    cw = codewords(data, v)
    bits = []
    for b in cw:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    bits += [0] * REMAINDER[v]
    idx = 0
    upward = True
    c = size - 1
    while c > 0:
        if c == 6:
            c -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for r in rows:
            for cc in (c, c - 1):
                if not fn[r][cc]:
                    m[r][cc] = bits[idx] if idx < len(bits) else 0
                    idx += 1
        upward = not upward
        c -= 2

    def masked(k):
        out = [row[:] for row in m]
        for r in range(size):
            for cc in range(size):
                if fn[r][cc]:
                    continue
                cond = [(r + cc) % 2 == 0, r % 2 == 0, cc % 3 == 0, (r + cc) % 3 == 0,
                         (r // 2 + cc // 3) % 2 == 0, (r * cc) % 2 + (r * cc) % 3 == 0,
                         ((r * cc) % 2 + (r * cc) % 3) % 2 == 0, ((r + cc) % 2 + (r * cc) % 3) % 2 == 0][k]
                if cond:
                    out[r][cc] ^= 1
        fmt = bch_format(1, k)  # level L = 01
        # horizontal row 8, columns 0..8 (skipping timing col 6) for bits 14..8 and 7..0 standard layout
        for i in range(15):
            bit = (fmt >> i) & 1
            if i < 8:
                out[8][size - 1 - i] = bit
            else:
                col = [7, 5, 4, 3, 2, 1, 0][i - 8]
                out[8][col] = bit
        for i in range(15):
            bit = (fmt >> i) & 1
            if i < 6:
                out[i][8] = bit
            elif i < 8:
                out[i + 1][8] = bit
            else:
                out[size - 15 + i][8] = bit
        if v >= 7:
            vb = bch_version(v)
            for i in range(18):
                bit = (vb >> i) & 1
                a, b = i // 3, i % 3
                out[a][size - 11 + b] = bit
                out[size - 11 + b][a] = bit
        return out

    def penalty(g):
        n = len(g)
        p = 0
        for rows in (g, [list(x) for x in zip(*g)]):
            for row in rows:
                run, prev = 0, None
                for val in row:
                    if val == prev:
                        run += 1
                    else:
                        if run >= 5:
                            p += run - 2
                        run, prev = 1, val
                if run >= 5:
                    p += run - 2
        for r in range(n - 1):
            for cc in range(n - 1):
                s = g[r][cc] + g[r + 1][cc] + g[r][cc + 1] + g[r + 1][cc + 1]
                if s in (0, 4):
                    p += 3
        dark = sum(map(sum, g))
        p += abs(dark * 20 // (n * n) - 10) * 10
        return p

    if mask is not None:
        return masked(mask)
    best = min(range(8), key=lambda k: penalty(masked(k)))
    return masked(best)
