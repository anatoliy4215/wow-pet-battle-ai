//! Cards and hand strength.
//!
//! A card is 0..51 = rank * 4 + suit, rank 0 = deuce .. 12 = ace. A hand value is a u32 that
//! orders hands exactly as poker does: category << 20 | up to five 4-bit ranks (most significant
//! first). Categories: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house,
//! 7 quads, 8 straight flush. No tables: bit masks and counts (~tens of millions of 7-card hands/s).

pub const RANKS: &[u8; 13] = b"23456789TJQKA";
pub const SUITS: &[u8; 4] = b"cdhs";

#[inline]
pub fn rank(c: u8) -> u8 {
    c >> 2
}
#[inline]
pub fn suit(c: u8) -> u8 {
    c & 3
}

pub fn card_str(c: u8) -> String {
    format!("{}{}", RANKS[rank(c) as usize] as char, SUITS[suit(c) as usize] as char)
}

pub fn parse_card(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let r = RANKS.iter().position(|&x| x == b[0].to_ascii_uppercase())? as u8;
    let s = SUITS.iter().position(|&x| x == b[1].to_ascii_lowercase())? as u8;
    Some(r * 4 + s)
}

pub const HIGH: u32 = 0;
pub const PAIR: u32 = 1;
pub const TWO_PAIR: u32 = 2;
pub const TRIPS: u32 = 3;
pub const STRAIGHT: u32 = 4;
pub const FLUSH: u32 = 5;
pub const FULL_HOUSE: u32 = 6;
pub const QUADS: u32 = 7;
pub const STRAIGHT_FLUSH: u32 = 8;
pub const CATEGORY_NAMES: [&str; 9] =
    ["старшая карта", "пара", "две пары", "сет", "стрит", "флеш", "фулл-хаус", "каре", "стрит-флеш"];

/// Highest straight in a 13-bit rank mask: its top rank, or None. A-2-3-4-5 counts with top 5 (rank 3).
#[inline]
fn straight_top(mask: u16) -> Option<u32> {
    let m = mask as u32;
    for hi in (4..13).rev() {
        let need = 0x1f << (hi - 4);
        if m & need == need {
            return Some(hi);
        }
    }
    if m & 0x100f == 0x100f {
        return Some(3); // wheel
    }
    None
}

/// The `n` highest ranks set in `mask`, packed 4 bits each (first = most significant).
#[inline]
fn top_ranks(mask: u16, n: u32) -> u32 {
    let mut out = 0u32;
    let mut left = n;
    let mut r = 12i32;
    while left > 0 && r >= 0 {
        if mask & (1 << r) != 0 {
            out = (out << 4) | r as u32;
            left -= 1;
        }
        r -= 1;
    }
    out << (4 * left)
}

#[inline]
fn pack(cat: u32, ranks: u32, n: u32) -> u32 {
    // `ranks` holds n ranks; left-align them into the 20 kicker bits
    (cat << 20) | (ranks << (4 * (5 - n)))
}

/// Value of the best 5-card hand among 5..7 cards.
pub fn evaluate(cards: &[u8]) -> u32 {
    let mut suit_mask = [0u16; 4];
    let mut counts = [0u8; 13];
    let mut all: u16 = 0;
    for &c in cards {
        let r = rank(c);
        suit_mask[suit(c) as usize] |= 1 << r;
        counts[r as usize] += 1;
        all |= 1 << r;
    }
    // flush / straight flush
    for &sm in &suit_mask {
        if sm.count_ones() >= 5 {
            if let Some(hi) = straight_top(sm) {
                return pack(STRAIGHT_FLUSH, hi, 1);
            }
            return pack(FLUSH, top_ranks(sm, 5), 5);
        }
    }
    let (mut quad, mut trips, mut pairs) = (0u16, 0u16, 0u16);
    for r in 0..13 {
        match counts[r] {
            4 => quad |= 1 << r,
            3 => trips |= 1 << r,
            2 => pairs |= 1 << r,
            _ => {}
        }
    }
    if quad != 0 {
        let q = 15 - quad.leading_zeros(); // highest set bit index
        let q = q as u32;
        let kick = top_ranks(all & !(1 << q), 1);
        return pack(QUADS, (q << 4) | kick, 2);
    }
    if trips != 0 {
        let t = 15 - trips.leading_zeros();
        let rest_pairs = pairs | (trips & !(1 << t));
        if rest_pairs != 0 {
            let p = 15 - rest_pairs.leading_zeros();
            return pack(FULL_HOUSE, (t << 4) | p, 2);
        }
    }
    if let Some(hi) = straight_top(all) {
        return pack(STRAIGHT, hi, 1);
    }
    if trips != 0 {
        let t = 15 - trips.leading_zeros();
        let k = top_ranks(all & !(1 << t), 2);
        return pack(TRIPS, (t << 8) | k, 3);
    }
    if pairs.count_ones() >= 2 {
        let p1 = 15 - pairs.leading_zeros();
        let p2 = 15 - (pairs & !(1 << p1)).leading_zeros();
        let k = top_ranks(all & !(1 << p1) & !(1 << p2), 1);
        return pack(TWO_PAIR, (p1 << 8) | (p2 << 4) | k, 3);
    }
    if pairs != 0 {
        let p = 15 - pairs.leading_zeros();
        let k = top_ranks(all & !(1 << p), 3);
        return pack(PAIR, (p << 12) | k, 4);
    }
    pack(HIGH, top_ranks(all, 5), 5)
}

#[inline]
pub fn category(value: u32) -> u32 {
    value >> 20
}

/// Slow, obviously correct reference: best of all 5-card subsets, each judged from first principles.
pub fn evaluate_reference(cards: &[u8]) -> u32 {
    fn five(c: &[u8; 5]) -> u32 {
        let mut ranks: Vec<u32> = c.iter().map(|&x| rank(x) as u32).collect();
        ranks.sort_unstable_by(|a, b| b.cmp(a));
        let flush = c.iter().all(|&x| suit(x) == suit(c[0]));
        let mut uniq = ranks.clone();
        uniq.dedup();
        let straight_hi = if uniq.len() == 5 && ranks[0] - ranks[4] == 4 {
            Some(ranks[0])
        } else if uniq == vec![12, 3, 2, 1, 0] {
            Some(3)
        } else {
            None
        };
        // groups by (count desc, rank desc)
        let mut groups: Vec<(u32, u32)> = uniq.iter().map(|&r| (ranks.iter().filter(|&&x| x == r).count() as u32, r)).collect();
        groups.sort_unstable_by(|a, b| b.cmp(a));
        let order: Vec<u32> = groups.iter().map(|g| g.1).collect();
        let packn = |cat: u32, rs: &[u32]| {
            let mut v = 0u32;
            for &r in rs {
                v = (v << 4) | r;
            }
            (cat << 20) | (v << (4 * (5 - rs.len() as u32)))
        };
        match (straight_hi, flush) {
            (Some(h), true) => return packn(STRAIGHT_FLUSH, &[h]),
            _ => {}
        }
        if groups[0].0 == 4 {
            return packn(QUADS, &order);
        }
        if groups[0].0 == 3 && groups[1].0 == 2 {
            return packn(FULL_HOUSE, &order);
        }
        if flush {
            return packn(FLUSH, &ranks);
        }
        if let Some(h) = straight_hi {
            return packn(STRAIGHT, &[h]);
        }
        if groups[0].0 == 3 {
            return packn(TRIPS, &order);
        }
        if groups[0].0 == 2 && groups[1].0 == 2 {
            return packn(TWO_PAIR, &order);
        }
        if groups[0].0 == 2 {
            return packn(PAIR, &order);
        }
        packn(HIGH, &ranks)
    }
    let n = cards.len();
    let mut best = 0;
    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                for d in c + 1..n {
                    for e in d + 1..n {
                        best = best.max(five(&[cards[a], cards[b], cards[c], cards[d], cards[e]]));
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(s: &str) -> Vec<u8> {
        s.split_whitespace().map(|x| parse_card(x).unwrap()).collect()
    }

    #[test]
    fn known_orderings() {
        let order = [
            "2c 3d 4h 5s 7c", "2c 2d 4h 5s 7c", "2c 2d 4h 4s 7c", "2c 2d 2h 5s 7c", "Ac 2d 3h 4s 5c",
            "6c 2d 3h 4s 5c", "2c 4c 6c 8c Tc", "2c 2d 2h 5s 5c", "2c 2d 2h 2s 5c", "Ac 2c 3c 4c 5c",
            "Tc Jc Qc Kc Ac",
        ];
        let vals: Vec<u32> = order.iter().map(|h| evaluate(&hand(h))).collect();
        for w in vals.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn random_hands_match_reference() {
        let mut x: u64 = 88172645463325252;
        for _ in 0..200_000 {
            let mut deck: Vec<u8> = (0..52).collect();
            for i in 0..7 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let j = i + (x % (52 - i as u64)) as usize;
                deck.swap(i, j);
            }
            let cards = &deck[..7];
            assert_eq!(evaluate(cards), evaluate_reference(cards), "{:?}", cards.iter().map(|&c| card_str(c)).collect::<Vec<_>>());
        }
    }
}

#[cfg(test)]
mod exhaustive {
    use super::*;

    /// All C(52,7) = 133,784,560 seven-card hands: the count of each category is known exactly.
    #[test]
    #[ignore] // ~seconds in release: cargo test --release -- --ignored
    fn seven_card_category_counts() {
        let t = std::time::Instant::now();
        let mut counts = [0u64; 9];
        let mut c = [0u8; 7];
        for a in 0..46u8 {
            c[0] = a;
            for b in a + 1..47 {
                c[1] = b;
                for d in b + 1..48 {
                    c[2] = d;
                    for e in d + 1..49 {
                        c[3] = e;
                        for f in e + 1..50 {
                            c[4] = f;
                            for g in f + 1..51 {
                                c[5] = g;
                                for h in g + 1..52 {
                                    c[6] = h;
                                    counts[category(evaluate(&c)) as usize] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        let secs = t.elapsed().as_secs_f64();
        println!("133,784,560 hands in {secs:.2}s = {:.0}M hands/s", 133.78456 / secs);
        assert_eq!(counts, [23294460, 58627800, 31433400, 6461620, 6180020, 4047644, 3473184, 224848, 41584]);
    }
}
