//! One hand of heads-up no-limit hold'em.
//!
//! Seat `button` posts the small blind and acts first preflop; the other seat (big blind) acts first
//! on the flop, turn and river. Amounts are chips (big blind = 2). A raise is given as the total the
//! player will have put in on this street ("raise to"). A raise must add at least the previous raise
//! (at least one big blind), unless it is an all-in for less. Uncalled chips go back to their owner.

use crate::eval::evaluate;

pub const SB: u32 = 1;
pub const BB: u32 = 2;

/// xorshift64*: small, fast, good enough for dealing and Monte Carlo
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next() >> 32) * n) >> 32
    }
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Fold,
    Check,
    Call,
    Raise(u32), // raise to this street total
}

#[derive(Clone, Debug)]
pub struct Legal {
    pub fold: bool,
    pub check: bool,
    pub call: u32, // chips to call (0 = cannot call)
    pub min_raise: u32,
    pub max_raise: u32, // 0 = cannot raise
}

#[derive(Clone)]
pub struct Hand {
    pub deck: [u8; 52],
    pub next_card: usize,
    pub hole: [[u8; 2]; 2],
    pub board: Vec<u8>,
    pub button: usize,
    pub stack: [u32; 2],     // chips behind
    pub street_in: [u32; 2], // put in on this street
    pub total_in: [u32; 2],  // put in this hand
    pub street: usize,       // 0 preflop .. 3 river, 4 over
    pub to_act: usize,
    pub last_raise: u32,
    pub acted: [bool; 2],
    pub folded: Option<usize>,
    pub history: Vec<(usize, usize, Action)>, // (street, seat, action)
    pub result: Option<[i64; 2]>,              // chips won / lost by each seat
}

impl Hand {
    pub fn new(stacks: [u32; 2], button: usize, rng: &mut Rng) -> Hand {
        let mut deck = [0u8; 52];
        for (i, c) in deck.iter_mut().enumerate() {
            *c = i as u8;
        }
        for i in (1..52).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            deck.swap(i, j);
        }
        let bb = 1 - button;
        let mut h = Hand {
            deck,
            next_card: 4,
            hole: [[deck[0], deck[2]], [deck[1], deck[3]]],
            board: Vec::with_capacity(5),
            button,
            stack: stacks,
            street_in: [0; 2],
            total_in: [0; 2],
            street: 0,
            to_act: button,
            last_raise: BB,
            acted: [false; 2],
            folded: None,
            history: Vec::new(),
            result: None,
        };
        h.post(button, SB);
        h.post(bb, BB);
        h
    }

    fn post(&mut self, seat: usize, amount: u32) {
        let a = amount.min(self.stack[seat]);
        self.stack[seat] -= a;
        self.street_in[seat] += a;
        self.total_in[seat] += a;
    }

    pub fn over(&self) -> bool {
        self.result.is_some()
    }

    pub fn pot(&self) -> u32 {
        self.total_in[0] + self.total_in[1]
    }

    pub fn legal(&self) -> Legal {
        let me = self.to_act;
        let opp = 1 - me;
        let to_call = self.street_in[opp].saturating_sub(self.street_in[me]);
        let mut l = Legal { fold: to_call > 0, check: to_call == 0, call: 0, min_raise: 0, max_raise: 0 };
        if self.over() {
            return Legal { fold: false, check: false, call: 0, min_raise: 0, max_raise: 0 };
        }
        if to_call > 0 {
            l.call = to_call.min(self.stack[me]);
        }
        let max_to = self.street_in[me] + self.stack[me];
        // raising needs chips beyond the call and an opponent who can still act
        if self.stack[me] > to_call && self.stack[opp] > 0 {
            let min_to = self.street_in[opp] + self.last_raise.max(BB);
            l.max_raise = max_to;
            l.min_raise = min_to.min(max_to);
        }
        l
    }

    /// Apply an action of the seat to act; illegal amounts are clamped to the legal range, illegal
    /// kinds become check / call.
    pub fn act(&mut self, a: Action, rng: &mut Rng) {
        if self.over() {
            return;
        }
        let me = self.to_act;
        let opp = 1 - me;
        let l = self.legal();
        let a = match a {
            Action::Fold if !l.fold => Action::Check,
            Action::Check if !l.check => Action::Call,
            Action::Call if l.call == 0 => Action::Check,
            Action::Raise(_) if l.max_raise == 0 => if l.call > 0 { Action::Call } else { Action::Check },
            Action::Raise(to) => Action::Raise(to.clamp(l.min_raise, l.max_raise)),
            x => x,
        };
        self.history.push((self.street, me, a));
        self.acted[me] = true;
        match a {
            Action::Fold => {
                self.folded = Some(me);
                self.finish(rng);
                return;
            }
            Action::Check => {}
            Action::Call => {
                let c = l.call;
                self.post(me, c);
            }
            Action::Raise(to) => {
                let add = to - self.street_in[me];
                let raise_by = to.saturating_sub(self.street_in[opp]);
                if raise_by >= self.last_raise {
                    self.last_raise = raise_by;
                }
                self.post(me, add);
                self.acted[opp] = false;
            }
        }
        let matched = self.street_in[0] == self.street_in[1];
        let all_in_short = (self.stack[me] == 0 || self.stack[opp] == 0) && matches!(a, Action::Call);
        if (self.acted[0] && self.acted[1] && matched) || all_in_short {
            self.next_street(rng);
        } else {
            self.to_act = opp;
        }
    }

    fn deal(&mut self, n: usize) {
        for _ in 0..n {
            self.board.push(self.deck[self.next_card]);
            self.next_card += 1;
        }
    }

    fn next_street(&mut self, rng: &mut Rng) {
        // nobody can bet any more: run the board out
        if self.stack[0] == 0 || self.stack[1] == 0 {
            while self.board.len() < 5 {
                let n = if self.board.is_empty() { 3 } else { 1 };
                self.deal(n);
            }
            self.street = 4;
            self.finish(rng);
            return;
        }
        if self.street == 3 {
            self.street = 4;
            self.finish(rng);
            return;
        }
        self.street += 1;
        self.deal(if self.street == 1 { 3 } else { 1 });
        self.street_in = [0; 2];
        self.acted = [false; 2];
        self.last_raise = BB;
        self.to_act = 1 - self.button; // big blind first after the flop
    }

    fn finish(&mut self, _rng: &mut Rng) {
        let matched = self.total_in[0].min(self.total_in[1]);
        let mut back = [0u32; 2];
        for s in 0..2 {
            back[s] = self.total_in[s] - matched; // uncalled chips
        }
        let pot = 2 * matched;
        let mut win = [0u32; 2];
        match self.folded {
            Some(f) => win[1 - f] = pot,
            None => {
                let v0 = self.value(0);
                let v1 = self.value(1);
                if v0 > v1 {
                    win[0] = pot;
                } else if v1 > v0 {
                    win[1] = pot;
                } else {
                    win[0] = pot / 2;
                    win[1] = pot - pot / 2;
                }
            }
        }
        let mut res = [0i64; 2];
        for s in 0..2 {
            self.stack[s] += win[s] + back[s];
            res[s] = (win[s] + back[s]) as i64 - self.total_in[s] as i64;
        }
        self.street = 4;
        self.result = Some(res);
    }

    pub fn value(&self, seat: usize) -> u32 {
        let mut cards = [0u8; 7];
        cards[0] = self.hole[seat][0];
        cards[1] = self.hole[seat][1];
        let n = self.board.len();
        cards[2..2 + n].copy_from_slice(&self.board);
        evaluate(&cards[..2 + n])
    }
}

/// Chance to win (ties count half) against one random hand, by Monte Carlo over the unseen cards.
pub fn equity(hole: [u8; 2], board: &[u8], samples: u32, rng: &mut Rng) -> f64 {
    let mut used = [false; 52];
    used[hole[0] as usize] = true;
    used[hole[1] as usize] = true;
    for &c in board {
        used[c as usize] = true;
    }
    let rest: Vec<u8> = (0..52u8).filter(|&c| !used[c as usize]).collect();
    let need = 2 + (5 - board.len());
    let mut deck = rest.clone();
    let mut score = 0.0;
    let mut me = [0u8; 7];
    let mut op = [0u8; 7];
    for _ in 0..samples {
        // partial shuffle of the first `need` cards
        let n = deck.len();
        for i in 0..need {
            let j = i + rng.below((n - i) as u64) as usize;
            deck.swap(i, j);
        }
        let mut full = [0u8; 5];
        full[..board.len()].copy_from_slice(board);
        full[board.len()..].copy_from_slice(&deck[2..2 + 5 - board.len()]);
        me[0] = hole[0];
        me[1] = hole[1];
        me[2..].copy_from_slice(&full);
        op[0] = deck[0];
        op[1] = deck[1];
        op[2..].copy_from_slice(&full);
        let (a, b) = (evaluate(&me), evaluate(&op));
        score += if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 };
    }
    score / samples as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chips(h: &Hand) -> u32 {
        h.stack[0] + h.stack[1] + h.total_in[0] + h.total_in[1] - if h.over() { h.total_in[0] + h.total_in[1] } else { 0 }
    }

    #[test]
    fn chips_are_conserved_in_random_play() {
        let mut rng = Rng::new(7);
        for g in 0..20000 {
            let stacks = [1 + rng.below(400) as u32, 1 + rng.below(400) as u32];
            let mut h = Hand::new(stacks, (g % 2) as usize, &mut rng);
            let mut steps = 0;
            while !h.over() {
                let l = h.legal();
                let a = match rng.below(4) {
                    0 if l.fold => Action::Fold,
                    1 if l.max_raise > 0 => Action::Raise(l.min_raise + rng.below((l.max_raise - l.min_raise + 1) as u64) as u32),
                    2 if l.check => Action::Check,
                    _ => if l.call > 0 { Action::Call } else { Action::Check },
                };
                h.act(a, &mut rng);
                steps += 1;
                assert!(steps < 200, "hand does not end");
            }
            assert_eq!(chips(&h), stacks[0] + stacks[1], "chips not conserved");
            let r = h.result.unwrap();
            assert_eq!(r[0] + r[1], 0);
            assert!(h.folded.is_some() || h.board.len() == 5);
        }
    }

    #[test]
    fn preflop_order_and_big_blind_option() {
        let mut rng = Rng::new(1);
        let mut h = Hand::new([200, 200], 0, &mut rng);
        assert_eq!(h.to_act, 0); // button / small blind first
        assert_eq!(h.legal().call, 1);
        h.act(Action::Call, &mut rng); // limp
        assert_eq!(h.street, 0);
        assert_eq!(h.to_act, 1); // big blind still has the option
        assert!(h.legal().check);
        h.act(Action::Check, &mut rng);
        assert_eq!(h.street, 1);
        assert_eq!(h.board.len(), 3);
        assert_eq!(h.to_act, 1); // big blind first after the flop
    }

    #[test]
    fn min_raise_rules() {
        let mut rng = Rng::new(2);
        let mut h = Hand::new([200, 200], 0, &mut rng);
        assert_eq!(h.legal().min_raise, 4); // to 2 BB
        h.act(Action::Raise(10), &mut rng); // raise by 8
        assert_eq!(h.legal().min_raise, 18); // re-raise must add at least 8 more
        h.act(Action::Raise(3), &mut rng); // too small: clamped up to the minimum
        assert_eq!(h.street_in[1], 18);
    }

    #[test]
    fn short_all_in_returns_uncalled_chips() {
        let mut rng = Rng::new(3);
        let mut h = Hand::new([50, 200], 0, &mut rng);
        h.act(Action::Raise(50), &mut rng); // all in for 50
        h.act(Action::Raise(200), &mut rng); // big blind "raises" all in: only 50 can be called
        assert!(!h.over() || h.board.len() == 5);
        while !h.over() {
            let l = h.legal();
            h.act(if l.call > 0 { Action::Call } else { Action::Check }, &mut rng);
        }
        assert_eq!(h.stack[0] + h.stack[1], 250);
        let r = h.result.unwrap();
        assert!(r[0].abs() <= 50 && r[1].abs() <= 50);
    }

    #[test]
    fn equity_sanity() {
        let mut rng = Rng::new(4);
        let aa = equity([48, 49], &[], 20000, &mut rng); // aces
        let low = equity([0, 21], &[], 20000, &mut rng); // 2c 7d
        assert!((aa - 0.85).abs() < 0.02, "AA {aa}");
        assert!((low - 0.35).abs() < 0.03, "72o {low}");
    }
}
