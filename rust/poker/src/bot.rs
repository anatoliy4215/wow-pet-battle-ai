//! Bot v0: equity against a random hand vs the price of the pot, value bets, some bluffs.
//! A simple, honest baseline (and a sparring partner) before the CFR bot.

use crate::game::{equity, Action, Hand, Rng, BB};

pub struct Decision {
    pub action: Action,
    pub equity: f64,
}

pub fn act(h: &Hand, samples: u32, rng: &mut Rng) -> Decision {
    let me = h.to_act;
    let opp = 1 - me;
    let l = h.legal();
    let e = equity(h.hole[me], &h.board, samples, rng);
    let pot = h.pot();
    let to_call = l.call;
    let raise_to = |frac: f64| -> u32 {
        // a raise of `frac` of the pot after calling
        let after_call = pot + to_call;
        let size = (after_call as f64 * frac).round() as u32;
        let to = h.street_in[opp] + size.max(BB);
        to.clamp(l.min_raise, l.max_raise)
    };
    let r = rng.unit();
    let action = if to_call > 0 {
        // facing a bet: the bettor's range is stronger than a random hand, so demand a margin
        let price = to_call as f64 / (pot + to_call) as f64;
        let need = price + 0.08 + if h.street == 0 { 0.0 } else { 0.05 };
        if e > 0.78 && l.max_raise > 0 && r < 0.7 {
            Action::Raise(raise_to(0.8))
        } else if e > need {
            Action::Call
        } else if l.max_raise > 0 && r < 0.04 {
            Action::Raise(raise_to(1.0)) // occasional bluff raise
        } else {
            Action::Fold
        }
    } else {
        // nobody bet yet: value bet strong hands, bluff a little with weak ones
        if e > 0.68 && l.max_raise > 0 && r < 0.8 {
            Action::Raise(raise_to(if e > 0.85 { 0.9 } else { 0.66 }))
        } else if e < 0.35 && l.max_raise > 0 && r < 0.15 {
            Action::Raise(raise_to(0.5))
        } else {
            Action::Check
        }
    };
    Decision { action, equity: e }
}
