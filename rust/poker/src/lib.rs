//! bppoker: heads-up no-limit hold'em engine (eval, game, bots). Python module when built with the
//! "python" feature (default):
//!   import bppoker
//!   h = bppoker.Hand(seed, [200, 200], button=0)
//!   h.legal(); h.act("raise", 10); h.bot_act(); h.view(seat)

pub mod bot;
pub mod eval;
pub mod game;

#[cfg(feature = "python")]
mod py {
    use crate::eval::{card_str, category, evaluate, parse_card, CATEGORY_NAMES};
    use crate::game::{equity, Action, Hand, Rng};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    fn parse_action(kind: &str, to: u32) -> PyResult<Action> {
        Ok(match kind {
            "fold" => Action::Fold,
            "check" => Action::Check,
            "call" => Action::Call,
            "raise" | "bet" => Action::Raise(to),
            _ => return Err(PyValueError::new_err(format!("unknown action {kind}"))),
        })
    }

    fn action_pair(a: Action) -> (String, u32) {
        match a {
            Action::Fold => ("fold".into(), 0),
            Action::Check => ("check".into(), 0),
            Action::Call => ("call".into(), 0),
            Action::Raise(t) => ("raise".into(), t),
        }
    }

    #[pyclass(name = "Hand", module = "bppoker")]
    struct PyHand {
        h: Hand,
        rng: Rng,
    }

    #[pymethods]
    impl PyHand {
        #[new]
        #[pyo3(signature = (seed, stacks, button=0))]
        fn new(seed: u64, stacks: [u32; 2], button: usize) -> Self {
            let mut rng = Rng::new(seed);
            let h = Hand::new(stacks, button, &mut rng);
            PyHand { h, rng }
        }

        #[getter]
        fn over(&self) -> bool {
            self.h.over()
        }
        #[getter]
        fn to_act(&self) -> usize {
            self.h.to_act
        }
        #[getter]
        fn stacks(&self) -> [u32; 2] {
            self.h.stack
        }
        #[getter]
        fn result(&self) -> Option<[i64; 2]> {
            self.h.result
        }

        fn legal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let l = self.h.legal();
            let d = PyDict::new(py);
            d.set_item("fold", l.fold)?;
            d.set_item("check", l.check)?;
            d.set_item("call", l.call)?;
            d.set_item("min_raise", l.min_raise)?;
            d.set_item("max_raise", l.max_raise)?;
            Ok(d)
        }

        #[pyo3(signature = (kind, to=0))]
        fn act(&mut self, kind: &str, to: u32) -> PyResult<()> {
            let a = parse_action(kind, to)?;
            self.h.act(a, &mut self.rng);
            Ok(())
        }

        /// Bot v0 plays the seat to act. -> (action, raise_to, equity)
        #[pyo3(signature = (samples=3000))]
        fn bot_act(&mut self, samples: u32) -> (String, u32, f64) {
            let d = crate::bot::act(&self.h, samples, &mut self.rng);
            self.h.act(d.action, &mut self.rng);
            let (k, t) = action_pair(d.action);
            (k, t, d.equity)
        }

        /// What `seat` can see: own cards, board, chips, whose turn; the other hand only at showdown.
        fn view<'py>(&self, py: Python<'py>, seat: usize) -> PyResult<Bound<'py, PyDict>> {
            let h = &self.h;
            let d = PyDict::new(py);
            let showdown = h.over() && h.folded.is_none();
            let cards = |s: usize| -> Vec<String> { h.hole[s].iter().map(|&c| card_str(c)).collect() };
            d.set_item("hole", cards(seat))?;
            d.set_item("opp_hole", if showdown { Some(cards(1 - seat)) } else { None })?;
            d.set_item("board", h.board.iter().map(|&c| card_str(c)).collect::<Vec<_>>())?;
            d.set_item("stacks", h.stack)?;
            d.set_item("street_in", h.street_in)?;
            d.set_item("pot", h.pot())?;
            d.set_item("street", h.street)?;
            d.set_item("button", h.button)?;
            d.set_item("to_act", h.to_act)?;
            d.set_item("over", h.over())?;
            d.set_item("result", h.result)?;
            d.set_item("folded", h.folded)?;
            let hist: Vec<(usize, usize, String, u32)> = h.history.iter().map(|&(st, s, a)| {
                let (k, t) = action_pair(a);
                (st, s, k, t)
            }).collect();
            d.set_item("history", hist)?;
            if h.board.len() >= 3 || showdown {
                let name = |s: usize| CATEGORY_NAMES[category(h.value(s)) as usize];
                d.set_item("hand_name", name(seat))?;
                if showdown {
                    d.set_item("opp_hand_name", name(1 - seat))?;
                }
            }
            Ok(d)
        }
    }

    /// Best 5-card value of 5..7 cards given as strings ("As", "Td", ...).
    #[pyfunction]
    fn evaluate_cards(cards: Vec<String>) -> PyResult<u32> {
        let c: Option<Vec<u8>> = cards.iter().map(|s| parse_card(s)).collect();
        let c = c.ok_or_else(|| PyValueError::new_err("bad card"))?;
        Ok(evaluate(&c))
    }

    #[pyfunction]
    #[pyo3(signature = (hole, board, samples=10000, seed=1))]
    fn hand_equity(hole: Vec<String>, board: Vec<String>, samples: u32, seed: u64) -> PyResult<f64> {
        let h: Option<Vec<u8>> = hole.iter().map(|s| parse_card(s)).collect();
        let b: Option<Vec<u8>> = board.iter().map(|s| parse_card(s)).collect();
        let (h, b) = (h.ok_or_else(|| PyValueError::new_err("bad card"))?, b.ok_or_else(|| PyValueError::new_err("bad card"))?);
        Ok(equity([h[0], h[1]], &b, samples, &mut Rng::new(seed)))
    }

    #[pymodule]
    fn bppoker(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyHand>()?;
        m.add_function(wrap_pyfunction!(evaluate_cards, m)?)?;
        m.add_function(wrap_pyfunction!(hand_equity, m)?)?;
        Ok(())
    }
}
