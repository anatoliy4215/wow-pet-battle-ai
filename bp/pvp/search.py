"""Look-ahead on top of the network ("what happens if I press X?").

For every legal action a of the player:
    Q(a) = average over k samples of the network's value after
           - hidden enemy abilities are sampled from the loadout head (we never peek at the real ones),
           - the opponent's reply is sampled from its predicted policy,
           - the round is simulated with fresh dice (hits, crits, damage rolls),
           - optionally `depth` more rounds are played by the network for both sides.
This catches what a one-shot policy can miss: an attack that would hit a dodge / an untargetable pet,
a kill that is available right now, a setup that pays off next round.  It is used for advice and
later as a policy-improvement target for training (expert iteration).
"""
from __future__ import annotations

import copy
import random
from functools import lru_cache

import numpy as np
import torch

from ..sim import constants as C
from ..sim import ops as OPS
from ..sim.clone import fast_clone
from ..sim.data import game_data
from .features import KEYS, PHASE_LEAD, PHASE_REPLACE, PHASE_ROUND, action_mask, encode
from .model import masked


@lru_cache(maxsize=1)
def _base_memo():
    gd = game_data()
    memo = {id(OPS): OPS, id(gd): gd}
    for d in list(gd.abilities.values()) + list(gd.species.values()):
        memo[id(d)] = d
    return memo


def clone(battle, seed=None):
    """Independent copy with fresh dice (bp.sim.clone: ~15x faster than deepcopy, same result)."""
    return fast_clone(battle, seed)


def determinize(battle, side, lo_prob, rng):
    """Replace the opponent's unrevealed ability choices with samples (lo_prob[9] = P(second option))."""
    gd = game_data()
    for pi, pet in enumerate(battle.teams[1 - side].pets):
        opts = gd.species[pet.spec.species]["abilities"]
        for sl in range(3):
            if sl in pet.revealed or len(opts[sl]) < 2:
                continue
            pet.abilities[sl] = opts[sl][1] if rng.random() < lo_prob[pi * 3 + sl] else opts[sl][0]


def _phase(b):
    return PHASE_REPLACE if (b.needs_replacement(0) or b.needs_replacement(1)) else PHASE_ROUND


def _pending(b):
    if b.over:
        return []
    if _phase(b) == PHASE_REPLACE:
        return [s for s in (0, 1) if b.needs_replacement(s)]
    return [0, 1]


class Searcher:
    def __init__(self, net, device="cuda"):
        self.net, self.device = net, device

    def _run(self, obs, masks):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            x = {k: torch.as_tensor(np.stack([o[k] for o in obs]), device=self.device) for k in KEYS}
            logits, v, opp, lo = self.net(x)
            p = torch.softmax(masked(logits, torch.as_tensor(np.stack(masks), device=self.device)), -1)
        return p.float().cpu().numpy(), v.float().cpu().numpy(), torch.sigmoid(lo.float()).cpu().numpy()

    def evaluate(self, battle, side, phase=PHASE_ROUND, k=16, depth=1, seed=0, max_rounds=100):
        """-> {"prior": [7], "value": v, "q": {action: (mean, stderr, n)}} from `side`'s point of view."""
        rng = random.Random(seed)
        o, m = encode(battle, side, phase), action_mask(battle, side, phase)
        prior, value, lo = self._run([o], [m])
        prior, value, lo = prior[0], float(value[0]), lo[0]
        legal = [a for a in range(C.N_ACTIONS) if m[a] > 0]
        # k determinized worlds, opponent reply sampled in each
        worlds = []
        for j in range(k):
            b = clone(battle, rng.getrandbits(32))
            determinize(b, side, lo, rng)
            worlds.append(b)
        opp_moves = [None] * k
        if phase != PHASE_REPLACE or battle.needs_replacement(1 - side):
            oo = [(encode(b, 1 - side, phase), action_mask(b, 1 - side, phase)) for b in worlds]
            po, _, _ = self._run([x[0] for x in oo], [x[1] for x in oo])
            opp_moves = [int(rng.choices(range(C.N_ACTIONS), po[j])[0]) for j in range(k)]
        # play every legal action in every world
        states, tags = [], []
        for a in legal:
            for j, w in enumerate(worlds):
                b = clone(w, rng.getrandbits(32))
                acts = {side: a}
                if opp_moves[j] is not None:
                    acts[1 - side] = opp_moves[j]
                _apply(b, phase, acts)
                states.append(b)
                tags.append(a)
        for _ in range(depth):
            self._advance(states, rng, max_rounds)
        vals = self._values(states, side)
        q = {}
        for a in legal:
            x = np.array([v for v, t in zip(vals, tags) if t == a])
            q[a] = (float(x.mean()), float(x.std() / np.sqrt(len(x))), len(x))
        return {"prior": prior, "value": value, "q": q, "loadout": lo}

    def _advance(self, states, rng, max_rounds):
        """One more decision point for every unfinished state, both sides sampling from the network."""
        req = [(b, s) for b in states if not b.over and b.round < max_rounds for s in _pending(b)]
        if not req:
            return
        obs = [(encode(b, s, _phase(b)), action_mask(b, s, _phase(b))) for b, s in req]
        p, _, _ = self._run([x[0] for x in obs], [x[1] for x in obs])
        acts = {}
        for (b, s), pi in zip(req, p):
            acts.setdefault(id(b), (b, {}))[1][s] = int(rng.choices(range(C.N_ACTIONS), pi)[0])
        for b, a in acts.values():
            _apply(b, _phase(b), a)

    def _values(self, states, side):
        out = np.zeros(len(states), dtype=np.float32)
        live = []
        for i, b in enumerate(states):
            if b.over:
                out[i] = 0.0 if b.winner not in (0, 1) else (1.0 if b.winner == side else -1.0)
            else:
                live.append(i)
        if live:
            obs = [encode(states[i], side, _phase(states[i])) for i in live]
            masks = [action_mask(states[i], side, _phase(states[i])) for i in live]
            _, v, _ = self._run(obs, masks)
            out[live] = v
        return out


def _apply(b, phase, acts):
    if phase == PHASE_LEAD:
        for s, a in acts.items():
            b.teams[s].active = a - 3
        return
    full = []
    for s in (0, 1):
        a = acts.get(s, C.A_PASS)
        legal = b.legal_actions(s)
        if a not in legal:
            a = legal[0] if b.needs_replacement(s) else C.A_PASS
        full.append(a)
    b.step(full[0], full[1])


class RustSearcher(Searcher):
    """The same look-ahead on the Rust engine (bpsim.Search). NOTE the depth: here depth = rounds played
    in the look-ahead (1 = only the round of the move), while Searcher's depth counts the extra rounds
    after it (Searcher depth=1 == RustSearcher depth=2)."""

    def _net(self, obs, mask):
        from .model import infer
        return infer(self.net, obs, mask, self.device)

    def evaluate(self, battle, side, phase=PHASE_ROUND, k=16, depth=2, seed=0, max_rounds=100, lo=None):
        from ..sim.rust import module, to_rust
        rb = to_rust(battle, seed)
        if lo is None:  # hidden enemy abilities sampled from the network's current guess
            o = rb.encode(side, phase)
            _, _, lo_ = self._net({k_: v[None] for k_, v in o.items()}, rb.action_mask(side, phase)[None])
            lo = [float(x) for x in lo_[0]]
        s = module().Search(rb, side, phase, k, depth, 0.2, seed, lo)
        while not s.done:
            obs, mask = s.requests()
            s.reply(*self._net(obs, mask))
        r = s.result()
        value = float(r["value"])
        return {"prior": np.array(r["prior"], dtype=np.float32), "value": value,
                "q": {int(a): ((float(m), float(se), int(n)) if n else (value, 0.0, 0)) for a, (m, se, n) in r["q"].items()},
                "loadout": np.array(r["lo"], dtype=np.float32)}
