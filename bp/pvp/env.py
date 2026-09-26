"""Two-player PvP environment.

Phases:
  lead    - both players see both teams and pick their first pet at the same time (if lead_select)
  round   - both choose simultaneously: ability 0-2, swap 3-5, pass 6
  replace - a player whose active pet died picks the next one (the other player waits, then sees it)

The env never decides for a player: pending() says which sides must act, act() takes their actions.
"""
from __future__ import annotations

import random

from ..sim.battle import Battle
from .features import PHASE_LEAD, PHASE_REPLACE, PHASE_ROUND, action_mask, encode, loadout_truth


class PvPEnv:
    def __init__(self, sampler, seed=None, lead_select=True, hide=True, max_rounds=100):
        self.sampler = sampler
        self.rng = random.Random(seed)
        self.lead_select = lead_select
        self.hide = hide
        self.max_rounds = max_rounds
        self.battle = None

    def reset(self, teams=None):
        a, b = teams if teams else (self.sampler.team(), self.sampler.team())
        self.battle = Battle(a, b, seed=self.rng.getrandbits(32))
        self.phase = PHASE_LEAD if self.lead_select else PHASE_ROUND
        self.done = False
        self.last_actions = {}
        self._settle()

    def _settle(self):
        b = self.battle
        if self.phase == PHASE_LEAD:
            return
        if b.over or b.round >= self.max_rounds:
            self.done = True
            return
        self.phase = PHASE_REPLACE if (b.needs_replacement(0) or b.needs_replacement(1)) else PHASE_ROUND

    def pending(self):
        if self.done:
            return []
        if self.phase == PHASE_REPLACE:
            return [s for s in (0, 1) if self.battle.needs_replacement(s)]
        return [0, 1]

    def observe(self, side):
        return encode(self.battle, side, self.phase, self.hide), action_mask(self.battle, side, self.phase)

    def act(self, actions):
        """actions: {side: action} for every pending side."""
        b = self.battle
        self.last_actions = dict(actions)
        if self.phase == PHASE_LEAD:
            for side in (0, 1):
                idx = actions[side] - 3
                if 0 <= idx < len(b.teams[side].pets):
                    b.teams[side].active = idx
            self.phase = PHASE_ROUND
        else:
            full = []
            for side in (0, 1):
                a = actions.get(side, 6)
                legal = b.legal_actions(side)
                if a not in legal:
                    a = legal[0] if b.needs_replacement(side) else 6
                full.append(a)
            b.step(full[0], full[1])
        self._settle()

    @property
    def winner(self):
        w = self.battle.winner
        return w if w in (0, 1) else -1

    def loadout_truth(self, side):
        return loadout_truth(self.battle, side)
