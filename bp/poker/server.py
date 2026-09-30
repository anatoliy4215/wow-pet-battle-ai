"""Poker vs the bot, for the phone: a heads-up no-limit hold'em match run on this PC (bppoker, Rust).

Served by the advisor server (same tunnel): https://<your-tunnel-host>/poker?k=<ADVISOR_TOKEN>
The server deals and keeps every hand, the page only shows the table and sends the player's moves,
so the bot's cards never reach the browser before the showdown.

API (POST, header X-Token):
  /poker/api/new    {}                      -> table
  /poker/api/state  {id}                    -> table
  /poker/api/act    {id, kind, to}          -> table after the player's move and the bot's replies
  /poker/api/next   {id}                    -> table with a new hand
"""
from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
STACK = 200  # 100 big blinds
HUMAN, BOT = 0, 1
BOT_NAME = "Бот v0"


class Table:
    def __init__(self):
        import bppoker
        self.bppoker = bppoker
        self.seed = secrets.randbits(48)
        self.hands = 0
        self.net = 0  # human chips won over the match
        self.rebuys = 0
        self.stacks = [STACK, STACK]
        self.button = secrets.randbits(1)
        self.events = []  # bot moves since the player's last move (for the page to animate)
        self.t = time.time()
        self.new_hand()

    def new_hand(self):
        if min(self.stacks) <= 0:  # someone is broke: both start again
            self.stacks = [STACK, STACK]
            self.rebuys += 1
        self.button = 1 - self.button
        self.hands += 1
        self.hand = self.bppoker.Hand(self.seed + self.hands, list(self.stacks), self.button)
        self.events = []
        self.run_bot()

    def run_bot(self):
        while not self.hand.over and self.hand.to_act == BOT:
            kind, to, eq = self.hand.bot_act(3000)
            self.events.append({"kind": kind, "to": to, "equity": round(eq, 3)})
        if self.hand.over and self.hand.result is not None and not getattr(self, "_counted", False):
            self.net += self.hand.result[HUMAN]
            self.stacks = list(self.hand.stacks)
            self._counted = True

    def act(self, kind, to):
        if self.hand.over or self.hand.to_act != HUMAN:
            return
        self.events = []
        self.hand.act(kind, int(to or 0))
        self.run_bot()

    def next(self):
        if self.hand.over:
            self._counted = False
            self.new_hand()

    def view(self):
        v = self.hand.view(HUMAN)
        v["legal"] = self.hand.legal() if (not self.hand.over and self.hand.to_act == HUMAN) else None
        v["events"] = self.events
        v["match"] = {"hands": self.hands, "net": self.net, "bb_per_100": round(self.net / 2 / max(1, self.hands) * 100, 1),
                      "rebuys": self.rebuys, "bot": BOT_NAME}
        return v


class Tables:
    def __init__(self):
        self.lock = threading.Lock()
        self.tables = {}

    def new(self):
        with self.lock:
            tid = secrets.token_urlsafe(8)
            self.tables[tid] = Table()
            # forget idle tables (a day without moves)
            for k in [k for k, t in self.tables.items() if time.time() - t.t > 86400]:
                del self.tables[k]
            return tid, self.tables[tid]

    def get(self, tid):
        with self.lock:
            t = self.tables.get(tid)
            if t is not None:
                t.t = time.time()
            return t


TABLES = Tables()


def handle_get(h, path):
    """GET /poker -> the page (the key is checked by the API calls it makes)."""
    if path in ("/poker", "/poker/"):
        h._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        return True
    return False


def handle_post(h, path, body):
    if path == "/poker/api/new":
        tid, t = TABLES.new()
        return {"id": tid, **t.view()}
    t = TABLES.get(body.get("id", ""))
    if t is None:
        return {"error": "стол не найден", "gone": True}
    with TABLES.lock:
        if path == "/poker/api/act":
            t.act(body.get("kind", "check"), body.get("to", 0))
        elif path == "/poker/api/next":
            t.next()
        return {"id": body["id"], **t.view()}
