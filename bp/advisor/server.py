"""Phone advisor server: the phone page scans the BPAdvisor QR and posts it here; the answer is the
network's advice with a one-round look-ahead for every legal move.

  python -m bp.advisor.server            (listens on 0.0.0.0:5173, the cloudflared tunnel target)
Open on the phone:  https://<your-tunnel-host>/?k=<ADVISOR_TOKEN from .env>
"""
from __future__ import annotations

import base64
import json
import secrets
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from ..pvp.features import KEYS, PHASE_LEAD, PHASE_REPLACE, action_mask, encode
from ..pvp.model import masked
from ..pvp.probe import probe
from ..pvp.search import RustSearcher
from ..pvp.teambuilder import load_net
from ..sim import constants as C
from ..sim.data import game_data
from .state import build, decode

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent / "web"
FAMILY_RU = ["Гуманоид", "Дракон", "Летающий", "Нежить", "Зверёк", "Магический", "Элементаль", "Животное", "Водный", "Механизм"]


def token():
    env = ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    for line in lines:
        if line.startswith("ADVISOR_TOKEN="):
            return line.split("=", 1)[1].strip()
    t = secrets.token_urlsafe(12)
    with open(env, "a", encoding="utf-8") as f:
        f.write(("\n" if lines and lines[-1] else "") + f"ADVISOR_TOKEN={t}\n")
    return t


def ab_name(aid):
    a = game_data().abilities.get(aid, {})
    return a.get("name_ru") or a.get("name") or str(aid)


def pet_name(pet):
    return game_data().species[pet.spec.species]["name"]


def wasted(battle, side, slot, hints):
    """Why the ability would do nothing this round (the simulator's probe, bp/pvp/probe.py), or None.
    The network sometimes likes such moves out of habit (stun on a Critter)."""
    h = hints[slot]
    if not h[0] or h[1] > 0 or h[3] > 0 or h[2] > 0:
        return None
    if h[4]:
        foe = battle.teams[1 - side].active_pet
        return "зверьки невосприимчивы к контролю" if foe.family == C.CRITTER else "цель сейчас невосприимчива"
    if h[5]:
        return "цель недосягаема в этом раунде"
    return None


class Advisor:
    def __init__(self, ckpt):
        self.ckpt = Path(ckpt)
        self.lock = threading.Lock()
        self.mtime = 0
        self.load()

    def load(self):
        m = self.ckpt.stat().st_mtime
        if m != self.mtime:
            self.net = load_net(self.ckpt)
            self.search = RustSearcher(self.net)
            self.update = torch.load(self.ckpt, map_location="cpu", weights_only=False).get("update")
            self.mtime = m

    def label(self, b, side, phase, a):
        team = b.teams[side]
        if a < 3:
            pet = team.active_pet
            known = side == 0 or a in pet.revealed
            if known:
                return f"{a + 1} · {ab_name(pet.abilities[a])}"
            opts = game_data().species[pet.spec.species]["abilities"][a]
            return f"{a + 1} · {' / '.join(ab_name(x) for x in opts)}"
        if a < 6:
            p = team.pets[a - 3]
            verb = "выпустить" if phase in (PHASE_LEAD, PHASE_REPLACE) else "сменить на"
            return f"{verb} {pet_name(p)}"
        return "пропустить ход"

    def advise(self, data: bytes):
        t0 = time.time()
        st = decode(data)
        b, phase = build(st)
        if phase == PHASE_REPLACE and not b.needs_replacement(0) and not (st["flags"] & 1):
            return {"wait": "Ждём, пока соперник выберет питомца"}
        with self.lock:
            self.load()
            # 64 sampled worlds, 3 rounds ahead (Rust engine, ~0.15 s)
            res = self.search.evaluate(b, 0, phase, k=64, depth=3, seed=int(t0))
            o, m = encode(b, 0, phase), action_mask(b, 0, phase)
            with torch.no_grad():
                x = {k: torch.as_tensor(o[k][None], device="cuda") for k in KEYS}
                _, _, opp, lo = self.net(x)
                opp = torch.softmax(opp.float(), -1)[0].cpu().numpy()
                lo = torch.sigmoid(lo.float())[0].cpu().numpy()
        acts = []
        for a, (q, se, n) in res["q"].items():
            acts.append({"a": a, "label": self.label(b, 0, phase, a), "win": (q + 1) / 2, "prior": float(res["prior"][a])})
        # the value head is often sure (0% / 100%): then the network's own preference breaks the tie
        # rank like the training target: prior * exp(Q / tau). Q alone is noisy (sampled dice, hidden
        # abilities, an over-confident value head) and let never-played swaps / passes win by luck
        tau = 0.15
        lg = {x["a"]: np.log(max(x["prior"], 1e-6)) + (2 * x["win"] - 1) / tau for x in acts}
        mx = max(lg.values())
        z = sum(np.exp(v - mx) for v in lg.values())
        for x in acts:
            x["score"] = float(np.exp(lg[x["a"]] - mx) / z)
        # control into immunity (stun on a Critter): only works if the opponent swaps the target out
        # (swaps resolve before abilities), so its weight is the predicted swap chance, not habit
        p_swap = 0.0
        if phase not in (PHASE_LEAD, PHASE_REPLACE):
            om = action_mask(b, 1, phase)
            ps = opp * om
            if ps.sum() > 0:
                p_swap = float(ps[3:6].sum() / ps.sum())
        hints = probe(b, 0) if phase not in (PHASE_LEAD, PHASE_REPLACE) else None
        for x in acts:
            if x["a"] < 3 and hints is not None:
                why = wasted(b, 0, x["a"], hints)
                if why:
                    x["score"] *= p_swap
                    x["label"] += (f" — по нему не подействует ({why}); сработает, только если соперник сменит питомца"
                                   f" (≈{p_swap:.0%})" if p_swap >= 0.05 else f" — не подействует: {why}")
        tot = sum(x["score"] for x in acts) or 1.0
        for x in acts:
            x["score"] /= tot
        acts.sort(key=lambda x: -x["score"])
        spread = max(x["win"] for x in acts) - min(x["win"] for x in acts)
        note = None
        if spread < 0.02 and len(acts) > 1:
            v = (res["value"] + 1) / 2
            note = ("Модель считает позицию почти проигранной" if v < 0.1 else
                    "Модель считает позицию почти выигранной" if v > 0.9 else "Варианты почти равны") +                    " — порядок по её предпочтению"
        # what the opponent will probably do (only for real rounds)
        opp_list = []
        if phase not in (PHASE_LEAD, PHASE_REPLACE):
            om = action_mask(b, 1, phase)
            ps = opp * om
            if ps.sum() > 0:
                ps = ps / ps.sum()
                for a in np.argsort(-ps)[:3]:
                    if ps[a] > 0.03:
                        opp_list.append({"label": self.label(b, 1, phase, int(a)), "p": float(ps[a])})
        hidden = []
        gd = game_data()
        for pi, pet in enumerate(b.teams[1].pets):
            opts = gd.species[pet.spec.species]["abilities"]
            for s in range(3):
                if s not in pet.revealed and len(opts[s]) >= 2:
                    p2 = float(lo[pi * 3 + s])
                    hidden.append({"pet": pet_name(pet), "slot": s + 1, "a": ab_name(opts[s][0]), "b": ab_name(opts[s][1]), "pb": p2})
        me, en = b.teams[0].active_pet, b.teams[1].active_pet
        return {
            "phase": {PHASE_LEAD: "выбор первого питомца", PHASE_REPLACE: "замена питомца"}.get(phase, f"раунд {b.round + 1}"),
            "value": (res["value"] + 1) / 2, "actions": acts, "opp": opp_list, "hidden": hidden,
            "me": f"{pet_name(me)} ({FAMILY_RU[me.family]}) {me.hp}/{me.max_hp}",
            "enemy": f"{pet_name(en)} ({FAMILY_RU[en.family]}) {en.hp}/{en.max_hp}",
            "model": self.update, "ms": round((time.time() - t0) * 1000), "note": note,
        }


class Reranker:
    """Dashboard exclusion list: save data/pvp/rank_exclude.json, then redraw the dashboard at once
    (lists filtered) and recompute the teams without the excluded pets in the background."""
    path = ROOT / "data" / "pvp" / "rank_exclude.json"

    def __init__(self):
        self.lock = threading.Lock()
        self.busy = False
        self.again = False

    def items(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get("items", []) if self.path.exists() else []
        except Exception:
            return []

    def change(self, add=None, remove=None):
        """One entry at a time, so two open dashboards don't overwrite each other's list."""
        key = lambda e: (int(e["species"]), e.get("breed") or None)
        items = self.items()
        if remove:
            items = [e for e in items if key(e) != key(remove)]
        if add and key(add) not in {key(e) for e in items}:
            items.append(add)
        self.save(items)
        return self.items()

    def save(self, items):
        clean = [{"species": int(e["species"]), "name": str(e.get("name", ""))[:80],
                  "breed": (str(e["breed"])[:4] if e.get("breed") else None)} for e in items][:300]
        self.path.write_text(json.dumps({"items": clean}, ensure_ascii=False, indent=1), encoding="utf-8")
        with self.lock:
            if self.busy:
                self.again = True
                return
            self.busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        py, log = sys.executable, open(ROOT / "logs" / "pvp_rankings.log", "w")
        try:
            while True:
                subprocess.run([py, "-m", "bp.pvp.dashboard"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                subprocess.run([py, "-m", "bp.pvp.rankings", "--teams-only", "--ckpt", str(ROOT / "checkpoints" / "pvp" / "stage2.pt")],
                               cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                subprocess.run([py, "-m", "bp.pvp.dashboard"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                with self.lock:
                    if not self.again:
                        self.busy = False
                        return
                    self.again = False
        except Exception:
            traceback.print_exc()
            with self.lock:
                self.busy = False


def make_handler(adv, tok):
    rr = Reranker()

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")  # the dashboard may be opened as a local file
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            if path.startswith("/poker"):
                from ..poker import server as poker
                if not poker.handle_get(self, path):
                    self._send(404, {"error": "not found"})
            elif path in ("/", "/index.html"):
                self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/dashboard" and f"k={tok}" in query.split("&"):
                self._send(200, (ROOT / "logs" / "pvp_dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/exclude" and self.headers.get("X-Token") == tok:
                self._send(200, {"items": rr.items(), "busy": rr.busy})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.startswith("/poker/api/"):  # heads-up hold'em vs the bot (bp/poker)
                if self.headers.get("X-Token") != tok:
                    return self._send(403, {"error": "нужен ключ: откройте ссылку с ?k=..."})
                try:
                    from ..poker import server as poker
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    return self._send(200, poker.handle_post(self, self.path.split("?")[0], body))
                except Exception as ex:
                    traceback.print_exc()
                    return self._send(400, {"error": str(ex)[:200]})
            if self.path.startswith("/api/exclude"):
                if self.headers.get("X-Token") != tok:
                    return self._send(403, {"error": "нет ключа"})
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n))
                    items = rr.change(body.get("add"), body.get("remove"))
                    return self._send(200, {"ok": True, "items": items})
                except Exception as ex:
                    return self._send(400, {"error": str(ex)[:200]})
            if not self.path.startswith("/api/advise"):
                return self._send(404, {"error": "not found"})
            if self.headers.get("X-Token") != tok:
                return self._send(403, {"error": "нужен ключ: откройте ссылку с ?k=..."})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                data = base64.b64decode(body["qr"])
                res = adv.advise(data)
                with open(ROOT / "logs" / "advisor_requests.jsonl", "a", encoding="utf-8") as f:  # for later analysis
                    f.write(json.dumps({"t": time.time(), "qr": body["qr"], "best": (res.get("actions") or [{}])[0].get("a"),
                                        "value": res.get("value")}) + "\n")
                self._send(200, res)
            except Exception as ex:
                traceback.print_exc()
                self._send(400, {"error": str(ex)[:200]})
    return H


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "pvp" / "actor.pt"))
    ap.add_argument("--port", type=int, default=5173)
    args = ap.parse_args()
    tok = token()
    adv = Advisor(args.ckpt)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(adv, tok))
    print(f"advisor on :{args.port}, model update {adv.update}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
