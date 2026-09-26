"""Rented stage 2 actors on vast.ai: start boxes, keep them fed with fresh weights, collect their samples.

  python -m bp.cloud.fleet start --boxes 3 [--gpu "RTX 5060 Ti"] [--max-dph 0.2]
  python -m bp.cloud.fleet run            # loop: push weights, pull shards, replace dead boxes, watch the budget
  python -m bp.cloud.fleet run --target 6 --until 12    # also rents boxes itself (best fast-CPU cores per $)
                                                        # and replaces idle / slow ones
  python -m bp.cloud.fleet status
  python -m bp.cloud.fleet stop           # destroy every fleet box

Safety: `run` destroys everything when the credit drops below --reserve, at --until, or when
logs/STOP_FLEET appears; every box also has a hard stop time built into its actor command.
Health checks in `run` (a box that costs money but brings nothing is destroyed and its host banned):
  no shard 25 min after start, no new shard for 20 min, or samples per $ below 45% of the best box.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import threading
import tarfile
import time
from pathlib import Path

import torch

from . import vast
from .run_bench import IMAGE, bundle

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "logs" / "fleet.json"
SHARDS = ROOT / "data" / "pvp" / "shards"
WEIGHTS = ROOT / "checkpoints" / "pvp" / "actor.pt"
LABEL = "bp-fleet"
AVOID = ("CN", "China")
# Ampere or newer: the image's PyTorch (CUDA 12.8) and bf16 inference need it
GPU_OK = re.compile(r"RTX (30|40|50)\d\d|RTX A\d000|^A(10|40|4000|5000|6000)\b|^L4|RTX [456]000 Ada")
STOP_FILE = ROOT / "logs" / "STOP_FLEET"
BUILD_RUST = ("set -e; export DEBIAN_FRONTEND=noninteractive; "
              "(command -v cc >/dev/null && command -v curl >/dev/null) || "
              "(apt-get update -qq && apt-get install -y -qq build-essential curl >/dev/null); "
              "[ -x $HOME/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal; "
              "export PATH=$HOME/.cargo/bin:$PATH; PY=$(command -v python); "
              "$PY -m pip install -q maturin 2>/dev/null || $PY -m pip install -q --break-system-packages maturin; "
              "cd /root/bp/rust/bpsim && $PY -m maturin build --release -q -i $PY 2>&1 | tail -3; "
              "($PY -m pip install -q --force-reinstall --no-deps target/wheels/*.whl 2>/dev/null || "
              "$PY -m pip install -q --break-system-packages --force-reinstall --no-deps target/wheels/*.whl) && "
              "cd /root/bp && python -c 'import bpsim; print(\"bpsim ok\")'")
ACTOR_STOP = ROOT / "logs" / "STOP_ACTORS"


def cpu_speed(name):
    """Rough single-core speed of a host CPU relative to a current desktop CPU (bench: Xeon E5 v4 is
    ~0.47 of a Ryzen 7800X3D in our simulator). Our actors are CPU-bound, so this decides the value."""
    c = (name or "").lower()
    if re.search(r"ryzen.*\b(7|9)\d\d\d|i[3579]-1[234]\d\d\d|ultra [579]|threadripper (pro )?7\d|epyc [49]\d\d4", c):
        return 1.0
    if re.search(r"ryzen.*\b5\d\d\d|i[3579]-1[01]\d\d\d|threadripper|epyc 7\d\d3|epyc 9\d\d\d", c):
        return 0.75
    if re.search(r"epyc|xeon.*(gold|platinum|silver)|xeon w", c):
        return 0.55
    return 0.45


def offer_score(o):
    """Expected samples per dollar. Since the actor runs on the Rust engine (bp.pvp.actor_rs) the
    network on the GPU is the bottleneck: vast.ai's DL performance score per $/h; a few cores are
    enough (checked by min_cores)."""
    return (o.get("dlperf") or 0) / max(0.01, o["dph_total"])


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"boxes": {}}


def save_state(st):
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def slim(src=WEIGHTS):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: v.half() if v.is_floating_point() else v for k, v in ck["model"].items()}
    out = ROOT / "logs" / "fleet_weights_fp16.pt"
    torch.save({"model": sd, "cfg": ck["cfg"], "update": ck.get("update")}, out)
    return out, ck.get("update")


def offers(gpu, max_dph, min_cores=8, ranked=False):
    q = {"reliability2": {"gte": 0.97}, "cuda_max_good": {"gte": 12.8}, "inet_down": {"gte": 300 if ranked else 500},
         "disk_space": {"gte": 30}, "dph_total": {"lte": max_dph}, "num_gpus": {"eq": 1},
         "cpu_cores_effective": {"gte": min_cores}}
    if gpu:
        q["gpu_name"] = {"in": gpu.split(",")}
    out = [o for o in vast.search(q) if not any(a in (o.get("geolocation") or "") for a in AVOID)]
    if ranked:  # best fast-CPU cores per dollar, only GPUs our image runs on
        return sorted([o for o in out if GPU_OK.search(o.get("gpu_name") or "")], key=lambda o: -offer_score(o))
    return sorted(out, key=lambda o: o["dph_total"])


def start_box(offer, files, hours):
    iid = vast.create(offer["id"], IMAGE, disk=30, label=LABEL).get("new_contract")
    t0 = time.time()
    inst = None
    while time.time() - t0 < 420:
        i = vast.instance(iid)
        if i and i.get("actual_status") == "running" and i.get("ssh_host"):
            inst = i
            break
        if i and ("error" in (i.get("status_msg") or "").lower() or i.get("actual_status") in ("exited", "offline")):
            break
        time.sleep(10)
    if inst is None:
        vast.destroy(iid)
        raise RuntimeError(f"box {iid} did not start")
    try:
        return _setup_box(iid, inst, offer, files, hours)
    except Exception:
        vast.destroy(iid)  # a box we cannot use must not keep costing money (ssh refused, upload failed...)
        raise


def _setup_box(iid, inst, offer, files, hours):
    vast.connect(inst)
    for f in files:
        vast.scp_to(inst, f, "/root/")
    workers = max(2, min(40, int(offer.get("cpu_cores_effective") or 4) - 1))
    vast.ssh(inst, "mkdir -p /root/bp /root/shards && cd /root/bp && tar xzf /root/bp_bundle.tar.gz && "
                   "cp /root/fleet_weights_fp16.pt /root/w.pt && echo ok", timeout=300)
    # the Rust engine (bpsim): build it on the box (toolchain + wheel, ~2-4 min)
    vast.ssh(inst, BUILD_RUST, timeout=1200)
    vast.ssh(inst, "cd /root/bp && "
                   f"(nohup timeout {int(hours * 3600)} python -m bp.pvp.actor_rs --weights /root/w.pt --out /root/shards "
                   f"--threads {workers} > /root/actor.log 2>&1 &) ; sleep 1; echo started")
    return iid, inst["_ssh"], workers


def pull(iid, ssh_ep):
    """Copy finished shards of one box into data/pvp/shards/<box>/ and delete them there."""
    inst = {"_ssh": tuple(ssh_ep)}
    out = vast.ssh(inst, "cd /root/shards && ls *.npz 2>/dev/null | grep -v tmp > /tmp/list; "
                         "n=$(wc -l < /tmp/list); if [ $n -gt 0 ]; then tar cf /tmp/pull.tar -T /tmp/list; fi; echo $n",
                   timeout=300)
    n = int(out.strip().splitlines()[-1] or 0)
    if not n:
        return 0
    dest = SHARDS / f"box{iid}"
    dest.mkdir(parents=True, exist_ok=True)
    local = ROOT / "logs" / f"pull_{iid}.tar"
    vast.scp_from(inst, "/tmp/pull.tar", local, timeout=900)
    with tarfile.open(local) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".npz") and "/" not in m.name:
                tar.extract(m, dest, filter="data")
    local.unlink()
    vast.ssh(inst, "cd /root/shards && xargs rm -f < /tmp/list; rm -f /tmp/pull.tar", timeout=120)
    return n


def push(iid, ssh_ep, wfile):
    inst = {"_ssh": tuple(ssh_ep)}
    vast.scp_to(inst, wfile, "/root/w_new.pt")
    vast.ssh(inst, "mv /root/w_new.pt /root/w.pt", timeout=60)


def cmd_start(a):
    st = load_state()
    files = [bundle(), slim()[0]]
    cands = offers(a.gpu, a.max_dph)
    started = 0
    for o in cands:
        if started >= a.boxes:
            break
        try:
            iid, ep, workers = start_box(o, files, a.hours)
        except Exception as ex:
            print("skip offer", o["id"], str(ex)[:120], flush=True)
            continue
        st["boxes"][str(iid)] = {"gpu": o["gpu_name"], "dph": o["dph_total"], "where": o.get("geolocation"),
                                 "ssh": list(ep), "workers": workers, "started": time.time(), "pulled": 0}
        save_state(st)
        started += 1
        print(f"started box {iid}: {o['gpu_name']} ${o['dph_total']:.3f}/h {o.get('geolocation')} {workers} workers", flush=True)


def cmd_redeploy(a):
    """Upload the current code to every box and restart its actor (weights stay)."""
    st = load_state()
    path = bundle()
    for iid, b in st["boxes"].items():
        inst = {"_ssh": tuple(b["ssh"])}
        try:
            vast.scp_to(inst, path, "/root/")
            left = max(0.5, a.hours - (time.time() - b["started"]) / 3600)
            # kill and start in separate sessions: a pkill pattern must not match its own shell's command line
            vast.ssh(inst, "pkill -f '[b]p[.]pvp[.]actor'; sleep 3; echo killed", timeout=60, check=False)
            vast.ssh(inst, "cd /root/bp && tar xzf /root/bp_bundle.tar.gz && echo ok", timeout=120)
            vast.ssh(inst, BUILD_RUST, timeout=1200)
            vast.ssh(inst, "cd /root/bp && "
                           f"(nohup timeout {int(left * 3600)} python -m bp.pvp.actor_rs --weights /root/w.pt --out /root/shards "
                           f"--threads {b['workers']} > /root/actor.log 2>&1 &) ; sleep 1; echo restarted", timeout=120)
            print("redeployed", iid, flush=True)
        except Exception as ex:
            print("redeploy failed", iid, str(ex)[:150], flush=True)


def cmd_stop(_a=None):
    for i in vast.instances():
        if i.get("label") == LABEL:
            vast.destroy(i["id"])
            print("destroyed", i["id"], flush=True)
    st = load_state()
    st["boxes"] = {}
    save_state(st)


def cmd_status(_a=None):
    st = load_state()
    live = {i["id"]: i for i in vast.instances() if i.get("label") == LABEL}
    for iid, b in st["boxes"].items():
        i = live.get(int(iid), {})
        print(iid, b["gpu"], b["where"], f"${b['dph']:.3f}/h", i.get("actual_status", "gone"), "pulled", b.get("pulled", 0))
    print(vast.balance())


def _box_alive(b):
    """(actor running?, last lines of its log) over ssh."""
    inst = {"_ssh": tuple(b["ssh"])}
    out = vast.ssh(inst, "pgrep -f '[b]p[.]pvp[.]actor' >/dev/null && echo RUNNING || echo DEAD; tail -n 4 /root/actor.log",
                   timeout=60, check=False)
    return "RUNNING" in out, out.strip()[-400:]


def cmd_run_safe(a):
    """cmd_run that survives errors (2026-09-26: one vast.ai API timeout ended the loop at 10:50 and
    the boxes ran unattended for an hour); deadline and start time are fixed once."""
    a.t_end = time.time() + a.until * 3600 if a.until else None
    a.t_start = time.time()
    STOP_FILE.unlink(missing_ok=True)
    while True:
        try:
            return cmd_run(a)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print("fleet loop error, going on in 60 s:", str(ex)[:200], flush=True)
            time.sleep(60)


def cmd_run(a):
    last_w = 0.0
    t_end = a.t_end
    starting = {}  # offer id -> $/h of boxes being rented in the background
    started_q = queue.Queue()
    t_start = a.t_start

    def starter(o, files, hours):
        try:
            iid, ep, workers = start_box(o, files, hours)
            started_q.put((o, iid, ep, workers, None))
        except Exception as ex:
            started_q.put((o, None, None, None, str(ex)[:150]))

    while True:
        st = load_state()
        st.setdefault("banned", [])
        now = time.time()
        bal = vast.balance()
        credit = bal["credit"]
        if credit is not None and credit < a.reserve:
            print(f"credit {credit:.2f} below reserve {a.reserve}: stopping the fleet", flush=True)
            cmd_stop()
            return
        learner_done = ACTOR_STOP.exists() and ACTOR_STOP.stat().st_mtime > t_start  # learner finished
        if (t_end and now > t_end) or STOP_FILE.exists() or learner_done:
            print("deadline / stop file: stopping the fleet", flush=True)
            cmd_stop()
            return
        # boxes whose start finished in the background
        while not started_q.empty():
            o, iid, ep, workers, err = started_q.get()
            starting.pop(o["id"], None)
            if err:
                print("start failed", o["id"], o.get("cpu_name"), err, flush=True)
                continue
            st["boxes"][str(iid)] = {"gpu": o["gpu_name"], "dph": o["dph_total"], "where": o.get("geolocation"),
                                     "cpu": o.get("cpu_name"), "machine": o.get("machine_id"), "ssh": list(ep),
                                     "workers": workers, "started": time.time(), "pulled": 0}
            print(f"started box {iid}: {o.get('cpu_name')} x{workers} | {o['gpu_name']} ${o['dph_total']:.3f}/h "
                  f"{o.get('geolocation')}", flush=True)
        live = {i["id"]: i for i in vast.instances() if i.get("label") == LABEL}
        wfile = None
        if WEIGHTS.exists() and WEIGHTS.stat().st_mtime > last_w:
            wfile, upd = slim()
            last_w = WEIGHTS.stat().st_mtime
            print(f"new weights (update {upd})", flush=True)

        def drop(iid, why):
            b = st["boxes"].pop(iid)
            print(f"box {iid} ({b.get('cpu')}, {b['gpu']}): {why} -> destroyed", flush=True)
            try:
                vast.destroy(int(iid))
            except Exception as ex:
                print("destroy failed", iid, ex, flush=True)

        for iid, b in list(st["boxes"].items()):
            i = live.get(int(iid))
            if i is None or i.get("actual_status") != "running":
                if now - b["started"] > 120:  # a just-started box may not be listed as running yet
                    drop(iid, f"gone ({None if i is None else i.get('actual_status')})")
                continue
            try:
                n = pull(iid, b["ssh"])
                b["pulled"] = b.get("pulled", 0) + n
                if n:
                    b.setdefault("first_shard", now)
                    b["last_shard"] = now
                if wfile:
                    push(iid, b["ssh"], wfile)
                b["errors"] = 0
            except (subprocess.SubprocessError, RuntimeError, OSError) as ex:
                b["errors"] = b.get("errors", 0) + 1
                print(f"box {iid}: {str(ex)[:120]}", flush=True)
                if b["errors"] >= 5:
                    drop(iid, "unreachable 5 times in a row")
                    continue
            age = now - b["started"]
            if not b.get("pulled") and age > 25 * 60:
                alive, tail = _box_alive(b) if b.get("errors", 0) == 0 else (False, "")
                drop(iid, f"no samples after {age / 60:.0f} min (actor {'running' if alive else 'dead'}) {tail!r}")
            elif b.get("last_shard") and now - b["last_shard"] > 20 * 60:
                drop(iid, f"no new samples for {(now - b['last_shard']) / 60:.0f} min")
        # samples per dollar: replace boxes far behind the best measured one
        rated = {}
        for iid, b in st["boxes"].items():
            if b.get("first_shard") and now - b["started"] > 50 * 60:
                rated[iid] = b["pulled"] * 4096 / max(1, now - b["started"]) / b["dph"]
        if len(rated) >= 2:
            best = max(rated.values())
            for iid, r in rated.items():
                if r < 0.45 * best:
                    drop(iid, f"slow: {r * 3600 / 1e6:.2f}M samples/$ vs best {best * 3600 / 1e6:.2f}M")
        for iid, b in st["boxes"].items():
            b["per_dollar_M"] = round(b["pulled"] * 4096 / max(1, now - b["started"]) / b["dph"] * 3600 / 1e6, 2)
        total_dph = sum(b["dph"] for b in st["boxes"].values())
        # rent more while under the target and the budget lasts until the end
        if a.target and credit is not None and len(st["boxes"]) + len(starting) < a.target and len(starting) < 2:
            hours_left = max(0.5, ((t_end - now) / 3600) if t_end else a.hours)
            afford = (credit - a.reserve) / hours_left
            used = {b.get("machine") for b in st["boxes"].values()}
            banned = set(st["banned"]) | used
            try:
                cands = [o for o in offers(None, a.max_dph, min_cores=a.min_cores, ranked=True)
                         if o.get("machine_id") not in banned]
            except Exception as ex:
                print("offer search failed", str(ex)[:120], flush=True)
                cands = []
            for o in cands:
                if total_dph + sum(starting.values()) + o["dph_total"] <= afford:
                    files = [bundle(), slim()[0]]
                    starting[o["id"]] = o["dph_total"]
                    st["banned"].append(o.get("machine_id"))  # never rent the same host twice
                    threading.Thread(target=starter, args=(o, files, hours_left), daemon=True).start()
                    print(f"renting {o.get('cpu_name')} x{o.get('cpu_cores_effective')} | {o['gpu_name']} "
                          f"${o['dph_total']:.3f}/h {o.get('geolocation')} (score {offer_score(o):.0f}, "
                          f"budget ${afford:.2f}/h)", flush=True)
                    break
        save_state(st)
        print(time.strftime("%H:%M"), f"{len(st['boxes'])} boxes (+{len(starting)} starting), ${total_dph:.2f}/h, "
              f"credit ${credit:.2f}", " ".join(f"{k}:{b['per_dollar_M']}M/$" for k, b in st["boxes"].items()), flush=True)
        time.sleep(a.every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "run", "status", "stop", "redeploy"])
    ap.add_argument("--boxes", type=int, default=2)
    ap.add_argument("--gpu", default="RTX 5060 Ti")
    ap.add_argument("--max-dph", type=float, default=0.2)
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--reserve", type=float, default=0.3, help="destroy everything below this credit ($)")
    ap.add_argument("--every", type=int, default=180)
    ap.add_argument("--target", type=int, default=0, help="run: keep this many boxes (rents / replaces them itself)")
    ap.add_argument("--until", type=float, default=0, help="run: stop everything after this many hours")
    ap.add_argument("--min-cores", type=int, default=6)
    a = ap.parse_args()
    {"start": cmd_start, "run": cmd_run_safe, "status": cmd_status, "stop": cmd_stop, "redeploy": cmd_redeploy}[a.cmd](a)


if __name__ == "__main__":
    main()
