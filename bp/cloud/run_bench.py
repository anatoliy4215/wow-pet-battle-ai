"""Rent a few different vast.ai boxes at once, run bench_remote.py on each, destroy them.

  python -m bp.cloud.run_bench            -> logs/vast_bench.json + a table

Every instance is destroyed in a finally block (and on timeout), so a failure cannot leave it running.
"""
from __future__ import annotations

import io
import json
import tarfile
import threading
import time
from pathlib import Path

from . import vast

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"

TARGETS = [  # (label, search filter)
    ("RTX 3060", {"gpu_name": {"eq": "RTX 3060"}, "num_gpus": {"eq": 1}}),
    ("RTX 3060 Ti, много ядер", {"gpu_name": {"eq": "RTX 3060 Ti"}, "num_gpus": {"eq": 1}, "cpu_cores_effective": {"gte": 20}}),
    ("RTX 4070 / 4070S", {"gpu_name": {"in": ["RTX 4070", "RTX 4070S"]}, "num_gpus": {"eq": 1}}),
    ("RTX 3090", {"gpu_name": {"eq": "RTX 3090"}, "num_gpus": {"eq": 1}}),
    ("RTX 5060 Ti", {"gpu_name": {"eq": "RTX 5060 Ti"}, "num_gpus": {"eq": 1}}),
    ("RTX 5090", {"gpu_name": {"eq": "RTX 5090"}, "num_gpus": {"eq": 1}}),
    ("много ядер CPU", {"cpu_cores_effective": {"gte": 48}, "dph_total": {"lte": 0.45}}),
]
COMMON = {"reliability2": {"gte": 0.97}, "cuda_max_good": {"gte": 12.8}, "inet_down": {"gte": 300},
          "disk_space": {"gte": 30}, "dph_total": {"lte": 0.8}}


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in (ROOT / "bp").rglob("*.py"):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
        for pattern in ("data/processed/*.json", "data/raw/petsim/*.json", "data/pvp/*.json"):
            for p in ROOT.glob(pattern):
                tar.add(p, arcname=str(p.relative_to(ROOT)))
    path = ROOT / "logs" / "bp_bundle.tar.gz"
    path.write_bytes(buf.getvalue())
    return path


def pick_offers():
    chosen, regions = [], set()
    for label, q in TARGETS:
        f = dict(COMMON)
        f.update(q)
        offers = vast.search(f)
        if not offers:
            chosen.append((label, None))
            continue
        offers.sort(key=lambda o: o["dph_total"])
        best = offers[0]
        # prefer a region we have not tried yet if it costs at most 30% more
        for o in offers[:15]:
            reg = (o.get("geolocation") or "").split(",")[-1].strip()
            if reg not in regions and o["dph_total"] <= best["dph_total"] * 1.3:
                best = o
                break
        regions.add((best.get("geolocation") or "").split(",")[-1].strip())
        backups = [o for o in offers[:6] if o["id"] != best["id"]]
        chosen.append((label, [best] + backups))
    return chosen


class Poller:
    """One shared view of our instances, refreshed at most every 15 s (the API is rate limited)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.t = 0.0
        self.data = {}

    def get(self, iid):
        with self.lock:
            if time.time() - self.t > 15:
                try:
                    self.data = {i["id"]: i for i in vast.instances()}
                    self.t = time.time()
                except Exception:
                    pass
            return self.data.get(iid)


POLL = Poller()


def wait_running(iid, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        inst = POLL.get(iid)
        if inst and inst.get("actual_status") == "running" and inst.get("ssh_host"):
            return inst
        if inst and inst.get("actual_status") in ("exited", "offline"):
            raise RuntimeError(f"instance {iid} status {inst.get('actual_status')}: {inst.get('status_msg')}")
        time.sleep(5)
    raise TimeoutError(f"instance {iid} not running after {timeout}s")


def run_one(label, offers, bundle_path, results, lock):
    iid = None
    t0 = time.time()
    rec = {"label": label}
    try:
        for offer in offers:
            rec = {"label": label, "offer": offer["id"], "gpu": offer["gpu_name"], "n_gpu": offer["num_gpus"],
                   "dph": offer["dph_total"], "where": offer.get("geolocation"),
                   "cores_listed": offer.get("cpu_cores_effective"), "dlperf": offer.get("dlperf")}
            try:
                r = vast.create(offer["id"], IMAGE, disk=24, label="bp-bench")
                iid = r.get("new_contract")
                break
            except RuntimeError as ex:
                if "no_such_ask" not in str(ex):
                    raise
        if iid is None:
            raise RuntimeError("no offer could be rented")
        rec["instance"] = iid
        inst = wait_running(iid)
        rec["ready_s"] = round(time.time() - t0)
        # sshd may need a few more seconds after "running"
        for _ in range(12):
            try:
                vast.ssh(inst, "echo ok", timeout=60)
                break
            except Exception:
                time.sleep(10)
                inst = POLL.get(iid) or inst
        t1 = time.time()
        vast.scp_to(inst, bundle_path, "/root/bp_bundle.tar.gz")
        rec["upload_s"] = round(time.time() - t1, 1)
        out = vast.ssh(inst, "mkdir -p /root/bp && cd /root/bp && tar xzf /root/bp_bundle.tar.gz && "
                             "python -m bp.cloud.bench_remote --seconds 20 2>&1 | tail -3", timeout=900)
        line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
        if line:
            rec.update(json.loads(line[7:]))
        else:
            rec["error"] = out[-400:]
    except Exception as ex:
        rec["error"] = str(ex)[:400]
    finally:
        if iid:
            try:
                vast.destroy(iid)
                rec["destroyed"] = True
            except Exception as ex:
                rec["destroy_error"] = str(ex)[:200]
        rec["total_s"] = round(time.time() - t0)
        rec["cost_est"] = round(rec.get("dph", 0) * rec["total_s"] / 3600, 4)
        with lock:
            results.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)


def main():
    before = vast.balance()
    path = bundle()
    chosen = pick_offers()
    print("bundle", path.stat().st_size // 1024, "KB; offers:", flush=True)
    for label, o in chosen:
        o = o[0] if o else None
        print(" ", label, "->", None if o is None else (o["gpu_name"], o["num_gpus"], round(o["dph_total"], 3),
                                                        o.get("geolocation"), o.get("cpu_cores_effective")), flush=True)
    results, lock, threads = [], threading.Lock(), []
    try:
        for label, o in chosen:
            if o is None:
                continue
            t = threading.Thread(target=run_one, args=(label, o, path, results, lock), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(2)
        for t in threads:
            t.join(timeout=2400)
    finally:
        for i in vast.instances():  # safety net: nothing may keep running
            if i.get("label") == "bp-bench":
                try:
                    vast.destroy(i["id"])
                    print("safety destroy", i["id"], flush=True)
                except Exception:
                    pass
    after = vast.balance()
    out = {"before": before, "after": after, "image": IMAGE, "results": results}
    (ROOT / "logs" / "vast_bench.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("credit before", before["credit"], "after", after["credit"], flush=True)


if __name__ == "__main__":
    main()
