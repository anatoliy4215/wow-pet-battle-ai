"""Second vast.ai test: run the real stage 2 workload (bp.pvp.actor) for a few minutes on several box types
and measure look-ahead samples per second and per dollar.

  python -m bp.cloud.actor_test [--minutes 3]      -> logs/vast_actor_test.json
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import torch

from . import vast
from .run_bench import COMMON, IMAGE, POLL, bundle

ROOT = Path(__file__).resolve().parents[2]

TARGETS = [
    ("RTX 5060 Ti", {"gpu_name": {"eq": "RTX 5060 Ti"}, "num_gpus": {"eq": 1}}),
    ("RTX 4070S / 4070 Ti", {"gpu_name": {"in": ["RTX 4070S", "RTX 4070 Ti", "RTX 4070S Ti"]}, "num_gpus": {"eq": 1}}),
    ("RTX 3060", {"gpu_name": {"eq": "RTX 3060"}, "num_gpus": {"eq": 1}, "cpu_cores_effective": {"gte": 8}}),
    ("RTX 3090", {"gpu_name": {"eq": "RTX 3090"}, "num_gpus": {"eq": 1}, "cpu_cores_effective": {"gte": 10}}),
    ("RTX 4090", {"gpu_name": {"eq": "RTX 4090"}, "num_gpus": {"eq": 1}, "cpu_cores_effective": {"gte": 12}}),
    ("RTX 5090", {"gpu_name": {"eq": "RTX 5090"}, "num_gpus": {"eq": 1}, "cpu_cores_effective": {"gte": 12}}),
    ("RTX 5070 Ti", {"gpu_name": {"eq": "RTX 5070 Ti"}, "num_gpus": {"eq": 1}}),
]
AVOID = ("CN", "China")  # docker hub pulls there crawl (seen in the first test)


def slim_weights():
    ck = torch.load(ROOT / "checkpoints" / "pvp" / "main.pt", map_location="cpu", weights_only=False)
    sd = {k: v.half() if v.is_floating_point() else v for k, v in ck["model"].items()}
    path = ROOT / "logs" / "actor_weights_fp16.pt"
    torch.save({"model": sd, "cfg": ck["cfg"], "update": ck["update"]}, path)
    return path


def pick():
    chosen = []
    for label, q in TARGETS:
        f = dict(COMMON)
        f.update(q)
        f["inet_down"] = {"gte": 500}
        offers = [o for o in vast.search(f) if not any(a in (o.get("geolocation") or "") for a in AVOID)]
        offers.sort(key=lambda o: o["dph_total"])
        if offers:
            chosen.append((label, offers[:6]))
    return chosen


def run_one(label, offers, files, minutes, results, lock):
    rec, iid, t0 = {"label": label}, None, time.time()
    try:
        for offer in offers:
            rec = {"label": label, "gpu": offer["gpu_name"], "dph": offer["dph_total"], "where": offer.get("geolocation"),
                   "cores_listed": offer.get("cpu_cores_effective"), "cpu": offer.get("cpu_name")}
            try:
                iid = vast.create(offer["id"], IMAGE, disk=24, label="bp-actor-test").get("new_contract")
            except RuntimeError as ex:
                if "no_such_ask" in str(ex):
                    continue
                raise
            # give each host 6 minutes to start; a broken one ("GPU error", stuck pull) is replaced
            inst, t_c = None, time.time()
            while time.time() - t_c < 360:
                i = POLL.get(iid)
                if i and i.get("actual_status") == "running" and i.get("ssh_host"):
                    inst = i
                    break
                if i and ("error" in (i.get("status_msg") or "").lower() or i.get("actual_status") in ("exited", "offline")):
                    break
                time.sleep(5)
            if inst is None:
                rec["skipped"] = (rec.get("skipped") or []) + [offer["id"]]
                vast.destroy(iid)
                iid = None
                continue
            break
        if iid is None:
            raise RuntimeError("no host started")
        rec["ready_s"] = round(time.time() - t0)
        vast.connect(inst)
        for f in files:
            vast.scp_to(inst, f, "/root/")
        workers = max(2, min(24, int(rec["cores_listed"] or 4) - 1))
        cmd = ("mkdir -p /root/bp /root/shards && cd /root/bp && tar xzf /root/bp_bundle.tar.gz && "
               f"timeout {int(minutes * 60 + 60)} python -m bp.pvp.actor --weights /root/actor_weights_fp16.pt "
               f"--out /root/shards --workers {workers} --envs 24 --shard 512 --hours {minutes / 60:.4f} > /root/actor.log 2>&1; "
               "python -c \"import glob, numpy as np; fs=glob.glob('/root/shards/*.npz'); "
               "print('SAMPLES', sum(len(np.load(f)['z']) for f in fs))\"; tail -3 /root/actor.log; nproc")
        t1 = time.time()
        out = vast.ssh(inst, cmd, timeout=int(minutes * 60 + 400))
        run_s = time.time() - t1
        for line in out.splitlines():
            if line.startswith("SAMPLES"):
                rec["samples"] = int(line.split()[1])
        rec["workers"] = workers
        rec["run_s"] = round(run_s)
        rec["log"] = out[-300:]
        if rec.get("samples"):
            rate = rec["samples"] / (minutes * 60)
            rec["samples_per_s"] = round(rate, 1)
            rec["samples_per_dollar_M"] = round(rate * 3600 / rec["dph"] / 1e6, 2)
    except Exception as ex:
        rec["error"] = str(ex)[-300:]
    finally:
        if iid:
            try:
                vast.destroy(iid)
            except Exception as ex:
                rec["destroy_error"] = str(ex)[:200]
        rec["total_s"] = round(time.time() - t0)
        with lock:
            results.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=3)
    args = ap.parse_args()
    before = vast.balance()
    files = [bundle(), slim_weights()]
    chosen = pick()
    for label, offers in chosen:
        o = offers[0]
        print(label, "->", o["gpu_name"], round(o["dph_total"], 3), o.get("geolocation"), o.get("cpu_cores_effective"), o.get("cpu_name"), flush=True)
    results, lock, threads = [], threading.Lock(), []
    try:
        for label, offers in chosen:
            t = threading.Thread(target=run_one, args=(label, offers, files, args.minutes, results, lock), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(3)
        for t in threads:
            t.join(timeout=3600)
    finally:
        for i in vast.instances():
            if i.get("label") == "bp-actor-test":
                try:
                    vast.destroy(i["id"])
                    print("safety destroy", i["id"], flush=True)
                except Exception:
                    pass
    after = vast.balance()
    (ROOT / "logs" / "vast_actor_test.json").write_text(json.dumps({"before": before, "after": after, "results": results},
                                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print("credit before", before["credit"], "after", after["credit"], flush=True)


if __name__ == "__main__":
    main()
