"""The friend's GPU (Docker container bp-train-node, reached over Tailscale) as the remote learner.

  this PC:   bp.pvp.actor_rs plays the games on the whole local GPU  ->  data/pvp/shards/local
  the node:  bp.pvp.learner --quiet trains on them, publishes actor.pt
  sync:      shards go up, fresh weights come down (and a stage2.pt backup), exam + benchmark run here

  python -m bp.cloud.node deploy          # upload the code (no container restart), keep the job running
  python -m bp.cloud.node start [--init]  # (re)start training there; --init first uploads our stage2.pt
  python -m bp.cloud.node sync            # loop: push shards, pull weights / backups, evaluate
  python -m bp.cloud.node status | stop | logs

The node container only has SSH + PyTorch; everything else lives in its /workspace volume. After a
host reboot the container runs /workspace/autostart.sh, which resumes training by itself.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = os.environ.get("BP_NODE_HOST", "")  # the node's address (e.g. its Tailscale IP)
PORT = int(os.environ.get("BP_NODE_PORT", "2222"))
KEY = Path.home() / ".ssh" / "bp_node_ed25519"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
LOCAL = ROOT / "data" / "pvp" / "shards" / "local"
CKPT = ROOT / "checkpoints" / "pvp"
EVALS = ROOT / "logs" / "remote_evals.json"
LEARNER = ("cd /workspace/bp && exec python -m bp.pvp.learner --resume --quiet --out /workspace/ckpt/stage2.pt "
           "--weights /workspace/ckpt/actor.pt --shards /workspace/shards --buffer 2000000 --reload 450")


def _need_host():
    if not HOST:
        raise RuntimeError("set BP_NODE_HOST to the training node's address")


def ssh(cmd, timeout=600, check=True):
    _need_host()
    r = subprocess.run(["ssh", "-i", str(KEY), "-p", str(PORT), *SSH_OPTS, f"root@{HOST}", cmd],
                       capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr[-400:]}")
    return r.stdout


def scp_to(local, remote, timeout=1800):
    _need_host()
    subprocess.run(["scp", "-i", str(KEY), "-P", str(PORT), *SSH_OPTS, str(local), f"root@{HOST}:{remote}"],
                   check=True, capture_output=True, timeout=timeout)


def scp_from(remote, local, timeout=1800):
    _need_host()
    subprocess.run(["scp", "-i", str(KEY), "-P", str(PORT), *SSH_OPTS, f"root@{HOST}:{remote}", str(local)],
                   check=True, capture_output=True, timeout=timeout)


def deploy():
    """New code into /workspace/bp; a running learner is restarted on the new code."""
    from .run_bench import bundle
    path = bundle()
    ssh("mkdir -p /workspace/bp/logs /workspace/ckpt /workspace/shards/home")
    scp_to(path, "/workspace/bp_bundle.tar.gz")
    ssh("cd /workspace/bp && tar xzf /workspace/bp_bundle.tar.gz && rm /workspace/bp_bundle.tar.gz && echo ok")
    running = "RUNNING" in ssh("pgrep -f '[b]p[.]pvp[.]learner' >/dev/null && echo RUNNING || true")
    if "YES" in ssh("[ -f /workspace/autostart.sh ] && echo YES || true"):
        write_autostart()  # the learner command may have changed with the code
    if running:
        restart()
    print("code deployed" + (", learner restarted" if running else ""), flush=True)


def write_autostart():
    """The job the container (re)starts: the learner in a loop (a crash or reboot resumes it)."""
    script = ("#!/bin/bash\n# training job of bp-train-node (written by bp.cloud.node)\n"
              "while [ ! -f /workspace/STOP ]; do\n"
              f"  bash -c \"{LEARNER}\" >> /workspace/learner.log 2>&1\n"
              "  sleep 20\ndone\n")
    tmp = ROOT / "logs" / "node_autostart.sh"
    tmp.write_bytes(script.encode())  # LF line endings
    scp_to(tmp, "/workspace/autostart.sh")
    ssh("chmod +x /workspace/autostart.sh")


def kill_job():
    # its own session: a pkill pattern must not find the shell that also starts the job
    ssh("pkill -f '[a]utostart[.]sh'; pkill -f '[b]p[.]pvp[.]learner'; sleep 3; echo killed", check=False)


def restart():
    kill_job()
    ssh("rm -f /workspace/STOP; (nohup /workspace/autostart.sh >> /workspace/autostart.log 2>&1 &) ; sleep 1; "
        "echo started", check=False)


def start(init=False):
    ssh("mkdir -p /workspace/ckpt /workspace/shards/home")
    have = "YES" in ssh("[ -f /workspace/ckpt/stage2.pt ] && echo YES || true")
    if init or not have:
        print("uploading stage2.pt ...", flush=True)
        scp_to(CKPT / "stage2.pt", "/workspace/ckpt/stage2.tmp")
        ssh("mv /workspace/ckpt/stage2.tmp /workspace/ckpt/stage2.pt")
    write_autostart()
    restart()
    print("training started on the node", flush=True)


def stop():
    ssh("touch /workspace/STOP", check=False)
    kill_job()


def status():
    print(ssh("pgrep -af '[b]p[.]pvp[.]learner' | cut -c1-120; tail -n 3 /workspace/learner.log; "
              "ls /workspace/shards/home | wc -l; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw "
              "--format=csv,noheader; df -h /workspace | tail -1", check=False))


def push_shards():
    """Finished shards of the local actor -> the node (then removed here)."""
    files = [p for p in sorted(LOCAL.glob("*.npz")) if not p.name.endswith(".tmp.npz")]
    if not files:
        return 0
    tar_path = ROOT / "logs" / "node_push.tar"
    with tarfile.open(tar_path, "w") as tar:
        for p in files:
            tar.add(p, arcname=p.name)
    scp_to(tar_path, "/workspace/shards/push.tar")
    # extract under a temp name first: the learner must never read a half-written shard
    ssh("mkdir -p /workspace/incoming /workspace/shards/home && cd /workspace/shards && "
        "tar xf push.tar -C /workspace/incoming && rm push.tar && "
        "for f in /workspace/incoming/*.npz; do mv \"$f\" home/; done")
    for p in files:
        p.unlink(missing_ok=True)
    tar_path.unlink(missing_ok=True)
    return len(files)


def remote_mtime(path):
    out = ssh(f"stat -c %Y {path} 2>/dev/null || echo 0", check=False).strip()
    try:
        return int(out.splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def pull(remote, local):
    tmp = Path(str(local) + ".part")
    scp_from(remote, tmp)
    tmp.replace(local)


def evaluate(bench_pairs, suite):
    """Benchmark + exam of the current weights on this PC (the node runs --quiet)."""
    import numpy as np
    import torch
    from ..pvp import bench, tactics
    from ..pvp.teambuilder import load_net
    net = load_net(CKPT / "actor.pt")
    upd = torch.load(CKPT / "actor.pt", map_location="cpu", weights_only=False).get("update")
    row = {"upd": upd, "t": time.time()}
    row["bench_greedy"], _ = bench.play(net, "greedy", bench_pairs, seed=1, record=None)
    for k, v in tactics.evaluate(net, suite).items():
        if k != "n":
            row[f"tac_{k}"] = v
    rows = json.loads(EVALS.read_text(encoding="utf-8")) if EVALS.exists() else []
    rows.append(row)
    EVALS.write_text(json.dumps(rows), encoding="utf-8")
    del net
    torch.cuda.empty_cache()
    _ = np
    return row


def sync(every=30, backup_every=1200, eval_every=1800):
    last_w = last_backup = last_eval = 0.0
    bench_pairs = suite = None
    while True:
        t0 = time.time()
        try:
            n = push_shards()
            m = remote_mtime("/workspace/ckpt/actor.pt")
            if m and m != last_w:
                pull("/workspace/ckpt/actor.pt", CKPT / "actor.pt")
                last_w = m
            if time.time() - last_backup > backup_every and remote_mtime("/workspace/ckpt/stage2.pt"):
                pull("/workspace/ckpt/stage2.pt", CKPT / "stage2.pt")  # backup, and the dashboard history
                last_backup = time.time()
                try:
                    subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "bp.pvp.dashboard"],
                                   cwd=str(ROOT), capture_output=True, timeout=600)
                except Exception as ex:
                    print("dashboard:", ex, flush=True)
            if time.time() - last_eval > eval_every:
                if bench_pairs is None:
                    from ..pvp import bench, tactics
                    bench_pairs, suite = bench.make_pairs(), tactics.build_suite(n=300)
                row = evaluate(bench_pairs, suite)
                last_eval = time.time()
                print(time.strftime("%H:%M"), "eval", {k: round(v, 3) if isinstance(v, float) else v
                                                       for k, v in row.items() if k != "t"}, flush=True)
            tail = ssh("tail -n 1 /workspace/learner.log", check=False).strip()[-140:]
            print(time.strftime("%H:%M"), f"pushed {n} shards |", tail, flush=True)
        except Exception as ex:  # node offline / Tailscale down: keep the shards here, try again
            print(time.strftime("%H:%M"), "sync error:", str(ex)[:200], flush=True)
        time.sleep(max(1, every - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["deploy", "start", "stop", "sync", "status", "logs"])
    ap.add_argument("--init", action="store_true", help="start: upload our stage2.pt first")
    a = ap.parse_args()
    if a.cmd == "deploy":
        deploy()
    elif a.cmd == "start":
        start(a.init)
    elif a.cmd == "stop":
        stop()
    elif a.cmd == "sync":
        sync()
    elif a.cmd == "status":
        status()
    elif a.cmd == "logs":
        print(ssh("tail -n 40 /workspace/learner.log", check=False))
    _ = shutil


if __name__ == "__main__":
    main()
