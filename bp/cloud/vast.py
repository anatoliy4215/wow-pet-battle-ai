"""Minimal vast.ai client (REST API): search offers, rent / inspect / destroy instances, run commands
and copy files over SSH.  The API key is read from .env (VAST_AI_KEY) and never printed.

  python -m bp.cloud.vast balance
  python -m bp.cloud.vast instances
  python -m bp.cloud.vast destroy-all          (stops the spending: removes every instance)
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
API = "https://console.vast.ai/api/v0"
SSH_KEY = Path.home() / ".ssh" / "vastai_ed25519"


def _key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("VAST_AI_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("VAST_AI_KEY not found in .env")


_lock = threading.Lock()
_last_call = [0.0]


def api(method, path, retries=8, **kw):
    """One API call; calls are spaced out (the API allows ~3 per second) and 429s are retried."""
    h = {"Authorization": f"Bearer {_key()}", "Accept": "application/json"}
    for attempt in range(retries):
        with _lock:
            wait = _last_call[0] + 0.5 - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        try:
            r = requests.request(method, API + path, headers=h, timeout=60, **kw)
        except requests.exceptions.RequestException as ex:  # network hiccup / read timeout: try again
            if attempt == retries - 1:
                raise RuntimeError(f"{method} {path}: {ex}") from ex
            time.sleep(5 + 5 * attempt)
            continue
        if r.status_code == 429:
            try:
                ra = float(r.json().get("retry_after", 2))
            except Exception:
                ra = 2.0
            time.sleep(ra + 0.5 * attempt)
            continue
        if not r.ok:
            raise RuntimeError(f"{method} {path}: {r.status_code} {r.text[:300]}")
        return r.json() if r.text else {}
    raise RuntimeError(f"{method} {path}: rate limited")


def balance():
    u = api("GET", "/users/current/")
    return {"credit": u.get("credit"), "balance": u.get("balance"), "total_spend": u.get("total_spend")}


def ssh_keys():
    return api("GET", "/ssh/")


def add_ssh_key(pub):
    return api("POST", "/ssh/", json={"ssh_key": pub})


def search(q):
    base = {"rentable": {"eq": True}, "rented": {"eq": False}, "verified": {"eq": True}, "external": {"eq": False},
            "type": "on-demand", "limit": 1000, "order": [["dph_total", "asc"]]}
    base.update(q)
    return api("POST", "/bundles/", json=base)["offers"]


def create(offer_id, image, disk=20, label="bp", onstart=None, price=None):
    body = {"client_id": "me", "image": image, "disk": disk, "label": label, "runtype": "ssh_proxy ssh_direc ssh",
            "onstart": onstart or ""}
    if price is not None:
        body["price"] = price  # interruptible (bid) instance
    return api("PUT", f"/asks/{offer_id}/", json=body)


def instances():
    return api("GET", "/instances/").get("instances", [])


def instance(iid):
    for i in instances():
        if i["id"] == iid:
            return i
    return None


def destroy(iid):
    return api("DELETE", f"/instances/{iid}/")


def ssh_candidates(inst):
    """Possible (host, port) pairs: the direct port mapping first, then the vast.ai ssh proxy."""
    out = []
    ports = (inst.get("ports") or {}).get("22/tcp")
    if inst.get("public_ipaddr") and ports:
        out.append((inst["public_ipaddr"].strip(), int(ports[0]["HostPort"])))
    if inst.get("ssh_host") and inst.get("ssh_port"):
        out.append((inst["ssh_host"], int(inst["ssh_port"])))
    return out


def ssh_target(inst):
    if inst.get("_ssh"):
        return inst["_ssh"]
    c = ssh_candidates(inst)
    return c[0]


def connect(inst, tries=18, wait=10):
    """Find an ssh endpoint that answers (sshd can need a minute after 'running'); remembers it."""
    for _ in range(tries):
        for host, port in ssh_candidates(inst):
            r = subprocess.run(["ssh", "-i", str(SSH_KEY), "-p", str(port), *SSH_OPTS, f"root@{host}", "echo ok"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and "ok" in r.stdout:
                inst["_ssh"] = (host, port)
                return inst["_ssh"]
        time.sleep(wait)
    raise RuntimeError("no ssh endpoint answers")


SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]


def ssh(inst, cmd, timeout=3600, check=True):
    host, port = ssh_target(inst)
    r = subprocess.run(["ssh", "-i", str(SSH_KEY), "-p", str(port), *SSH_OPTS, f"root@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr[-500:]}")
    return r.stdout


def scp_to(inst, local, remote, timeout=1800):
    host, port = ssh_target(inst)
    subprocess.run(["scp", "-i", str(SSH_KEY), "-P", str(port), *SSH_OPTS, str(local), f"root@{host}:{remote}"],
                   check=True, capture_output=True, timeout=timeout)


def scp_from(inst, remote, local, timeout=1800):
    host, port = ssh_target(inst)
    subprocess.run(["scp", "-i", str(SSH_KEY), "-P", str(port), *SSH_OPTS, f"root@{host}:{remote}", str(local)],
                   check=True, capture_output=True, timeout=timeout)


def wait_running(iid, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        inst = instance(iid)
        if inst and inst.get("actual_status") == "running" and inst.get("ssh_host"):
            return inst
        if inst and inst.get("actual_status") in ("exited", "offline"):
            raise RuntimeError(f"instance {iid} status {inst.get('actual_status')}: {inst.get('status_msg')}")
        time.sleep(10)
    raise TimeoutError(f"instance {iid} not running after {timeout}s")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "balance"
    if cmd == "balance":
        print(json.dumps(balance()))
    elif cmd == "instances":
        for i in instances():
            print(i["id"], i.get("gpu_name"), i.get("num_gpus"), i.get("actual_status"), f"${i.get('dph_total', 0):.3f}/h",
                  i.get("geolocation"), i.get("label"))
    elif cmd == "destroy-all":
        for i in instances():
            destroy(i["id"])
            print("destroyed", i["id"])


if __name__ == "__main__":
    main()
