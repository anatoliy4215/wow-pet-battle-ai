"""Game data for the Rust simulator (rust/bpsim): one JSON with everything the engine reads.

  python -m bp.sim.export_rust        -> data/processed/sim_rust.json

Effect targets (targets.plan) and chain rows are resolved here, so the Rust side does not need the
wow-petsim rule files.
"""
from __future__ import annotations

import json

from .data import PROCESSED, game_data
from .targets import chain_rows, plan

OUT = PROCESSED / "sim_rust.json"


def export(path=OUT):
    gd = game_data()
    chains = chain_rows()
    abilities = {}
    for aid, a in gd.abilities.items():
        abilities[aid] = {
            "name": a["name"], "family": a["family"], "cooldown": a["cooldown"],
            "states": [[int(k), int(v)] for k, v in a["states"].items()],
            "turns": [{"type": t["type"], "event": t["event"],
                       "effects": [{"op": e["op"], "aura": e["aura"] or 0,
                                    "params": {k: int(v) for k, v in e["params"].items()}}
                                   for e in t["effects"]]} for t in a["turns"]],
            "plan": [[ti, ei, code] for (ti, ei), code in sorted(plan(aid, 0).items())],
            "chain": sorted([ti, ei] for (x, ti, ei) in chains if x == aid),
        }
    species = {}
    for sid, s in gd.species.items():
        species[sid] = {"name": s["name"] or str(sid), "family": s["family"],
                        "abilities": s["abilities"], "states": [[int(k), int(v)] for k, v in s["states"].items()]}
    path.write_text(json.dumps({"abilities": abilities, "species": species}), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(export())
