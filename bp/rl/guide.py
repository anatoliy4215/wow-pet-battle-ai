"""Write a human battle guide from a rule plan, using the pets' exact ability ids.

The rule controller (rules.py) falls back to the first usable ability in slot order when nothing in
the priority list applies; the guide states that fallback explicitly so every round is covered.
"""
from __future__ import annotations

from ..sim.data import game_data


def ability_line(aid, cond_text=None):
    ab = game_data().abilities[aid]
    cd = ab["cooldown"]
    s = f"**{ab['name']}**"
    extra = []
    if cd:
        extra.append(f"перезарядка {cd}")
    if cond_text:
        extra.append(cond_text)
    return s + (f" ({', '.join(extra)})" if extra else "")


def pet_section(spec, priorities, conds=None):
    """priorities: ability names in order; conds: {ability name: human condition text}."""
    gd = game_data()
    conds = conds or {}
    by_name = {gd.abilities[a]["name"]: a for a in spec.abilities if a}
    lines = []
    for i, name in enumerate(priorities, 1):
        aid = by_name[name]
        cd = gd.abilities[aid]["cooldown"]
        when = conds.get(name)
        if cd and when:
            cond = f"если доступна и {when}"
        elif cd:
            cond = "если доступна"
        elif when:
            cond = f"если {when}"
        else:
            cond = None
        lines.append(f"{i}. {ability_line(aid)}" + (f" — {cond}" if cond else ""))
    # explicit fallback: first usable in slot order (what the simulator did)
    fallback = next((gd.abilities[a]["name"] for a in spec.abilities
                     if a and not gd.abilities[a]["cooldown"]), None)
    if fallback:
        lines.append(f"Если ничего из этого нельзя — **{fallback}**.")
    return "\n".join(lines)
