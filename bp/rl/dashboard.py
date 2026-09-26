"""Training dashboard: a self-refreshing local HTML page written by the trainer.

Open logs/dashboard.html in a browser; it reloads every 30 s.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from ..sim import constants as C
from ..sim.data import game_data


def _fmt_dur(sec):
    sec = int(max(0, sec))
    h, m = divmod(sec // 60, 60)
    return f"{h} ч {m:02d} мин" if h else f"{m} мин"


def _chart(hist, key, w=760, h=220, color="#4f8cff", ymax=None):
    pts = [(x["upd"], x[key]) for x in hist if x.get(key) == x.get(key)]  # drop NaN
    if len(pts) < 2:
        return "<p class=muted>Мало данных для графика</p>"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y1 = ymax if ymax is not None else max(ys) * 1.1 or 1
    pad = 36

    def X(v):
        return pad + (v - x0) / max(1, x1 - x0) * (w - pad - 10)

    def Y(v):
        return h - 24 - v / y1 * (h - 40)

    path = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
    grid = ""
    for k in range(5):
        v = y1 * k / 4
        grid += (f'<line x1="{pad}" x2="{w - 10}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class=grid />'
                 f'<text x="{pad - 6}" y="{Y(v) + 4:.1f}" text-anchor=end class=axis>'
                 f'{v * 100:.0f}%</text>' if ymax == 1 else
                 f'<line x1="{pad}" x2="{w - 10}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class=grid />'
                 f'<text x="{pad - 6}" y="{Y(v) + 4:.1f}" text-anchor=end class=axis>{v:.0f}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%">{grid}'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" />'
            f'<text x="{pad}" y="{h - 6}" class=axis>шаг {x0}</text>'
            f'<text x="{w - 10}" y="{h - 6}" text-anchor=end class=axis>шаг {x1}</text></svg>')


def render(path: Path, stats: dict):
    gd = game_data()
    hist = stats["history"]
    last = hist[-1] if hist else {}
    done, total = last.get("upd", 0), stats["total_updates"]
    elapsed = time.time() - stats["start"]
    eta = elapsed / max(1, done - stats.get("start_upd", 0)) * (total - done) if done else 0
    rows = []
    for key, v in stats.get("per_task", {}).items():
        fam, tid = key.split(":")
        t = gd.tamers.get(int(tid), {})
        fam_name = C.FAMILY_NAMES[int(fam)] if fam != "None" else "любое"
        rows.append((v["wr"], v["n"], t.get("name", tid), t.get("zone", ""), fam_name))
    rows.sort()
    def table(items):
        return "".join(f"<tr><td>{html.escape(str(n))}</td><td>{html.escape(str(z))}</td><td>{html.escape(f)}</td>"
                       f"<td class=num>{wr:.0%}</td><td class=num>{cnt}</td></tr>" for wr, cnt, n, z, f in items)
    solved = sum(1 for r in rows if r[0] >= 0.8)
    page = f"""<!doctype html><html lang=ru><head><meta charset=utf-8><meta http-equiv=refresh content=30>
<title>Обучение: битвы питомцев</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0; --accent:#4f8cff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --card:#1f2024; --fg:#ececec; --muted:#9a9aa2; --line:#2e2f34; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,Segoe UI,sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:820px; margin:auto; }} h1 {{ font-size:22px; margin:0 0 4px; }} .muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:14px 0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
.kpi b {{ display:block; font-size:24px; font-variant-numeric:tabular-nums; }} .kpi span {{ color:var(--muted); font-size:13px; }}
.bar {{ height:12px; background:var(--line); border-radius:6px; overflow:hidden; }} .bar i {{ display:block; height:100%; background:var(--accent); }}
svg .grid {{ stroke:var(--line); }} svg .axis {{ fill:var(--muted); font-size:11px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }} td,th {{ padding:5px 6px; border-bottom:1px solid var(--line); text-align:left; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }} h2 {{ font-size:16px; margin:0 0 10px; }}
</style></head><body><main>
<h1>Обучение нейросети: битвы питомцев</h1>
<div class=muted>Запуск: {html.escape(stats.get('name', ''))} · обновлено {time.strftime('%H:%M:%S')} · страница обновляется каждые 30 с</div>
<div class=card><div class=kpis>
<div class=kpi><b>{last.get('winrate', 0):.0%}</b><span>доля побед (последние 2000 боёв)</span></div>
<div class=kpi><b>{last.get('games', 0):,}</b><span>сыграно боёв</span></div>
<div class=kpi><b>{solved} / {len(rows)}</b><span>задач с победами ≥ 80%</span></div>
<div class=kpi><b>{_fmt_dur(eta)}</b><span>осталось до конца запуска</span></div>
</div>
<p style="margin:14px 0 6px">Прогресс: шаг {done} из {total} · прошло {_fmt_dur(elapsed)} · {last.get('sps', 0):,.0f} ходов/с</p>
<div class=bar><i style="width:{100 * done / max(1, total):.1f}%"></i></div></div>
<div class=card><h2>Доля побед</h2>{_chart(hist, 'winrate', ymax=1)}
<p class=muted>Команды при обучении случайные (из твоей коллекции нужного семейства), поэтому часть боёв проигрышна при любой игре. Важен рост, а не абсолютное значение.</p></div>
<div class=card><h2>Средняя длина боя, раундов</h2>{_chart(hist, 'rounds', color='#e0873a')}</div>
<div class=card><h2>Самые трудные задачи</h2><table><tr><th>Тренер</th><th>Зона</th><th>Семейство</th><th class=num>Победы</th><th class=num>Боёв</th></tr>{table(rows[:15])}</table></div>
<div class=card><h2>Лучше всего получается</h2><table><tr><th>Тренер</th><th>Зона</th><th>Семейство</th><th class=num>Победы</th><th class=num>Боёв</th></tr>{table(rows[::-1][:10])}</table></div>
</main></body></html>"""
    path.write_text(page, encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(stats), encoding="utf-8")
