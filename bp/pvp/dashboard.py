"""PvP training dashboard: logs/pvp_dashboard.html, reloads itself every 30 s.

Written by the trainer after every update; reads logs/pvp_sample_game.json (bench.py) if present.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from ..rl.dashboard import _fmt_dur

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#4f8cff", "#2fa36b", "#c77d1a", "#8a5cf6", "#d9485f", "#888"]


def _pct(x, signed=False):
    if x is None:
        return "—"
    return f"{x * 100:+.0f}%" if signed else f"{x:.0%}"


def _lines(hist, series, w=760, h=210, y0=0.0, y1=1.0, pct=True):
    """Multi-line SVG chart. series: [(key, label, color)]."""
    xs = [r["upd"] for r in hist]
    if not xs:
        return "<p class=muted>Пока нет данных</p>"
    x0, xN = min(xs), max(xs)
    pad_l, pad_r, pad_t, pad_b = 40, 12, 10, 24

    def X(v):
        return pad_l + (v - x0) / max(1, xN - x0) * (w - pad_l - pad_r)

    def Y(v):
        v = min(max(v, y0), y1)
        return pad_t + (1 - (v - y0) / (y1 - y0)) * (h - pad_t - pad_b)

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for k in range(5):
        v = y0 + (y1 - y0) * k / 4
        lab = f"{v * 100:.0f}%" if pct else f"{v:.0f}"
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class=grid />'
                   f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor=end class=axis>{lab}</text>')
    if y0 < 0 < y1:
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class=zero />')
    drawn = 0
    for key, _, color in series:
        pts = [(r["upd"], r[key]) for r in hist if r.get(key) is not None and r.get(key) == r.get(key)]
        if len(pts) == 1:
            x, y = pts[0]
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="{color}" />')
            drawn += 1
        elif pts:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
            out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" />')
            drawn += 1
    out.append(f'<text x="{pad_l}" y="{h - 6}" class=axis>шаг {x0}</text>'
               f'<text x="{w - pad_r}" y="{h - 6}" text-anchor=end class=axis>шаг {xN}</text></svg>')
    if not drawn:
        return "<p class=muted>Пока нет данных — первая точка появится после ближайшей проверки</p>"
    legend = "".join(f'<span><i style="background:{c}"></i>{html.escape(lab)}</span>' for _, lab, c in series if len(series) > 1)
    return "".join(out) + (f"<div class=legend>{legend}</div>" if legend else "")


def _last(history, key):
    for r in reversed(history):
        if r.get(key) is not None:
            return r[key]
    return None


def _run_progress(last):
    """'37 / 100 шагов' card while a limited training run is going (the learner writes last['run'])."""
    run = last.get("run") or {}
    if not run or time.time() - last.get("t", 0) > 900:
        return ""  # no run, or the last update is old: training is not running
    done = last["upd"] - run["from"]
    if run.get("steps"):
        total, unit = run["steps"], "шагов"
        frac = done / total
    elif run.get("minutes"):
        total, unit = run["minutes"], "минут"
        done = int((time.time() - run["started"]) / 60)
        frac = done / total
    else:
        return f"<div class=card><h2>Идёт обучение</h2><p>{done} шагов с начала запуска</p></div>"
    frac = max(0.0, min(1.0, frac))
    el = time.time() - run["started"]
    eta = el / frac - el if frac > 0.02 else None
    eta_txt = f" · осталось примерно {_fmt_dur(eta)}" if eta else ""
    return (f"<div class=card><h2>Идёт обучение: {done} / {total} {unit}</h2>"
            f"<div class=runbar><i style='width:{frac * 100:.1f}%'></i></div>"
            f"<p class=muted>{frac:.0%}{eta_txt} · после завершения обновится рейтинг и модель подсказчика</p></div>")


def _stages(current=1):
    items = [("1. Самообучение", "играет сама с собой и прошлыми версиями, учит основы"),
             ("2. Расчёт вперёд", "каждый ход просчитывается в симуляторе, сеть учится на расчётах"),
             ("3. Маленькая копия", "лёгкая версия для работы прямо в игре")]
    cells = []
    for i, (t, d) in enumerate(items, start=1):
        st = "идёт сейчас" if i == current else ("пройден" if i < current else "впереди")
        cls = "stage now" if i == current else "stage"
        cells.append(f'<div class="{cls}"><b>{t}</b><span>{d}</span><em>{st}</em></div>')
    return f'<div class=stages>{"".join(cells)}</div>'


FAMILY_RU = ["Гуманоид", "Дракон", "Летающий", "Нежить", "Зверёк", "Магический", "Элементаль", "Животное", "Водный", "Механизм"]


def _load_exclude():
    path = ROOT / "data" / "pvp" / "rank_exclude.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("items", []) if path.exists() else []
    except Exception:
        return []


def _pick_data():
    """name -> [species id used by the ratings, breed names] for the exclusion picker."""
    from ..sim.data import game_data
    from .pool import BREED_NAMES, breeds_of, classes
    gd = game_data()
    out = {}
    for c in classes():
        br = [BREED_NAMES.get(b, "?") for b in breeds_of(c[0])]
        for s in c:
            out.setdefault(gd.species[s]["name"], [c[0], br])
    return dict(sorted(out.items()))


def _advisor_token():
    env = ROOT / ".env"
    for line in (env.read_text(encoding="utf-8").splitlines() if env.exists() else []):
        if line.startswith("ADVISOR_TOKEN="):
            return line.split("=", 1)[1].strip()
    return ""


EXCLUDE_JS = """
(function(){
const $=id=>document.getElementById(id);
const BASE=location.protocol==='file:'?'http://localhost:5173':'';
let items=EX_ITEMS;
function draw(){
  $('exList').innerHTML=items.length?items.map((e,i)=>`<span class=chip>${e.name} · ${e.breed||'все породы'} <a href=# data-i=${i} title="убрать">✕</a></span>`).join(''):'<span class=muted><small>пусто</small></span>';
  $('exList').querySelectorAll('a').forEach(a=>a.onclick=ev=>{ev.preventDefault();save({remove:items[+a.dataset.i]});});
}
function status(t){$('exStatus').innerHTML='<small>'+t+'</small>';}
async function save(op){
  status('сохраняю…');
  try{
    const r=await fetch(BASE+'/api/exclude',{method:'POST',headers:{'Content-Type':'application/json','X-Token':EX_TOKEN},body:JSON.stringify(op)});
    const j=await r.json(); if(j.error) throw new Error(j.error);
    items=j.items; draw();
    status('сохранено · списки ниже обновятся через несколько секунд, команды пересчитываются в фоне (~5 мин)');
  }catch(e){status('не удалось сохранить — запущен ли советник (start_advisor.bat)? '+e.message);}
}
function breeds(){
  const p=EX_PETS[$('exName').value];
  $('exBreed').innerHTML='<option value="">все породы</option>'+(p?p[1].map(b=>`<option>${b}</option>`).join(''):'');
}
$('exName').oninput=breeds;
$('exAdd').onclick=()=>{
  const n=$('exName').value, p=EX_PETS[n];
  if(!p){status('нет такого питомца — выберите из подсказки');return;}
  const b=$('exBreed').value||null;
  $('exName').value='';breeds();save({add:{species:p[0],name:n,breed:b}});
};
draw();
fetch(BASE+'/api/exclude',{headers:{'X-Token':EX_TOKEN}}).then(r=>r.json()).then(j=>{items=j.items;draw();if(j.busy)status('команды пересчитываются…');}).catch(()=>{});
})();
"""


def _exclude_panel(items):
    try:
        data = _pick_data()
    except Exception:
        data = {}
    opts = "".join('<option value="%s">' % html.escape(n) for n in data)
    js = (EXCLUDE_JS.replace("EX_ITEMS", json.dumps(items, ensure_ascii=False))
          .replace("EX_TOKEN", json.dumps(_advisor_token())).replace("EX_PETS", "PETS"))
    return (f"<div class=excl><h3>Не предлагать (чёрный список)</h3>"
            f"<p class=muted><small>Питомцы или отдельные породы, которых у вас нет. Модель их по-прежнему знает и встречает у соперников, "
            f"просто в командах и списке питомцев ниже их не будет.</small></p>"
            f"<div class=exrow><input id=exName list=exNames placeholder='имя питомца, например Qiraji Guardling' autocomplete=off>"
            f"<select id=exBreed><option value=''>все породы</option></select><button id=exAdd>Добавить</button></div>"
            f"<datalist id=exNames>{opts}</datalist>"
            f"<div id=exList></div><div id=exStatus class=muted></div></div>"
            f"<script>const PETS={json.dumps(data, ensure_ascii=False)};{js}</script>")


def _rankings():
    path = ROOT / "logs" / "pvp_rankings.json"
    items = _load_exclude()
    panel = _exclude_panel(items)
    if not path.exists():
        return panel + "<p class=muted>Ещё не посчитано</p>"
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return panel + "<p class=muted>Нет данных</p>"
    excl = {}
    for e in items:
        s, b = int(e["species"]), e.get("breed")
        excl[s] = None if (b is None or excl.get(s, set()) is None) else excl.get(s, set()) | {b}

    def allowed(s, breeds):
        if s not in excl:
            return list(breeds)
        return [] if excl[s] is None else [b for b in breeds if b not in excl[s]]

    teams = [t for t in r.get("teams", []) if all(allowed(s, [b]) for s, b in t.get("specs", []))][:20]
    pets = []
    for p in r.get("pets", []):
        ok = allowed(p["species"], p["breeds"])
        if not ok:
            continue
        bs = p.get("breed_scores") or {}
        best = p.get("breed") if p.get("breed") in ok else max(ok, key=lambda b: bs.get(b, -9))
        pets.append(dict(p, breed=best, breeds=ok))
        if len(pets) >= 50:
            break
    when = time.strftime("%d.%m %H:%M", time.localtime(r.get("time", 0)))
    teams = "".join(
        f"<tr><td class=num>{i}</td><td>{'<br>'.join(html.escape(x) for x in t['pets'])}</td>"
        f"<td class=num>{t['winrate']:.0%}</td></tr>" for i, t in enumerate(teams, 1))
    pets = "".join(
        f"<tr><td class=num>{i}</td><td><b>{html.escape(p['name'])}</b> <span class=muted>{FAMILY_RU[p['family']]}"
        f" · порода {html.escape(p.get('breed') or p['breeds'][0])}"
        f"{(' (есть: ' + html.escape(', '.join(p['breeds'])) + ')') if len(p['breeds']) > 1 else ''}</span><br><small>{html.escape(' · '.join(p['abilities']))}</small></td>"
        f"<td class=num>{p['score']:.0%}</td></tr>" for i, p in enumerate(pets, 1))
    return (panel + f"<p class=muted>Посчитано {when} на версии {r.get('update')} · против: {html.escape(r.get('opponents', ''))}</p>"
            f"<h3>20 лучших команд</h3><div class=scroll><table><tr><th>#</th><th>команда (порода: способности)</th>"
            f"<th class=num>побед в симуляции</th></tr>{teams}</table></div>"
            f"<h3 style='margin-top:14px'>50 лучших питомцев</h3><div class=scroll><table><tr><th>#</th><th>питомец и лучший набор способностей</th>"
            f"<th class=num>оценка*</th></tr>{pets}</table></div>"
            f"<p class=muted><small>* как модель оценивает шансы команды с этим питомцем (и случайными напарниками) против меты. "
            f"Это мнение текущей модели, а не статистика реальных боёв.</small></p>")


def _sample_game():
    path = ROOT / "logs" / "pvp_sample_game.json"
    if not path.exists():
        return "<p class=muted>Появится после первой проверки</p>"
    try:
        g = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "<p class=muted>Нет данных</p>"

    def team(t):
        return "".join(f"<li><b>{html.escape(p['name'])}</b> <span class=muted>{html.escape(p['breed'])}</span><br>"
                       f"<small>{html.escape(' · '.join(p['abilities']))}</small></li>" for p in t)

    def hpbar(hp):
        return "".join(f'<i style="width:{max(0.0, h) * 38:.0f}px"></i>' if h > 0 else '<i class=dead></i>' for h in hp)

    rows = []
    for r in g["rounds"]:
        ph = {0: "выбор первого", 2: "замена"}.get(r["phase"], f"раунд {r['round']}")
        win = f"{r['win']:.0%}" if r.get("win") is not None else ""
        hp = r.get("hp") or [[], []]
        rows.append(f"<tr><td class=muted>{ph}</td><td>{html.escape(r['me'] or '—')}</td><td>{html.escape(r['opp'] or '—')}</td>"
                    f"<td class=num>{win}</td><td><div class=hp>{hpbar(hp[0])}</div><div class='hp foe'>{hpbar(hp[1])}</div></td></tr>")
    res = {0: "победа модели", 1: "победа бота", -1: "ничья"}.get(g.get("winner"), "")
    return (f"<p class=muted>Шаг обучения {g.get('upd', '?')} · модель против «жадного» бота · итог: <b>{res}</b></p>"
            f"<div class=teams><div><h3>Модель</h3><ul>{team(g['teams'][0])}</ul></div>"
            f"<div><h3>Бот</h3><ul>{team(g['teams'][1])}</ul></div></div>"
            f"<div class=scroll><table class=game><tr><th></th><th>ход модели</th><th>ход бота</th><th class=num>шанс<br>победы*</th><th>здоровье</th></tr>"
            f"{''.join(rows)}</table></div>"
            f"<p class=muted><small>* как модель сама оценивает свои шансы перед ходом. Полоски: верхняя — питомцы модели, нижняя — бота.</small></p>")


def render(path: Path, history, league, n_params, t_start, start_upd):
    last = history[-1] if history else {}
    elapsed = time.time() - t_start
    done = last.get("upd", 0) - start_upd
    per_upd = elapsed / done if done > 0 else 0
    per_hour = last.get("n", 65536) * 3600 / per_upd if per_upd else 0
    snaps = league.snaps[-10:]
    rows = []
    for sn in reversed(snaps):
        lab = "snap@%d" % sn["upd"]
        rows.append(f"<tr><td>версия {sn['upd']}</td><td class=num>{_pct(league.winrate(lab))}</td>"
                    f"<td class=num>{len(league.results.get(lab, []))}</td></tr>")
    rows = "".join(rows)
    bench_hist = [r for r in history if r.get("bench_greedy") is not None]
    tac_hist = [r for r in history if r.get("tac_finish") is not None]
    tac = [("tac_finish", "Добивает врага, у которого 1 здоровья", 0.0, 1.0, False, "доля верного удара, цель ~100%"),
           ("tac_family2", "Чаще берёт атаку, сильную против семейства врага", -0.1, 0.5, True, "на сколько чаще в той же позиции"),
           ("tac_untarget", "Не бьёт в закопавшегося / взлетевшего врага", -0.1, 0.5, True, "на сколько реже бьёт, больше — лучше"),
           ("tac_immortal", "Не тратит удары на «бессмертную» нежить", -0.1, 0.5, True, "на сколько реже бьёт, больше — лучше"),
           ("tac_cc_critter", "Не оглушает зверька (он невосприимчив)", -0.1, 0.5, True, "на сколько реже оглушает, больше — лучше")]
    tac_cards = "".join(
        f"<div class=mini><h3>{t}</h3><div class=big>{_pct(_last(history, k), signed)}</div>"
        f"{_lines(tac_hist, [(k, t, COLORS[i])], h=120, y0=y0, y1=y1)}<p class=muted><small>{note}</small></p></div>"
        for i, (k, t, y0, y1, signed, note) in enumerate(tac))
    page = f"""<!doctype html><html lang=ru><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>PvP-обучение</title>
<style>
:root {{ --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e0; --accent:#4f8cff; --good:#2fa36b; --bad:#d9485f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --card:#1f2024; --fg:#ececec; --muted:#9a9aa2; --line:#2e2f34; }} }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,Segoe UI,sans-serif; margin:0; padding:20px 16px; }}
main {{ max-width:900px; margin:auto; }} h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:0 0 10px; }}
h3 {{ font-size:14px; margin:0 0 6px; }} .muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:14px 0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; }}
.kpi b {{ display:block; font-size:23px; font-variant-numeric:tabular-nums; }} .kpi span {{ color:var(--muted); font-size:13px; }}
.runbar {{ height:14px; background:var(--line); border-radius:7px; overflow:hidden; }} .runbar i {{ display:block; height:100%; background:var(--good); }}
.stages {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:10px; }}
.stage {{ border:1px solid var(--line); border-radius:10px; padding:10px 12px; display:flex; flex-direction:column; gap:4px; opacity:.7; }}
.stage.now {{ border-color:var(--accent); opacity:1; box-shadow:0 0 0 1px var(--accent) inset; }}
.stage span {{ font-size:13px; color:var(--muted); }} .stage em {{ font-style:normal; font-size:12px; color:var(--accent); }}
svg .grid {{ stroke:var(--line); }} svg .zero {{ stroke:var(--muted); stroke-dasharray:3 3; }} svg .axis {{ fill:var(--muted); font-size:11px; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--muted); margin-top:4px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }}
.mini {{ border:1px solid var(--line); border-radius:10px; padding:10px 12px; }} .mini .big {{ font-size:22px; font-variant-numeric:tabular-nums; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }} td,th {{ padding:5px 6px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.teams {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }} .teams ul {{ margin:0; padding-left:18px; }} .teams li {{ margin-bottom:6px; }}
.scroll {{ overflow-x:auto; }}
.excl {{ border:1px dashed var(--line); border-radius:10px; padding:10px 12px; margin-bottom:12px; }}
.exrow {{ display:flex; gap:8px; flex-wrap:wrap; }} .exrow input {{ flex:1 1 220px; min-width:0; }}
.exrow input, .exrow select, .exrow button {{ font:inherit; padding:5px 8px; border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--fg); }}
.exrow button {{ background:var(--accent); color:#fff; border-color:var(--accent); cursor:pointer; }}
#exList {{ margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; }}
.chip {{ background:var(--bg); border:1px solid var(--line); border-radius:999px; padding:2px 10px; font-size:13px; }}
.chip a {{ color:var(--bad); text-decoration:none; margin-left:4px; }} table.game td {{ font-size:13px; }}
.hp {{ display:flex; gap:3px; width:124px; margin:2px 0; }} .hp i {{ display:block; height:6px; background:var(--good); border-radius:3px; flex:0 0 auto; }}
.hp.foe i {{ background:var(--bad); }} .hp i.dead {{ background:var(--line); width:38px; }}
@media (max-width:560px) {{ .teams {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>PvP-модель: как идёт обучение</h1>
<div class=muted>обновлено {time.strftime('%H:%M:%S')} · страница обновляется сама каждые 30 с</div>
{_run_progress(last)}<div class=card>{_stages(last.get('stage', 1))}</div>
<div class=card><div class=kpis>
<div class=kpi><b>{last.get('upd', 0):,}</b><span>шагов обучения</span></div>
<div class=kpi><b>{last.get('games', 0):,}</b><span>сыграно боёв</span></div>
<div class=kpi><b>{last.get('samples', 0) / 1e6:.1f} млн</b><span>обработано решений</span></div>
<div class=kpi><b>{per_hour / 1e6:.1f} млн</b><span>решений в час</span></div>
<div class=kpi><b>{n_params / 1e6:.1f} млн</b><span>параметров модели</span></div>
<div class=kpi><b>{_fmt_dur(elapsed)}</b><span>с последнего запуска</span></div>
</div></div>
<div class=card><h2>Сила модели на контрольных боях</h2>
{_lines(bench_hist, [('bench_greedy', 'против «жадного» бота', COLORS[0]), ('bench_early', 'против ранней версии (шаг 20)', COLORS[1])])}
<p class=muted>Каждые 25 шагов модель играет одни и те же 400 боёв (одинаковые пары команд, каждая команда за обе стороны), поэтому точки можно честно сравнивать между собой. Сейчас: против бота {_pct(_last(history, 'bench_greedy'))}, против ранней версии {_pct(_last(history, 'bench_early'))}.</p></div>
<div class=card><h2>Экзамен по приёмам</h2><div class=grid2>{tac_cards}</div>
<p class=muted>Позиции берутся из случайных боёв; одна и та же позиция сравнивается «с деталью» и без неё, поэтому видно именно реакцию на приём.</p></div>
<div class=card><h2>Чтение соперника</h2>
{_lines([r for r in history if r.get('opp_acc') is not None], [('opp_acc', 'угадывает его ход в раунде', COLORS[2]), ('lo_acc', 'угадывает его скрытые способности (50% = наугад)', COLORS[3])])}</div>
<div class=card><h2>Рейтинг по мнению модели</h2>{_rankings()}</div>
<div class=card><h2>Пример боя (последняя проверка)</h2>{_sample_game()}</div>
<div class=card><h2>Обучение внутри лиги</h2>
{_lines([r for r in history if r.get('wr_greedy') is not None], [('wr_greedy', 'против бота в тренировочных боях', COLORS[0]), ('wr_snaps', 'против своих прошлых версий', COLORS[1])])}
<p class=muted>Команды и соперники здесь всё время меняются, поэтому линии шумные. Выше 50% против прошлых версий — модель всё ещё растёт.</p>
<table><tr><th>соперник из лиги</th><th class=num>наши победы</th><th class=num>боёв</th></tr>{rows}</table></div>
<div class=card><h2>Скорость</h2><p>{per_upd:.0f} с на шаг обучения · средняя длина боя {last.get('rounds', 0):.1f} раунда</p></div>
</main>
<script>setInterval(()=>{{const a=document.activeElement;if(!a||!['INPUT','SELECT'].includes(a.tagName))location.reload();}},30000);</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(path)


def main():
    """Re-draw the page from the last checkpoint (when no training is running)."""
    import torch
    from .train import League
    ck = torch.load(ROOT / "checkpoints" / "pvp" / "stage2.pt", map_location="cpu", weights_only=False)
    lg = League(ROOT / "checkpoints" / "pvp" / "league")
    lg.load_state(ck.get("league", {}))
    n = sum(v.numel() for v in ck["model"].values())
    render(ROOT / "logs" / "pvp_dashboard.html", ck["history"], lg, n, time.time(), ck["history"][-1]["upd"])


if __name__ == "__main__":
    main()
