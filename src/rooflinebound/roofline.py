"""Roofline chart: one self-contained HTML file, no external resources.

Each operator family of a pass is a point at its arithmetic intensity
(FLOPs per compulsory HBM byte) and the performance its datasheet roof allows
there, min(peak, intensity x bandwidth). A point left of a roof's ridge is
HBM-bound. A point's area is its share of the pass's per-operator time. A
table under each chart carries every value, so nothing depends on hover.
"""

from __future__ import annotations

import html
import json
import math

from .bounds import PassBound
from .spec.hardware import HardwareSpec

__all__ = ["roofline_html"]

# Categorical slots 1-3 of the reference palette, validated all-pairs in both modes;
# everything else folds into a neutral "other". Shapes are a second encoding.
_GROUPS = [
    ("attention", "--series-1", "circle", {"attn_proj", "attn_core", "mtp"}),
    ("DSA indexer", "--series-2", "square", {"indexer_proj", "index_score"}),
    ("FFN / MoE", "--series-3", "triangle", {"dense_ffn", "shared_expert", "routed_expert",
                                             "router"}),
    ("other", "--muted", "diamond", set()),
]

_TEXT = {
    "en": {
        "title": "Roofline", "per_gpu": "Per GPU, datasheet peaks.",
        "intro": ("Each point is one operator family at the roof the datasheet allows: left of "
                  "a ridge it is limited by HBM bandwidth, right of it by compute. Point area is "
                  "its share of the pass's per-operator time. These are lower bounds on time, "
                  "not measurements."),
        "x": "arithmetic intensity (FLOP per HBM byte)", "y": "attainable FLOP/s per GPU",
        "op": "operator", "n": "count", "share": "time share", "bound": "bound by",
        "table": "All operators", "noflop": "no FLOPs (not plotted)",
        "pass": "pass bound", "decode": "Decode step, batch {b}", "prefill": "Prefill",
        "groups": ["attention", "DSA indexer", "FFN / MoE", "other"],
        "mtp": "MTP step (verify + drafts), batch {b}",
    },
    "zh": {
        "title": "Roofline", "per_gpu": "每卡，标称峰值。",
        "intro": ("每个点是一个算子族，位于标称规格允许的天花板上：在拐点左侧受 HBM 带宽限制，右侧受算力限制。"
                  "点的面积表示它在本次 pass 的逐算子时间中所占比例。这些是时间的下界，不是实测值。"),
        "x": "算术强度（每 HBM 字节的 FLOP）", "y": "每卡可达 FLOP/s",
        "op": "算子", "n": "数量", "share": "时间占比", "bound": "受限于",
        "table": "全部算子", "noflop": "无 FLOPs（未绘制）",
        "pass": "本次 pass 下界", "decode": "Decode 一步，batch {b}", "prefill": "Prefill",
        "groups": ["注意力", "DSA indexer", "FFN / MoE", "其他"],
        "mtp": "MTP 一步（验证 + 草稿），batch {b}",
    },
}

_LAYER_KINDS = {"dense", "moe", "dense+idx", "moe+idx"}

W, H = 760, 460
ML, MR, MT, MB = 72, 28, 24, 52


def _title(key: str, tx: dict) -> str:
    kind, _, batch = key.partition("@")
    return tx[kind].format(b=batch) if kind in ("decode", "mtp", "prefill") else key


def _group(category: str) -> int:
    for i, (_, _, _, cats) in enumerate(_GROUPS):
        if category in cats:
            return i
    return len(_GROUPS) - 1


def _fmt_si(x: float, unit: str) -> str:
    for p, s in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= p:
            return f"{x / p:.3g} {s}{unit}".strip()
    return f"{x:.3g} {unit}".strip()


def _shape(kind: str, x: float, y: float, r: float, cls: str) -> str:
    if kind == "circle":
        return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/>'
    if kind == "square":
        s = r * 1.7
        return (f'<rect class="{cls}" x="{x - s / 2:.1f}" y="{y - s / 2:.1f}" width="{s:.1f}" '
                f'height="{s:.1f}" rx="2"/>')
    if kind == "triangle":
        h = r * 2.0
        pts = f"{x:.1f},{y - h * 0.62:.1f} {x - h * 0.58:.1f},{y + h * 0.38:.1f} " \
              f"{x + h * 0.58:.1f},{y + h * 0.38:.1f}"
        return f'<polygon class="{cls}" points="{pts}"/>'
    d = r * 1.3
    pts = f"{x:.1f},{y - d:.1f} {x + d:.1f},{y:.1f} {x:.1f},{y + d:.1f} {x - d:.1f},{y:.1f}"
    return f'<polygon class="{cls}" points="{pts}"/>'


def _chart(title: str, b: PassBound, hw: HardwareSpec, tx: dict) -> str:
    bw = hw["mem_bw"].value
    peaks = {d: hw.peak_flops(d).value for d in ("bf16", "fp8", "fp4")}
    # One point per operator family: the same operator in dense, MoE and indexer
    # layers sits at the same intensity, so plotting each would stack marks.
    fam: dict[str, list] = {}
    for o in b.ops:
        kind, _, rest = o.name.partition(".")
        key = rest if kind in _LAYER_KINDS and rest else o.name
        f, m = o.count * o.flops, o.count * o.hbm_bytes()
        e = fam.setdefault(key, [key, o.category, 0, 0.0, 0.0, 0.0, 0.0])
        e[2] += o.count
        e[3] += f
        e[4] += m
        e[5] += f / peaks[o.dtype]
        e[6] += m / bw
    rows, skipped = [], []
    for name, cat, n, f, m, tc, tm in fam.values():
        r = (name, cat, n, f, m, tc, tm, max(tc, tm))
        (rows if f > 0 and m > 0 else skipped).append(r)
    total = sum(r[7] for r in rows + skipped) or 1.0

    xs = [r[3] / r[4] for r in rows]
    ridge = [p / bw for p in peaks.values()]
    xlo = 10 ** math.floor(math.log10(min(xs + ridge) / 2))
    xhi = 10 ** math.ceil(math.log10(max(xs + ridge) * 2))
    ylo = 10 ** math.floor(math.log10(min(min(peaks.values()), xlo * bw) / 1.5))
    yhi = 10 ** math.ceil(math.log10(max(peaks.values()) * 1.5))

    def px(x):
        return ML + (math.log10(x) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo)) \
            * (W - ML - MR)

    def py(y):
        return H - MB - (math.log10(y) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) \
            * (H - MT - MB)

    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    # grid and axes (recessive)
    for e in range(int(math.log10(xlo)), int(math.log10(xhi)) + 1):
        x = px(10 ** e)
        svg.append(f'<line class="grid" x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H - MB}"/>')
        svg.append(f'<text class="tick" x="{x:.1f}" y="{H - MB + 18}" '
                   f'text-anchor="middle">{_fmt_si(10 ** e, "")}</text>')
    for e in range(int(math.log10(ylo)), int(math.log10(yhi)) + 1):
        y = py(10 ** e)
        svg.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}"/>')
        svg.append(f'<text class="tick" x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">'
                   f'{_fmt_si(10 ** e, "")}</text>')
    svg.append(f'<line class="axis" x1="{ML}" y1="{H - MB}" x2="{W - MR}" y2="{H - MB}"/>')
    svg.append(f'<text class="axlabel" x="{(ML + W - MR) / 2}" y="{H - 10}" '
               f'text-anchor="middle">{html.escape(tx["x"])}</text>')
    svg.append(f'<text class="axlabel" transform="translate(16 {(MT + H - MB) / 2}) rotate(-90)" '
               f'text-anchor="middle">{html.escape(tx["y"])}</text>')

    # roofs: the HBM slope up to the highest ridge, a flat roof per dtype from its ridge
    top_ridge = max(ridge)
    svg.append(f'<line class="roof" x1="{px(xlo):.1f}" y1="{py(xlo * bw):.1f}" '
               f'x2="{px(top_ridge):.1f}" y2="{py(top_ridge * bw):.1f}"/>')
    # Collision boxes: every mark first, so labels are placed around them.
    order = sorted(rows, key=lambda r: -r[7])
    pos = [(px(r[3] / r[4]), py(r[3] / r[7]), 4 + 10 * math.sqrt(r[7] / total)) for r in order]
    boxes: list[tuple[float, float, float, float]] = [
        (x - rad - 2, y - rad - 2, x + rad + 2, y + rad + 2) for x, y, rad in pos]
    for d, p in sorted(peaks.items(), key=lambda kv: kv[1]):
        r = p / bw
        cls = "roof" if p == max(peaks.values()) else "roof dashed"
        svg.append(f'<line class="{cls}" x1="{px(r):.1f}" y1="{py(p):.1f}" x2="{W - MR}" '
                   f'y2="{py(p):.1f}"/>')
        text = f"{d} {_fmt_si(p, 'FLOP/s')} · ridge {r:.0f}"
        ty = py(p) - 6
        svg.append(f'<text class="rooflabel" x="{W - MR - 4}" y="{ty:.1f}" '
                   f'text-anchor="end">{text}</text>')
        boxes.append((W - MR - 4 - 6.2 * len(text), ty - 11, W - MR, ty + 3))
        boxes.append((px(r), py(p) - 3, W - MR, py(p) + 3))  # keep labels off the roof

    def overlaps(bx):
        return any(not (bx[2] < o[0] or bx[0] > o[2] or bx[3] < o[1] or bx[1] > o[3])
                   for o in boxes)

    # HBM roof label: rotated along the slope, at the first spot clear of marks.
    ang = math.degrees(math.atan2(py(top_ridge * bw) - py(xlo * bw), px(top_ridge) - px(xlo)))
    label = f'HBM {_fmt_si(bw, "B/s")}'
    for k in range(12):
        lx = xlo * 1.6 * 2 ** k
        if lx * 8 > top_ridge:
            break
        hx, hy = px(lx), py(lx * bw) - 8
        bx = (hx - 4, hy - 34, hx + 84, hy + 6)
        if not overlaps(bx):
            svg.append(f'<text class="rooflabel" transform="translate({hx:.1f} {hy:.1f}) '
                       f'rotate({ang:.1f})">{label}</text>')
            boxes.append(bx)
            break

    # points, largest first so small ones stay on top
    marks, labels = [], []
    for i, ((name, cat, _n, f, m, tc, tm, t), (x, y, rad)) in enumerate(zip(order, pos,
                                                                          strict=True)):
        g = _group(cat)
        gname, var, kind, _ = _GROUPS[g]
        gname = tx["groups"][g]
        data = {"name": name, "group": gname, "t": t, "share": t / total, "ai": f / m,
                "flops": f, "bytes": m, "bound": "compute" if tc > tm else "HBM"}
        marks.append(f'<g class="pt" tabindex="0" style="--c: var({var})" '
                     f"data-op='{html.escape(json.dumps(data), quote=True)}'>"
                     f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="{max(rad, 12):.1f}"/>'
                     + _shape(kind, x, y, rad, "mark") + "</g>")
        if i < 3:  # direct labels for the three biggest, where they fit
            w = 6.2 * len(name)
            for dx, dy, anchor in ((rad + 4, -rad - 2, "start"), (rad + 4, rad + 12, "start"),
                                   (-rad - 4, -rad - 2, "end"), (-rad - 4, rad + 12, "end")):
                lx0 = x + dx if anchor == "start" else x + dx - w
                bx = (lx0, y + dy - 11, lx0 + w, y + dy + 3)
                own = (x - rad - 2, y - rad - 2, x + rad + 2, y + rad + 2)
                boxes.remove(own)  # a label may sit beside its own mark
                clear = not overlaps(bx)
                boxes.append(own)
                if clear and ML <= bx[0] and bx[2] <= W - MR:
                    boxes.append(bx)
                    labels.append(f'<text class="dlabel" x="{x + dx:.1f}" y="{y + dy:.1f}" '
                                  f'text-anchor="{anchor}">{html.escape(name)}</text>')
                    break
    svg += marks + labels
    svg.append("</svg>")

    legend = "".join(
        f'<span class="key"><svg width="14" height="14" viewBox="0 0 14 14" style="--c: '
        f'var({var})">{_shape(kind, 7, 7, 4.5, "mark")}</svg>{html.escape(name)}</span>'
        for name, (_, var, kind, _) in zip(tx["groups"], _GROUPS, strict=True))

    head = (f"<tr><th>{tx['op']}</th><th>{tx['n']}</th><th>GFLOP</th><th>MB</th>"
            f"<th>FLOP/B</th><th>t_compute µs</th><th>t_HBM µs</th><th>{tx['bound']}</th>"
            f"<th>{tx['share']}</th></tr>")
    body = []
    for name, _, n, f, m, tc, tm, t in sorted(rows + skipped, key=lambda r: -r[7]):
        ai = f"{f / m:.1f}" if f > 0 and m > 0 else "–"
        bound = tx["noflop"] if f <= 0 else ("compute" if tc > tm else "HBM")
        body.append(f"<tr><td>{html.escape(name)}</td><td>{n}</td>"
                    f"<td>{f / 1e9:.3f}</td><td>{m / 1e6:.2f}</td><td>{ai}</td>"
                    f"<td>{tc * 1e6:.1f}</td><td>{tm * 1e6:.1f}</td><td>{bound}</td>"
                    f"<td>{t / total:.1%}</td></tr>")
    return (f'<section><h2>{html.escape(title)}</h2>'
            f'<p class="sub">MLA {html.escape(b.mla_mode)} · {tx["pass"]} '
            f'{html.escape(b.time.to("ms", 1e3).format())}</p>'
            f'<div class="legend">{legend}</div><div class="chart">{"".join(svg)}</div>'
            f'<details><summary>{tx["table"]}</summary><table>{head}{"".join(body)}</table>'
            f"</details></section>")


_CSS = """
.viz-root { color-scheme: light;
  --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root {
  color-scheme: dark; --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#f0efec;
  --text-secondary:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark; --surface-1:#1a1a19; --page:#0d0d0d;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,0.10); --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; }
body { margin:0; background:var(--page); }
.viz-root { background:var(--page); color:var(--text-primary);
  font:14px/1.5 system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  padding:16px; max-width:820px; margin:0 auto; }
h1 { font-size:20px; margin:8px 0 4px; } h2 { font-size:16px; margin:0 0 2px; }
.sub, .intro { color:var(--text-secondary); margin:0 0 8px; }
section { background:var(--surface-1); border:1px solid var(--border); border-radius:8px;
  padding:16px; margin:16px 0; }
.chart svg { width:100%; height:auto; display:block; }
.grid { stroke:var(--grid); stroke-width:1; } .axis { stroke:var(--axis); stroke-width:1; }
.tick, .axlabel { fill:var(--muted); font-size:11px; } .axlabel { font-size:12px; }
.roof { stroke:var(--text-secondary); stroke-width:2; fill:none; }
.roof.dashed { stroke-dasharray:6 4; }
.rooflabel { fill:var(--text-secondary); font-size:11px; }
.dlabel { fill:var(--text-secondary); font-size:11px; }
.mark { fill:var(--c); stroke:var(--surface-1); stroke-width:2; }
.hit { fill:transparent; }
.pt { cursor:default; outline:none; }
.pt:hover .mark, .pt:focus .mark { stroke:var(--text-primary); }
.legend { display:flex; flex-wrap:wrap; gap:16px; color:var(--text-secondary); margin:4px 0 8px; }
.key { display:inline-flex; align-items:center; gap:6px; }
#tip { position:fixed; pointer-events:none; background:var(--surface-1); color:var(--text-primary);
  border:1px solid var(--border); border-radius:6px; padding:6px 8px; font-size:12px;
  box-shadow:0 2px 8px rgba(0,0,0,.15); display:none; max-width:280px; }
#tip b { font-size:13px; } #tip .s { color:var(--text-secondary); }
details { margin-top:8px; } summary { cursor:pointer; color:var(--text-secondary); }
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:12px; margin-top:8px; display:block;
  overflow-x:auto; }
th, td { text-align:right; padding:3px 6px; border-bottom:1px solid var(--grid);
  white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
.notes { color:var(--text-secondary); font-size:12px; }
"""

_JS = """
const tip = document.getElementById('tip');
function show(g, x, y) {
  const d = JSON.parse(g.dataset.op);
  tip.replaceChildren();
  const v = document.createElement('b');
  v.textContent = (d.t * 1e6).toFixed(1) + ' µs · ' + (d.share * 100).toFixed(1) + '%';
  const n = document.createElement('div'); n.textContent = d.name;
  const s = document.createElement('div'); s.className = 's';
  s.textContent = d.group + ' · ' + d.ai.toFixed(1) + ' FLOP/B · ' + d.bound + '-bound · '
    + (d.flops / 1e9).toFixed(3) + ' GFLOP · ' + (d.bytes / 1e6).toFixed(2) + ' MB';
  tip.append(v, n, s);
  tip.style.display = 'block';
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(x + 14, innerWidth - r.width - 8) + 'px';
  tip.style.top = Math.max(y - r.height - 10, 8) + 'px';
}
for (const g of document.querySelectorAll('.pt')) {
  g.addEventListener('pointermove', e => show(g, e.clientX, e.clientY));
  g.addEventListener('pointerleave', () => tip.style.display = 'none');
  g.addEventListener('focus', () => { const r = g.getBoundingClientRect();
    show(g, r.right, r.top); });
  g.addEventListener('blur', () => tip.style.display = 'none');
}
"""


def roofline_html(model: str, hw: HardwareSpec, tables: dict[str, PassBound],
                  notes: list[str], lang: str = "en") -> str:
    tx = _TEXT.get(lang, _TEXT["en"])
    sections = "".join(_chart(_title(key, tx), b, hw, tx) for key, b in tables.items())
    note_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{tx['title']} · {html.escape(model)}</title><style>{_CSS}</style></head>"
            f'<body><div class="viz-root"><h1>{tx["title"]} · {html.escape(model)} · '
            f'{html.escape(hw.display)}</h1><p class="intro">{tx["per_gpu"]} {tx["intro"]}</p>'
            f'{sections}<ul class="notes">{note_html}</ul></div><div id="tip" role="tooltip">'
            f"</div><script>{_JS}</script></body></html>")
