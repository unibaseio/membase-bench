"""Render the README figures in assets/ from the published runs' numbers."""

from __future__ import annotations

import math
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

BLUE, LIGHT, PALE, GREY = "#3E61FF", "#9DB0FF", "#DCE3FF", "#8B949E"
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'

# Mean tokens per question: reader context (efficiency sample) and the full history (o200k).
CONTEXT = [("LoCoMo", 6562, 19987), ("LongMemEval_S", 8970, 102795)]

LOCOMO = [("single_hop", 94.6), ("multi_hop", 93.6), ("temporal", 91.6), ("open_domain", 83.3)]
LONGMEMEVAL = [
    ("single-session-preference", 100.0), ("single-session-user", 98.6), ("knowledge-update", 97.4),
    ("temporal-reasoning", 92.5), ("multi-session", 88.0), ("single-session-assistant", 85.7),
]

# Wrong answers with gold evidence: its session not retrieved, or retrieved and still answered wrong.
MISSES = [("LoCoMo", 16, 89), ("LongMemEval_S", 0, 37)]

# LongMemEval_S 100-question sample, same stores, reader swapped.
READERS = [("gpt-5.4-mini", 77), ("gpt-4.1-mini", 83), ("gpt-4.1", 86), ("gpt-5", 87),
           ("gpt-5-mini", 87), ("gpt-5.4", 89), ("gpt-5.5", 96)]

# Serial timing in seconds: (search p50, search p95, total p50, total p95).
LATENCY = [("LoCoMo", 1.67, 7.02, 8.30, 18.0), ("LongMemEval_S", 2.53, 6.11, 14.7, 30.2),
           ("DMR", 1.13, 1.71, 3.21, 6.34)]


def _svg(width: int, height: int, body: str, label: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">'
            f'<style>text{{font-family:{FONT};fill:{GREY}}} .t{{font-size:15px;font-weight:600;fill:{BLUE}}}'
            f' .s{{font-size:13px}} .v{{font-size:13px;font-weight:600}} .w{{fill:#FFFFFF}}</style>'
            f'{body}</svg>\n')


def _legend(x: int, y: int, items: list[tuple[str, str]]) -> str:
    out = []
    for color, name in items:
        out.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{color}"/>'
                   f'<text class="s" x="{x + 18}" y="{y}">{name}</text>')
        x += 30 + 7 * len(name)
    return "".join(out)


def pipeline() -> str:
    steps = [("Ingest", "sessions → dated episodes"), ("Search", "multi-round LLM decider"),
             ("Answer", "reader model"), ("Judge", "gpt-4o-mini vs gold")]
    w, h, gap = 186, 74, 32
    body = []
    for i, (name, sub) in enumerate(steps):
        x = i * (w + gap)
        fill = BLUE if i < 2 else LIGHT
        body.append(f'<rect x="{x}" y="10" width="{w}" height="{h}" rx="14" fill="{fill}"/>'
                    f'<text class="w" x="{x + w / 2}" y="42" text-anchor="middle" '
                    f'style="font-size:17px;font-weight:700">{name}</text>'
                    f'<text class="w" x="{x + w / 2}" y="64" text-anchor="middle" '
                    f'style="font-size:12.5px">{sub}</text>')
        if i < len(steps) - 1:
            ax = x + w + 6
            body.append(f'<path d="M{ax} 47 h{gap - 14} m-7 -6 l7 6 l-7 6" stroke="{GREY}" '
                        f'stroke-width="2" fill="none"/>')
    body.append(f'<text class="s" x="{(w * 2 + gap) / 2}" y="108" text-anchor="middle">'
                f'membase-core, public API only</text>'
                f'<text class="s" x="{w * 3 + gap * 2.5}" y="108" text-anchor="middle">'
                f'OpenAI, outside the engine</text>')
    return _svg(4 * w + 3 * gap, 118, "".join(body), "Ingest, search, answer, judge")


def context() -> str:
    left, top, bar, row = 130, 46, 18, 64
    width = 840
    scale = (width - left - 90) / math.log10(150_000 / 1_000)

    def x(v: float) -> float:
        return left + scale * math.log10(max(v, 1_000) / 1_000)

    body = ['<text class="t" x="0" y="18">Tokens per question: what the reader gets vs the full history</text>',
            _legend(0, 40, [(BLUE, "context Membase passes"), (PALE, "full conversation history")])]
    for i, (name, ctx, full) in enumerate(CONTEXT):
        y = top + 16 + i * row
        body.append(f'<text class="v" x="0" y="{y + 22}">{name}</text>')
        for j, (val, color) in enumerate(((ctx, BLUE), (full, PALE))):
            yy = y + j * (bar + 4)
            body.append(f'<rect x="{left}" y="{yy}" width="{x(val) - left:.1f}" height="{bar}" rx="4" fill="{color}"/>'
                        f'<text class="s" x="{x(val) + 8:.1f}" y="{yy + 14}">{val:,}</text>')
    n = len(CONTEXT)
    for tick in (1_000, 10_000, 100_000):
        body.append(f'<text class="s" x="{x(tick):.1f}" y="{top + 16 + n * row}" text-anchor="middle">'
                    f'{tick // 1000}k</text>')
    return _svg(width, top + 24 + n * row, "".join(body), "Context tokens against full history")


def _hbars(title: str, rows: list[tuple[str, float]], lo: float, label_w: int, fmt: str) -> str:
    width, bar, gap = 840, 20, 10
    span = width - label_w - 60
    body = [f'<text class="t" x="0" y="18">{title}</text>']
    for i, (name, val) in enumerate(rows):
        y = 34 + i * (bar + gap)
        w = span * (val - lo) / (100 - lo)
        body.append(f'<text class="s" x="0" y="{y + 15}">{name}</text>'
                    f'<rect x="{label_w}" y="{y}" width="{span}" height="{bar}" rx="4" fill="{PALE}" opacity=".45"/>'
                    f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar}" rx="4" fill="{BLUE}"/>'
                    f'<text class="v" x="{label_w + w + 8:.1f}" y="{y + 15}">{fmt.format(val)}</text>')
    return _svg(width, 40 + len(rows) * (bar + gap), "".join(body), title)


def _inner(svg: str) -> str:
    return svg[svg.index(">", svg.index("<svg")) + 1:svg.rindex("</svg>")]


def categories() -> str:
    a = _hbars("LoCoMo accuracy by category", LOCOMO, 50, 210, "{:.1f}%")
    b = _hbars("LongMemEval_S accuracy by question type", LONGMEMEVAL, 50, 210, "{:.1f}%")
    ha, hb = int(a.split('height="')[1].split('"')[0]), int(b.split('height="')[1].split('"')[0])
    style = a[a.index("<style>"):a.index("</style>") + 8]
    body = _inner(a).replace(style, "") + f'<g transform="translate(0,{ha + 16})">' + _inner(b).replace(style, "") + "</g>"
    return _svg(840, ha + hb + 16, style + body, "Accuracy by category and question type")


def misses() -> str:
    left, bar, row, width = 130, 22, 40, 840
    most = max(r + m for _, r, m in MISSES)
    scale = (width - left - 230) / most
    body = ['<text class="t" x="0" y="18">Why answers were wrong</text>',
            _legend(0, 40, [(LIGHT, "gold session not retrieved"), (BLUE, "gold session retrieved, answer still wrong")])]
    for i, (name, ret, read) in enumerate(MISSES):
        y = 56 + i * row
        body.append(f'<text class="v" x="0" y="{y + 16}">{name}</text>')
        x0 = left
        for val, color in ((ret, LIGHT), (read, BLUE)):
            if val:
                body.append(f'<rect x="{x0:.1f}" y="{y}" width="{val * scale:.1f}" height="{bar}" rx="4" fill="{color}"/>')
                x0 += val * scale
        note = f"{ret} + {read} of {ret + read} misses" if ret else f"all {read} misses"
        body.append(f'<text class="s" x="{x0 + 8:.1f}" y="{y + 16}">{note}</text>')
    return _svg(width, 60 + len(MISSES) * row, "".join(body), "Retrieval misses against reader misses")


def readers() -> str:
    return _hbars("LongMemEval_S, 100-question sample: accuracy by reader (same memory)",
                  READERS, 50, 140, "{:.0f}%")


def latency() -> str:
    left, bar, row, width = 130, 16, 60, 840
    scale = (width - left - 120) / 32
    body = ['<text class="t" x="0" y="18">Latency per question, one at a time (seconds)</text>',
            _legend(0, 40, [(BLUE, "search"), (LIGHT, "total"), (PALE, "p95")])]
    for i, (name, sp50, sp95, tp50, tp95) in enumerate(LATENCY):
        y = 58 + i * row
        body.append(f'<text class="v" x="0" y="{y + 20}">{name}</text>')
        for j, (p50, p95, color) in enumerate(((sp50, sp95, BLUE), (tp50, tp95, LIGHT))):
            yy = y + j * (bar + 4)
            body.append(f'<rect x="{left}" y="{yy}" width="{p95 * scale:.1f}" height="{bar}" rx="4" fill="{PALE}" opacity=".6"/>'
                        f'<rect x="{left}" y="{yy}" width="{p50 * scale:.1f}" height="{bar}" rx="4" fill="{color}"/>'
                        f'<text class="s" x="{left + p95 * scale + 8:.1f}" y="{yy + 13}">p50 {p50:g} · p95 {p95:g}</text>')
    return _svg(width, 62 + len(LATENCY) * row, "".join(body), "Search and total latency")


FIGURES = {"pipeline": pipeline, "context": context, "categories": categories,
           "misses": misses, "readers": readers, "latency": latency}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (ASSETS / f"{name}.svg").write_text(fn())
        print(f"assets/{name}.svg")
