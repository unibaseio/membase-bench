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

# Share of gold evidence sessions among the retrieved episodes' sessions.
RECALL = [("LoCoMo · single_hop", 99.4), ("LoCoMo · temporal", 98.4), ("LoCoMo · multi_hop", 95.8),
          ("LoCoMo · open_domain", 92.3), ("LongMemEval_S · all types", 99.95)]

# LoCoMo, same stores, reader swapped: (gpt-4.1-mini, gpt-5.5).
SWAP = [("single_hop", 94.6, 95.0), ("multi_hop", 93.6, 93.3), ("temporal", 91.6, 91.0),
        ("open_domain", 83.3, 84.4)]

# LoCoMo accuracy per conversation, and by how many sessions the gold evidence spans.
CONVERSATIONS = [("conv-26", 94.1), ("conv-30", 95.1), ("conv-41", 92.1), ("conv-42", 89.9),
                 ("conv-43", 93.3), ("conv-44", 93.5), ("conv-47", 94.0), ("conv-48", 93.7),
                 ("conv-49", 92.3), ("conv-50", 94.9)]
EVIDENCE = [("1 session (1,207 q)", 93.5), ("2 sessions (203 q)", 90.1), ("3+ sessions (126 q)", 95.2)]

# LongMemEval_S 100-question sample: accuracy by reader and question type.
TYPES = ["single-session-user", "single-session-preference", "knowledge-update", "temporal-reasoning",
         "multi-session", "single-session-assistant"]
HEATMAP = {
    "gpt-5.4-mini": [100.0, 100.0, 86.7, 74.1, 63.0, 63.6],
    "gpt-4.1-mini": [100.0, 100.0, 86.7, 81.5, 70.4, 81.8],
    "gpt-4.1": [92.9, 83.3, 100.0, 88.9, 74.1, 81.8],
    "gpt-5": [100.0, 100.0, 93.3, 88.9, 77.8, 72.7],
    "gpt-5-mini": [100.0, 83.3, 100.0, 88.9, 74.1, 81.8],
    "gpt-5.4": [100.0, 83.3, 93.3, 88.9, 88.9, 72.7],
    "gpt-5.5": [100.0, 100.0, 100.0, 100.0, 92.6, 81.8],
}

# Context tokens per question over the efficiency sample: (min, p25, median, p75, max).
SPREAD = [("LoCoMo", 6214, 6463, 6520, 6642, 7208), ("LongMemEval_S", 3022, 8500, 9628, 9855, 11171),
          ("DMR", 1458, 1547, 1612, 1658, 1776)]


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


def recall() -> str:
    return _hbars("Retrieval recall: gold evidence sessions found among the retrieved episodes",
                  RECALL, 50, 210, "{:g}%")


def swap() -> str:
    left, bar, row, width = 130, 16, 46, 840
    span = width - left - 70
    x = lambda v: span * (v - 50) / 50  # noqa: E731
    body = ['<text class="t" x="0" y="18">LoCoMo by category, same memory, reader swapped</text>',
            _legend(0, 40, [(LIGHT, "gpt-4.1-mini"), (BLUE, "gpt-5.5")])]
    for i, (name, a, b) in enumerate(SWAP):
        y = 56 + i * row
        body.append(f'<text class="s" x="0" y="{y + 16}">{name}</text>')
        for j, (val, color) in enumerate(((a, LIGHT), (b, BLUE))):
            yy = y + j * (bar + 3)
            body.append(f'<rect x="{left}" y="{yy}" width="{x(val):.1f}" height="{bar}" rx="4" fill="{color}"/>'
                        f'<text class="s" x="{left + x(val) + 8:.1f}" y="{yy + 13}">{val:.1f}%</text>')
    return _svg(width, 60 + len(SWAP) * row, "".join(body), "LoCoMo accuracy with two readers")


def conversations() -> str:
    width, height, top, base = 840, 230, 40, 190
    n = len(CONVERSATIONS)
    slot = width / n
    y = lambda v: base - (base - top) * (v - 80) / 20  # noqa: E731
    body = ['<text class="t" x="0" y="18">LoCoMo accuracy per conversation (axis from 80%)</text>',
            f'<line x1="0" y1="{y(93.12):.1f}" x2="{width}" y2="{y(93.12):.1f}" stroke="{GREY}" '
            f'stroke-dasharray="4 4"/><text class="s" x="{width}" y="18" '
            f'text-anchor="end">dashed: all 1,540 questions, 93.1%</text>']
    for i, (name, val) in enumerate(CONVERSATIONS):
        cx = i * slot + slot / 2
        body.append(f'<rect x="{cx - 24:.1f}" y="{y(val):.1f}" width="48" height="{base - y(val):.1f}" rx="5" fill="{BLUE}"/>'
                    f'<text class="v" x="{cx:.1f}" y="{y(val) - 8:.1f}" text-anchor="middle">{val:.1f}</text>'
                    f'<text class="s" x="{cx:.1f}" y="{base + 20}" text-anchor="middle">{name}</text>')
    return _svg(width, height, "".join(body), "LoCoMo accuracy per conversation")


def evidence() -> str:
    return _hbars("LoCoMo accuracy by how many sessions the gold evidence spans", EVIDENCE, 50, 210,
                  "{:.1f}%")


def heatmap() -> str:
    left, top, cw, ch, width = 120, 92, 118, 30, 840
    body = ['<text class="t" x="0" y="18">LongMemEval_S sample: accuracy by reader and question type (%)</text>']
    for j, t in enumerate(TYPES):
        cx = left + j * cw + cw / 2
        a, b = t.rsplit("-", 1)
        body.append(f'<text class="s" x="{cx:.0f}" y="{top - 26}" text-anchor="middle">{a}</text>'
                    f'<text class="s" x="{cx:.0f}" y="{top - 10}" text-anchor="middle">{b}</text>')
    for i, (reader, vals) in enumerate(HEATMAP.items()):
        y = top + i * ch
        body.append(f'<text class="v" x="0" y="{y + 20}">{reader}</text>')
        for j, v in enumerate(vals):
            alpha = 0.12 + 0.88 * max(0.0, (v - 60) / 40)
            ink = "#FFFFFF" if alpha > 0.55 else GREY
            body.append(f'<rect x="{left + j * cw + 2}" y="{y + 2}" width="{cw - 4}" height="{ch - 4}" rx="4" '
                        f'fill="{BLUE}" fill-opacity="{alpha:.2f}"/>'
                        f'<text x="{left + j * cw + cw / 2:.0f}" y="{y + 20}" text-anchor="middle" '
                        f'style="font-size:13px;font-weight:600;fill:{ink}">{v:.0f}</text>')
    return _svg(width, top + len(HEATMAP) * ch + 8, "".join(body), "Accuracy by reader and question type")


def spread() -> str:
    left, row, width = 130, 44, 840
    scale = (width - left - 40) / 15_000
    x = lambda v: left + v * scale  # noqa: E731
    body = ['<text class="t" x="0" y="18">Context tokens per question: range, middle half and median</text>']
    for i, (name, lo, q1, med, q3, hi) in enumerate(SPREAD):
        y = 44 + i * row
        body.append(f'<text class="v" x="0" y="{y + 15}">{name}</text>'
                    f'<line x1="{x(lo):.1f}" y1="{y + 10}" x2="{x(hi):.1f}" y2="{y + 10}" stroke="{LIGHT}" stroke-width="2"/>'
                    f'<rect x="{x(q1):.1f}" y="{y}" width="{max(3.0, x(q3) - x(q1)):.1f}" height="20" rx="4" fill="{BLUE}"/>'
                    f'<line x1="{x(med):.1f}" y1="{y - 3}" x2="{x(med):.1f}" y2="{y + 23}" stroke="#FFFFFF" stroke-width="2"/>'
                    f'<text class="s" x="{x(hi) + 8:.1f}" y="{y + 15}">{lo:,}–{hi:,}, median {med:,}</text>')
    for tick in (0, 5_000, 10_000, 15_000):
        body.append(f'<text class="s" x="{x(tick):.1f}" y="{44 + 3 * row + 4}" text-anchor="middle">{tick // 1000}k</text>')
    return _svg(width, 52 + 3 * row + 4, "".join(body), "Context tokens per question")


def datasets() -> str:
    rows = [("LoCoMo", "10 conversations · 1,540 questions", "19–32 sessions per conversation (27 on average)",
             "~20k-token history", 27, 20),
            ("LongMemEval_S", "500 questions, each with its own history", "39–66 sessions per question (50 on average)",
             "~103k-token history", 50, 103),
            ("DMR", "500 questions, each with its own history", "5 sessions per question",
             "~1.6k-token history", 5, 1.6)]
    width, row = 840, 78
    body = ['<text class="t" x="0" y="18">What each benchmark asks the memory to hold</text>']
    for i, (name, what, sess, hist, n, k) in enumerate(rows):
        y = 34 + i * row
        body.append(f'<text class="v" x="0" y="{y + 16}" style="font-size:15px;fill:{BLUE}">{name}</text>'
                    f'<text class="s" x="0" y="{y + 36}">{what}</text>'
                    f'<text class="s" x="0" y="{y + 54}">{sess}</text>')
        for j in range(n):
            body.append(f'<rect x="{360 + j * 7}" y="{y + 6}" width="5" height="22" rx="1.5" fill="{BLUE}" opacity=".85"/>')
        bw = 270 * k / 103
        body.append(f'<rect x="360" y="{y + 36}" width="{max(bw, 3):.1f}" height="12" rx="3" fill="{LIGHT}"/>'
                    f'<text class="s" x="{360 + max(bw, 3) + 8:.1f}" y="{y + 46}">{hist}</text>')
    body.append(f'<text class="s" x="360" y="{34 + 3 * row}">each bar one session</text>')
    return _svg(width, 40 + 3 * row, "".join(body), "Sessions and history length per benchmark")


def fairness() -> str:
    w = 840
    body = ['<text class="t" x="0" y="18">Every system answers the same questions and is graded the same way</text>']
    def box(x, y, bw, bh, title, sub="", solid=False):
        fill, ink = (BLUE, "#FFFFFF") if solid else ("#E3E8FF", "#2A3FBF")
        out = f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="12" fill="{fill}"/>'
        out += (f'<text x="{x + bw / 2}" y="{y + (bh / 2 + 5 if not sub else 26)}" text-anchor="middle" '
                f'style="font-size:15px;font-weight:700;fill:{ink}">{title}</text>')
        if sub:
            out += f'<text x="{x + bw / 2}" y="{y + 46}" text-anchor="middle" style="font-size:12.5px;fill:{ink}">{sub}</text>'
        return out
    def arrow(x1, y1, x2, y2):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{GREY}" stroke-width="1.6"/>'
                f'<path d="M{x2} {y2} l-8 -4 l0 8 z" fill="{GREY}"/>')
    body.append(box(0, 70, 200, 70, "bench loaders", "same questions, same sessions", solid=True))
    systems = ["Membase", "mem0", "Memori", "LangMem", "Zep", "Graphiti", "full context"]
    for i, name in enumerate(systems):
        y = 34 + i * 22
        body.append(f'<rect x="300" y="{y}" width="200" height="18" rx="5" fill="{"#3E61FF" if i == 0 else "#E3E8FF"}"/>'
                    f'<text x="400" y="{y + 13}" text-anchor="middle" style="font-size:12.5px;font-weight:600;'
                    f'fill:{"#FFFFFF" if i == 0 else "#2A3FBF"}">{name}</text>')
    body.append(arrow(202, 105, 294, 105))
    body.append(box(600, 70, 240, 70, "bench judges", "one per benchmark, gpt-4o-mini", solid=True))
    body.append(arrow(502, 105, 594, 105))
    body.append(f'<text class="s" x="300" y="{34 + 7 * 22 + 12}">hypotheses.jsonl: question_id, hypothesis, category, gold</text>')
    return _svg(w, 34 + 7 * 22 + 20, "".join(body), "Shared loaders, competing systems, shared judges")


def example() -> str:
    width = 840
    body = ['<text class="t" x="0" y="18">One LoCoMo question, end to end (conv-26-q3)</text>']
    # history: 19 sessions
    body.append('<text class="v" x="0" y="50">History</text>'
                '<text class="s" x="0" y="68">19 sessions · 419 turns</text>'
                '<text class="s" x="0" y="84">~16k tokens</text>')
    for i in range(19):
        x, y = (i % 5) * 34, 98 + (i // 5) * 26
        fill = BLUE if i == 1 else PALE
        body.append(f'<rect x="{x}" y="{y}" width="28" height="18" rx="4" fill="{fill}"/>')
    body.append('<text class="s" x="0" y="218">session 2 holds the gold evidence</text>')
    body.append(f'<path d="M180 140 h44" stroke="{GREY}" stroke-width="1.6"/><path d="M224 140 l-8 -4 v8 z" fill="{GREY}"/>'
                f'<text class="s" x="202" y="130" text-anchor="middle">ingest</text>')
    # episodes
    eps = [("05-08", "LGBTQ support, career plans"), ("05-25", "charity race, adoption plans"),
           ("06-27", "necklace, camping, counseling"), ("…", "19 more, one per topic")]
    body.append('<text class="v" x="236" y="50">22 dated episodes</text>'
                '<text class="s" x="236" y="68">Caroline\'s, one per topic</text>')
    for i, (d, t) in enumerate(eps):
        y = 82 + i * 34
        gold = i == 1
        body.append(f'<rect x="236" y="{y}" width="220" height="28" rx="6" fill="{BLUE if gold else PALE}"/>'
                    f'<text x="246" y="{y + 18}" style="font-size:12px;font-weight:600;fill:{"#FFFFFF" if gold else BLUE}">'
                    f'{d}</text><text x="292" y="{y + 18}" style="font-size:12px;fill:{"#FFFFFF" if gold else GREY}">{t}</text>')
    body.append(f'<path d="M462 140 h44" stroke="{GREY}" stroke-width="1.6"/><path d="M506 140 l-8 -4 v8 z" fill="{GREY}"/>'
                f'<text class="s" x="484" y="130" text-anchor="middle">search</text>')
    # question, retrieval, answer, judge
    x = 518
    body.append(f'<text class="v" x="{x}" y="50">“What did Caroline research?”</text>'
                f'<text class="s" x="{x}" y="70">20 episodes handed to the reader;</text>'
                f'<text class="s" x="{x}" y="86">the session-2 episode ranks 2nd</text>')
    body.append(f'<rect x="{x}" y="102" width="{width - x}" height="62" rx="10" fill="{PALE}"/>'
                f'<text x="{x + 12}" y="122" style="font-size:12px;font-weight:600;fill:{BLUE}">Answer</text>'
                f'<text x="{x + 12}" y="140" style="font-size:12.5px;fill:{GREY}">Adoption agencies, especially ones</text>'
                f'<text x="{x + 12}" y="156" style="font-size:12.5px;fill:{GREY}">that support LGBTQ+ individuals…</text>')
    body.append(f'<rect x="{x}" y="174" width="{width - x}" height="44" rx="10" fill="{BLUE}"/>'
                f'<text x="{x + 12}" y="194" style="font-size:12px;fill:#DCE3FF">Gold: adoption agencies</text>'
                f'<text x="{x + 12}" y="211" style="font-size:13px;font-weight:700;fill:#FFFFFF">Judge: CORRECT</text>')
    return _svg(width, 228, "".join(body), "One LoCoMo question from history to judged answer")


FIGURES = {"pipeline": pipeline, "context": context, "categories": categories,
           "misses": misses, "readers": readers, "latency": latency, "recall": recall,
           "swap": swap, "conversations": conversations, "evidence": evidence, "heatmap": heatmap,
           "spread": spread, "datasets": datasets, "fairness": fairness,
           "example": example}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (ASSETS / f"{name}.svg").write_text(fn())
        print(f"assets/{name}.svg")
