"""Render the README figures in assets/ from the published runs' numbers."""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

BLUE, DEEP, LIGHT, PALE = "#3E61FF", "#2A3FBF", "#9DB0FF", "#E3E8FF"
NIGHT, SUB = "#12162B", "#5B6280"
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
W, PAD = 840, 24

SCORES = [("LOCOMO", 93.1), ("LONGMEMEVAL_S", 92.6), ("DMR", 92.2)]

LOCOMO = [("single_hop", 94.6), ("multi_hop", 93.6), ("temporal", 91.6), ("open_domain", 83.3)]
LONGMEMEVAL = [
    ("single-session-preference", 100.0), ("single-session-user", 98.6), ("knowledge-update", 97.4),
    ("temporal-reasoning", 92.5), ("multi-session", 88.0), ("single-session-assistant", 85.7),
]

# LongMemEval_S 100-question sample, same stores, reader swapped.
READERS = [("gpt-5.4-mini", 77), ("gpt-4.1-mini", 83), ("gpt-4.1", 86), ("gpt-5", 87),
           ("gpt-5-mini", 87), ("gpt-5.4", 89), ("gpt-5.5", 96)]

# Mean tokens per question: reader context (efficiency sample) and the full history (o200k).
CONTEXT = [("LoCoMo", 6562, 19987), ("LongMemEval_S", 8970, 102795)]

# Serial timing in seconds: (search p50, search p95, total p50, total p95).
LATENCY = [("LoCoMo", 1.67, 7.02, 8.30, 18.0), ("LongMemEval_S", 2.53, 6.11, 14.7, 30.2),
           ("DMR", 1.13, 1.71, 3.21, 6.34)]


def _svg(height: int, body: str, label: str, eyebrow: str = "", panel: bool = True) -> str:
    bg = f'<rect width="{W}" height="{height}" rx="20" fill="url(#p)"/>' if panel else ""
    if eyebrow:
        bg += _eyebrow(PAD, 38, eyebrow)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" role="img" aria-label="{label}"><defs>'
            '<linearGradient id="p" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#E9EDFF"/>'
            '<stop offset=".55" stop-color="#F6F7FD"/><stop offset="1" stop-color="#E4E1FB"/></linearGradient>'
            '<linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3E61FF"/>'
            '<stop offset="1" stop-color="#2A3FBF"/></linearGradient>'
            f'</defs><style>text{{font-family:{FONT}}}</style>{bg}{body}</svg>\n')


def _eyebrow(x, y, s):
    return (f'<text x="{x}" y="{y}" style="font-family:{MONO};font-size:12px;letter-spacing:.14em;'
            f'fill:{BLUE}">{s}</text>')


def _card(x, y, w, h, inner="", fill="#FFFFFF", rx=14):
    return (f'<rect x="{x + 2}" y="{y + 6}" width="{w}" height="{h}" rx="{rx}" fill="{DEEP}" opacity=".07"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>{inner}')


def _text(x, y, s, size=13, weight=400, color=NIGHT, anchor="start", mono=False):
    family = f"font-family:{MONO};" if mono else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{family}font-size:{size}px;'
            f'font-weight:{weight};fill:{color}">{s}</text>')


def _columns(n, gap=24):
    w = (W - 2 * PAD - (n - 1) * gap) / n
    return w, [PAD + i * (w + gap) for i in range(n)]


def results() -> str:
    body = []
    for x, (name, score) in zip([0, 288, 576], SCORES):
        body.append(f'<rect x="{x}" y="8" width="264" height="116" rx="16" fill="url(#g)"/>'
                    + _text(x + 24, 42, name, 15, 600, "#DCE3FF")
                    + f'<text x="{x + 24}" y="104" style="font-size:52px;font-weight:700;fill:#FFFFFF">{score}'
                      f'<tspan style="font-size:26px;font-weight:600">%</tspan></text>')
    return _svg(132, "".join(body), "LoCoMo 93.1%, LongMemEval_S 92.6%, DMR 92.2%", panel=False)


def _logo(cx, cy, r):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r + 9}" fill="#FFFFFF" stroke="{LIGHT}" stroke-width="2"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{BLUE}"/>'
            f'<path d="M{cx - r * .32} {cy - r * .38} v{r * .32} M{cx + r * .12} {cy - r * .38} v{r * .32}" '
            f'stroke="#FFFFFF" stroke-width="{r * .1:.1f}" stroke-linecap="round"/>'
            f'<path d="M{cx - r * .55} {cy + r * .08} q{r * .55} {r * .62} {r * 1.1} {-r * .1}" fill="none" '
            f'stroke="#FFFFFF" stroke-width="{r * .1:.1f}" stroke-linecap="round"/>')


def _flow(x1, x2, y):
    return (f'<path d="M{x1} {y} H{x2 - 8}" stroke="{LIGHT}" stroke-width="2" stroke-dasharray="5 5"/>'
            f'<path d="M{x2 - 9} {y - 6} l8 6 l-8 6" stroke="{LIGHT}" stroke-width="2" fill="none" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def example() -> str:
    cy, cap = 168, 272
    c0, c1, c2, c3 = 123, 321, 519, 730
    b = []
    # the history: a stack of sessions, one holding the evidence
    for k in range(4):
        x, y, front = c0 - 66 - (3 - k) * 10, cy - 50 - (3 - k) * 10, k == 3
        inner = ""
        if front:
            for j, (side, w) in enumerate([(0, 74), (1, 58), (0, 84), (1, 50)]):
                bx = x + 12 if side == 0 else x + 132 - 12 - w
                inner += (f'<rect x="{bx}" y="{y + 14 + j * 20}" width="{w}" height="12" rx="6" '
                          f'fill="{BLUE if j == 2 else (PALE if side == 0 else LIGHT)}"/>')
        b.append(_card(x, y, 132, 100, inner, "#FFFFFF", 12))
        if not front:
            b.append(f'<rect x="{x}" y="{y}" width="132" height="100" rx="12" fill="none" stroke="#D5DCFA" stroke-width="1.5"/>')
    b.append(_text(c0, cap, "<tspan style='font-weight:700;fill:#12162B'>19</tspan> sessions · 16k tokens",
                   13, 400, SUB, "middle"))
    # the question goes to Membase
    b.append(f'<rect x="{c1 - 112}" y="26" width="224" height="40" rx="14" fill="{BLUE}"/>'
             f'<path d="M{c1 - 8} 65 l8 9 l8 -9 z" fill="{BLUE}"/>'
             + _text(c1, 51, "What did Caroline research?", 13.5, 600, "#FFFFFF", "middle"))
    b.append(_logo(c1, cy, 34))
    b.append(_text(c1, cap, "Membase search", 13, 700, NIGHT, "middle"))
    # the episodes it hands the reader
    for j in range(3):
        y, gold = cy - 58 + j * 42, j == 1
        inner = (f'<circle cx="{c2 - 52}" cy="{y + 17}" r="5" fill="{"#FFFFFF" if gold else BLUE}"/>'
                 f'<rect x="{c2 - 38}" y="{y + 11}" width="{[84, 92, 70][j]}" height="5" rx="2.5" '
                 f'fill="{"#FFFFFF" if gold else LIGHT}"/>'
                 f'<rect x="{c2 - 38}" y="{y + 21}" width="{[56, 64, 48][j]}" height="4" rx="2" '
                 f'fill="{"#C9D3FF" if gold else PALE}"/>')
        b.append(_card(c2 - 70, y, 140, 34, inner, BLUE if gold else "#FFFFFF", 9))
    b.append(_text(c2, cap, "<tspan style='font-weight:700;fill:#12162B'>20</tspan> episodes · evidence in",
                   13, 400, SUB, "middle"))
    # the answer, judged
    ax = c3 - 82
    inner = (_text(ax + 18, cy - 20, "Adoption agencies", 14.5, 700)
             + _text(ax + 18, cy + 1, "that support LGBTQ+", 13, 400, SUB)
             + f'<rect x="{ax + 18}" y="{cy + 18}" width="104" height="26" rx="13" fill="#22A06B"/>'
             f'<path d="M{ax + 31} {cy + 31} l4 4 l8 -8" stroke="#FFFFFF" stroke-width="2.4" fill="none" '
             f'stroke-linecap="round" stroke-linejoin="round"/>'
             + _text(ax + 50, cy + 35, "CORRECT", 12, 700, "#FFFFFF"))
    b.append(_card(ax, cy - 52, 164, 112, inner))
    b.append(_text(c3, cap, "Answer, judged", 13, 700, NIGHT, "middle"))
    b.append(_flow(c0 + 76, c1 - 50, cy) + _flow(c1 + 50, c2 - 76, cy) + _flow(c2 + 76, ax - 6, cy))
    return _svg(304, "".join(b), "One LoCoMo question from history to judged answer")


def categories() -> str:
    cw, xs = _columns(2)
    top, h, rows_h = 56, 292, 6 * 36
    body = []
    for x, (name, score, rows) in zip(xs, [("LoCoMo", "93.1%", LOCOMO), ("LongMemEval_S", "92.6%", LONGMEMEVAL)]):
        inner = _text(x + 20, top + 34, name, 15, 700) + _text(x + cw - 20, top + 34, score, 15, 700, BLUE, "end")
        step = rows_h / len(rows)
        for i, (label, val) in enumerate(rows):
            y = top + 70 + i * step
            bw = cw - 40
            inner += (_text(x + 20, y, label, 13, 400, SUB) + _text(x + cw - 20, y, f"{val:.1f}%", 13, 600, NIGHT, "end")
                      + f'<rect x="{x + 20}" y="{y + 8}" width="{bw}" height="8" rx="4" fill="{PALE}"/>'
                      f'<rect x="{x + 20}" y="{y + 8}" width="{bw * val / 100:.1f}" height="8" rx="4" fill="{BLUE}"/>')
        body.append(_card(x, top, cw, h, inner))
    return _svg(top + h + PAD, "".join(body), "Accuracy by category and question type", "ACCURACY BY QUESTION TYPE")


def readers() -> str:
    top, h = 56, 230
    cw, n = W - 2 * PAD, len(READERS)
    col = (cw - 40) / n
    base, span = top + 180, 130
    inner = ""
    for i, (name, val) in enumerate(READERS):
        cx = PAD + 20 + col * (i + .5)
        bh = span * val / 100
        best = name == "gpt-5.5"
        inner += (f'<rect x="{cx - 28:.1f}" y="{base - bh:.1f}" width="56" height="{bh:.1f}" rx="8" '
                  f'fill="{BLUE if best else LIGHT}"/>'
                  + _text(f"{cx:.1f}", f"{base - bh - 10:.1f}", f"{val}%", 15, 700, BLUE if best else NIGHT, "middle")
                  + _text(f"{cx:.1f}", base + 26, name, 12.5, 600 if best else 400, NIGHT if best else SUB, "middle", True))
    body = _card(PAD, top, cw, h, inner)
    return _svg(top + h + PAD, body, "Accuracy by reader on a LongMemEval_S sample",
                "LONGMEMEVAL_S SAMPLE · SAME MEMORY, READER SWAPPED")


def context() -> str:
    cw, xs = _columns(2)
    top, h = 56, 176
    body = []
    for x, (name, ctx, full) in zip(xs, CONTEXT):
        bw = cw - 40
        inner = (_text(x + 20, top + 34, name, 15, 700)
                 + _text(x + cw - 20, top + 40, f"{full / ctx:.0f}× fewer", 28, 800, BLUE, "end"))
        for i, (label, val, color) in enumerate((("Membase context", ctx, BLUE), ("full history", full, PALE))):
            y = top + 80 + i * 46
            inner += (_text(x + 20, y, label, 13, 400, SUB) + _text(x + cw - 20, y, f"{val:,}", 13, 600, NIGHT, "end")
                      + f'<rect x="{x + 20}" y="{y + 8}" width="{bw * val / full:.1f}" height="14" rx="5" fill="{color}"/>')
        body.append(_card(x, top, cw, h, inner))
    return _svg(top + h + PAD, "".join(body), "Context tokens per question against the full history",
                "TOKENS PER QUESTION")


def latency() -> str:
    cw, xs = _columns(3)
    top, h = 56, 150
    body = []
    for x, (name, sp50, sp95, tp50, tp95) in zip(xs, LATENCY):
        inner = _text(x + 20, top + 34, name, 15, 700)
        for j, (label, p50, p95) in enumerate((("search", sp50, sp95), ("end to end", tp50, tp95))):
            lx = x + 20 + j * (cw - 40) / 2
            inner += (_text(lx, top + 68, label, 13, 400, SUB)
                      + _text(lx, top + 104, f"{p50:.1f}<tspan style='font-size:16px;font-weight:600'> s</tspan>",
                              30, 800, BLUE if j == 0 else NIGHT)
                      + _text(lx, top + 128, f"p95 {p95:.1f} s", 12.5, 400, SUB))
        body.append(_card(x, top, cw, h, inner))
    return _svg(top + h + PAD, "".join(body), "Search and total latency", "SECONDS PER QUESTION, MEDIAN")


FIGURES = {"results": results, "example": example, "categories": categories,
           "readers": readers, "context": context, "latency": latency}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (ASSETS / f"{name}.svg").write_text(fn())
        print(f"assets/{name}.svg")
