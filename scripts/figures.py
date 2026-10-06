"""Render the README figures in assets/ from the published runs' numbers."""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"

BLUE, DEEP, LIGHT, PALE = "#3E61FF", "#2A3FBF", "#9DB0FF", "#E3E8FF"
NIGHT, SUB, LINE = "#12162B", "#5B6280", "#A3ACD1"
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
W, PAD = 840, 24

SCORES = [("LOCOMO", 93.1), ("LONGMEMEVAL_S", 92.6), ("DMR", 92.2)]

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

# Mean tokens per question: reader context (efficiency sample) and the full history (o200k).
CONTEXT = [("LoCoMo", 6562, 19987), ("LongMemEval_S", 8970, 102795)]

# Serial timing in seconds: (search p50, search p95, total p50, total p95).
LATENCY = [("LoCoMo", 1.67, 7.02, 8.30, 18.0), ("LongMemEval_S", 2.53, 6.11, 14.7, 30.2),
           ("DMR", 1.13, 1.71, 3.21, 6.34)]

SYSTEMS = ["Membase", "mem0", "Memori", "LangMem", "Zep", "Graphiti", "full context"]


def _svg(height: int, body: str, label: str, eyebrow: str = "", panel: bool = True) -> str:
    bg = f'<rect width="{W}" height="{height}" rx="20" fill="url(#p)"/>' if panel else ""
    if eyebrow:
        bg += _eyebrow(PAD, 38, eyebrow)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" role="img" aria-label="{label}"><defs>'
            f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{LINE}"/></marker>'
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


def _arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="1.8" '
            f'marker-end="url(#a)"/>')


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


def example() -> str:
    b = [_eyebrow(PAD, 38, "ONE QUESTION, END TO END")]
    for i, line in enumerate(["From 16k tokens", "to the right", "episodes."]):
        b.append(f'<text x="{PAD}" y="{96 + i * 44}" style="font-size:38px;font-weight:800;letter-spacing:-.02em;'
                 f'fill:{NIGHT}">{line}</text>')
    b.append(_text(PAD, 226, "LoCoMo conv-26 · 19 sessions", 14, 400, SUB))
    for i in range(19):
        x, y = PAD + (i % 10) * 28, 262 + (i // 10) * 28
        b.append(f'<rect x="{x}" y="{y}" width="22" height="20" rx="5" fill="{BLUE if i == 1 else "#D5DCFA"}"/>')
    b.append(f'<rect x="{W - PAD - 270}" y="24" width="270" height="42" rx="14" fill="{BLUE}"/>'
             + _text(W - PAD - 135, 50, "What did Caroline research?", 14, 600, "#FFFFFF", "middle"))
    cx, cw = W - PAD - 440, 440
    rows = ["LGBTQ support, career plans", "Charity race, adoption plans", "Necklace, camping, counseling"]
    inner = _text(cx + 20, 108, "20 episodes to the reader", 13, 700)
    for i, t in enumerate(rows):
        y, gold = 122 + i * 42, i == 1
        inner += (f'<rect x="{cx + 20}" y="{y}" width="{cw - 40}" height="34" rx="9" fill="{BLUE if gold else "#F3F5FD"}"/>'
                  + _text(cx + 34, y + 22, f"#{i + 1}", 12, 700, "#FFFFFF" if gold else BLUE)
                  + _text(cx + 66, y + 22, t, 13, 400, "#FFFFFF" if gold else NIGHT))
        if gold:
            inner += _text(cx + cw - 34, y + 22, "evidence", 11, 400, "#DCE3FF", "end")
    b.append(_card(cx, 82, cw, 178, inner))
    inner = (_text(cx + 20, 304, "Answer", 13, 700)
             + _text(cx + 20, 326, "Adoption agencies, especially ones that support", 13)
             + _text(cx + 20, 345, "LGBTQ+ individuals…", 13)
             + f'<rect x="{cx + 20}" y="358" width="150" height="24" rx="12" fill="#22A06B"/>'
             + _text(cx + 95, 374, "Judge: CORRECT", 12, 700, "#FFFFFF", "middle"))
    b.append(_card(cx, 278, cw, 120, inner))
    return _svg(422, "".join(b), "One LoCoMo question from history to judged answer")


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


def misses() -> str:
    cw, xs = _columns(2)
    top, h = 56, 150
    body = []
    for x, (name, ret, read) in zip(xs, MISSES):
        bw, total = cw - 40, ret + read
        inner = (_text(x + 20, top + 34, name, 15, 700)
                 + _text(x + cw - 20, top + 34, f"{total} wrong", 15, 700, BLUE, "end"))
        x0 = x + 20
        for val, color in ((ret, LIGHT), (read, BLUE)):
            if val:
                inner += f'<rect x="{x0:.1f}" y="{top + 52}" width="{bw * val / total:.1f}" height="18" rx="5" fill="{color}"/>'
                x0 += bw * val / total
        for i, (val, color, label) in enumerate(((ret, LIGHT, "not retrieved"),
                                                  (read, BLUE, "retrieved, read wrong"))):
            y = top + 100 + i * 24
            inner += (f'<rect x="{x + 20}" y="{y - 10}" width="12" height="12" rx="3" fill="{color}"/>'
                      + _text(x + 40, y, f"<tspan style='font-weight:700;fill:{NIGHT}'>{val}</tspan>  {label}", 13, 400, SUB))
        body.append(_card(x, top, cw, h, inner))
    return _svg(top + h + PAD, "".join(body), "Retrieval misses against reader misses", "WHERE THE WRONG ANSWERS COME FROM")


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


def fairness() -> str:
    top, h = 56, 172
    body = []
    mid = top + h / 2
    for x, title in ((PAD, "Same questions"), (W - PAD - 210, "Same judges")):
        body.append(_card(x, mid - 32, 210, 64, _text(x + 105, mid + 5, title, 15, 700, "#FFFFFF", "middle"), BLUE))
    cx, cw = 315, 210
    inner = ""
    for i, name in enumerate(SYSTEMS):
        y = top + 26 + i * 21
        inner += _text(cx + cw / 2, y + 4, name, 13.5, 700 if i == 0 else 400, BLUE if i == 0 else NIGHT, "middle")
    body.append(_card(cx, top, cw, h, inner))
    body.append(_arrow(PAD + 214, mid, cx - 6, mid))
    body.append(_arrow(cx + cw + 4, mid, W - PAD - 216, mid))
    return _svg(top + h + PAD, "".join(body), "Shared loaders, competing systems, shared judges",
                "SAME QUESTIONS, SAME JUDGES")


FIGURES = {"results": results, "example": example, "categories": categories, "misses": misses,
           "readers": readers, "context": context, "latency": latency, "fairness": fairness}


if __name__ == "__main__":
    for name, fn in FIGURES.items():
        (ASSETS / f"{name}.svg").write_text(fn())
        print(f"assets/{name}.svg")
