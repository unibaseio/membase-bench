"""Aggregate judged jsonl files across benches into a single Markdown comparison table.

Usage:
  python -m bench.report \
    --in lme_s=runs/lme_s.judged.jsonl \
    --in lme_m=runs/lme_m.judged.jsonl \
    --in locomo=runs/locomo.judged.jsonl \
    --out runs/report.md
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REFERENCE = {
    "lme_s": {
        "Supermemory (gpt-4o)": {
            "overall": 81.6,
            "single_session_user": 97.14,
            "single_session_assistant": 96.43,
            "single_session_preference": 70.00,
            "knowledge_update": 88.46,
            "temporal_reasoning": 76.69,
            "multi_session": 71.43,
        },
        "Mastra OM (gpt-4o)": {"overall": 84.23},
        "Emergence (gpt-4o)": {"overall": 86.0},
    },
    "locomo": {
        # F1 on the LoCoMo paper task with GPT-4 over the raw conversation.
        "GPT-4 baseline (LoCoMo paper)": {"overall_f1": 41.0},
        # LLM-judged accuracy reported by peer projects on LoCoMo. They use the
        # same dataset (snap-research/locomo data/locomo10.json, 10 conversations)
        # but a 4o-mini / 4.1-mini judge instead of our default gpt-5-mini —
        # treat as an indicative ceiling, not a strict apples-to-apples figure.
        "Memori (Advanced Augmentation, gpt-4.1-mini judge)": {"overall_acc": "see memorilabs.ai/benchmark"},
        "memU (LoCoMo eval, gpt-4o-mini judge)": {"overall_acc": "see NevaMind-AI/memU-experiment"},
    },
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="bench=path/to/judged.jsonl (repeatable)")
    ap.add_argument("--out", default="runs/report.md")
    args = ap.parse_args(argv)

    sections: list[str] = []
    for spec in args.inputs:
        if "=" not in spec:
            print(f"[report] bad --in {spec}", file=sys.stderr)
            continue
        name, path = spec.split("=", 1)
        rows = [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        sections.append(_format(name, rows))

    out = "\n\n".join(sections)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out)
    print(out)
    return 0


def _format(name: str, rows: list[dict]) -> str:
    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    lines = [f"## {name} ({len(rows)} questions)\n"]
    has_acc = any("correct" in r for r in rows)
    has_f1 = any("f1" in r for r in rows)
    cols = ["category", "n"] + (["accuracy"] if has_acc else []) + (["f1"] if has_f1 else [])
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for cat in sorted(by_cat):
        v = by_cat[cat]
        row = [cat, str(len(v))]
        if has_acc:
            row.append(f"{sum(r.get('correct', False) for r in v)/len(v)*100:.2f}%")
        if has_f1:
            row.append(f"{sum(r.get('f1', 0.0) for r in v)/len(v)*100:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    overall = ["**overall**", f"**{len(rows)}**"]
    if has_acc:
        overall.append(f"**{sum(r.get('correct', False) for r in rows)/len(rows)*100:.2f}%**")
    if has_f1:
        overall.append(f"**{sum(r.get('f1', 0.0) for r in rows)/len(rows)*100:.2f}**")
    lines.append("| " + " | ".join(overall) + " |")

    ref_key = name.split("_")[0] + "_" + name.split("_")[-1] if name.startswith("lme_") else name
    if ref_key in REFERENCE:
        lines.append("\n**Reference (published):**\n")
        for sys_name, scores in REFERENCE[ref_key].items():
            sc = ", ".join(f"{k}={v}" for k, v in scores.items())
            lines.append(f"- {sys_name}: {sc}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
