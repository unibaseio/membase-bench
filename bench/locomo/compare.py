"""Paired comparison of judged LoCoMo runs over the questions they share."""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

_ABSTAIN = ("not answerable", "not mentioned", "no information", "not specified",
            "cannot be determined")


def _load(path: str) -> dict[str, dict]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[str(r["question_id"])] = r
    return rows


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return ((c - m) * 100, (c + m) * 100)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="judged jsonl files")
    args = ap.parse_args(argv)

    loaded = [(Path(a).stem.replace(".judged", ""), _load(a)) for a in args.arms]
    shared = set.intersection(*(set(d) for _, d in loaded))
    if not shared:
        print("No questions in common between these arms.", file=sys.stderr)
        return 1

    cats = sorted({loaded[0][1][q]["category"] for q in shared})
    name_w = max(len(n) for n, _ in loaded) + 2

    print(f"\nComparison over {len(shared)} shared questions")
    print("-" * (name_w + 12 * (len(cats) + 1) + 14))
    print(f"{'arm':<{name_w}}" + "".join(f"{c:>12s}" for c in cats)
          + f"{'micro':>12s}{'abstain':>8s}")
    print("-" * (name_w + 12 * (len(cats) + 1) + 14))

    base = None
    for name, rows in loaded:
        per = collections.Counter()
        tot = collections.Counter()
        ok = 0
        abst = 0
        for q in shared:
            r = rows[q]
            c = r["category"]
            tot[c] += 1
            hit = bool(r.get("correct"))
            per[c] += hit
            ok += hit
            if any(k in str(r.get("hypothesis", "")).lower()[:120] for k in _ABSTAIN):
                abst += 1
        micro = ok / len(shared) * 100
        cells = "".join(f"{per[c] / tot[c] * 100:>11.1f}%" for c in cats)
        mark = "" if base is None else f" ({micro - base:+.2f})"
        print(f"{name:<{name_w}}{cells}{micro:>11.2f}%{abst:>8d}{mark}")
        if base is None:
            base = micro
            lo, hi = _wilson(ok, len(shared))
            print(f"{'  95% CI':<{name_w}}{'':>{12 * len(cats)}}{f'{lo:.1f}-{hi:.1f}':>12s}")

    if len(loaded) > 1:
        print("\nPer-question flips (vs. first arm)")
        print("-" * 46)
        base_rows = loaded[0][1]
        for name, rows in loaded[1:]:
            fixed = sum(1 for q in shared
                        if not base_rows[q].get("correct") and rows[q].get("correct"))
            broke = sum(1 for q in shared
                        if base_rows[q].get("correct") and not rows[q].get("correct"))
            print(f"  {name:<{name_w}} fixed {fixed:>4d}   broke {broke:>4d}   net {fixed - broke:>+4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
