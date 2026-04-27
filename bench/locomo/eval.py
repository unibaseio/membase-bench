"""LoCoMo eval: per-category F1 (token-overlap) and judge-based accuracy.

The original LoCoMo paper reports F1 over QA categories. We compute both:
- F1: word-level overlap (their reference metric).
- Acc: same LLM judge as LongMemEval (parallel comparison across benches).
  Default judge model is gpt-5-mini (override via --judge-model).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from bench.longmemeval.eval import _judge


_PUNCT = set(string.punctuation)


def _normalize(s: str) -> list[str]:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return [w for w in re.split(r"\s+", s) if w]


def f1(pred: str, gold: str) -> float:
    p = _normalize(pred)
    g = _normalize(gold)
    if not p or not g:
        return 0.0
    common = collections.Counter(p) & collections.Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hypotheses", help="jsonl from runner")
    ap.add_argument("--judge-model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM judge; report F1 only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.hypotheses).read_text().splitlines()
        if line.strip()
    ]
    print(f"[eval] LoCoMo: {len(rows)} rows", file=sys.stderr)

    if not args.no_judge:
        judged: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_judge, r, args.judge_model) for r in rows]
            for fu in tqdm(as_completed(futs), total=len(futs), desc="judge"):
                judged.append(fu.result())
        rows = judged

    for r in rows:
        r["f1"] = f1(r.get("hypothesis", ""), r.get("gold", "") or "")

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in rows))

    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    print("\nLoCoMo results")
    print("-" * 60)
    print(f"  {'category':20s} {'F1':>8s}  {'Acc':>8s}  {'n':>5s}")
    for cat in sorted(by_cat):
        v = by_cat[cat]
        f1_avg = sum(r["f1"] for r in v) / len(v) * 100
        acc_avg = (sum(r.get("correct", False) for r in v) / len(v) * 100) if not args.no_judge else float("nan")
        print(f"  {cat:20s} {f1_avg:7.2f}  {acc_avg:7.2f}  {len(v):5d}")
    print("-" * 60)
    overall_f1 = sum(r["f1"] for r in rows) / max(1, len(rows)) * 100
    overall_acc = (sum(r.get("correct", False) for r in rows) / max(1, len(rows)) * 100) if not args.no_judge else float("nan")
    print(f"  {'overall':20s} {overall_f1:7.2f}  {overall_acc:7.2f}  {len(rows):5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
