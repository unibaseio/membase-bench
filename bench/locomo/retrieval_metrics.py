"""Evidence-recall metrics for LoCoMo, scored from gold session ids with no LLM call.

On a judged file, wrong answers are also split into retrieval misses (gold session never
reached the reader) and reader misses (gold session was in the context).

    python -m bench.locomo.retrieval_metrics runs/<arm>.judged.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import importlib


def _gold_lookup(bench: str = "locomo") -> dict[str, set[str]]:
    """Gold session ids per question for the named benchmark."""
    ad = importlib.import_module(f"bench.{bench}.adapter")
    # Category 5 included: this is a lookup table, not the question set.
    kwargs = {"include_adversarial": True} if bench == "locomo" else {}
    return {
        inst.question.question_id: ad.gold_sessions(inst.question)
        for inst in ad.load(limit=None, **kwargs)
    }


def score(rows: list[dict], bench: str = "locomo") -> dict:
    gold_map = _gold_lookup(bench)

    per_cat: dict[str, list[float]] = collections.defaultdict(list)
    recalls: list[float] = []
    full_hits = 0
    scored = 0
    skipped_no_gold = 0
    skipped_no_provenance = 0
    # Only meaningful on a judged file; stays zeroed otherwise.
    split = {"retrieval_miss": 0, "reader_miss": 0, "correct_with_gold": 0,
             "correct_without_gold": 0}
    judged = False

    for r in rows:
        qid = str(r.get("question_id", ""))
        gold = gold_map.get(qid) or set()
        if not gold:
            # An unlabelled question says nothing about retrieval; skip rather than score 0.
            skipped_no_gold += 1
            continue
        if "retrieved_sessions" not in r:
            # Rows predating provenance are counted, never scored as an empty retrieval.
            skipped_no_provenance += 1
            continue

        got = set(r.get("retrieved_sessions") or [])
        hit = len(got & gold)
        recall = hit / len(gold)
        recalls.append(recall)
        per_cat[str(r.get("category", "?"))].append(recall)
        has_all = hit == len(gold)
        full_hits += has_all
        scored += 1

        if "correct" in r:
            judged = True
            correct = bool(r.get("correct"))
            if correct:
                split["correct_with_gold" if has_all else "correct_without_gold"] += 1
            else:
                split["reader_miss" if has_all else "retrieval_miss"] += 1

    return {
        "scored": scored,
        "skipped_no_gold": skipped_no_gold,
        "skipped_no_provenance": skipped_no_provenance,
        "recall": (sum(recalls) / scored) if scored else 0.0,
        "full_recall_rate": (full_hits / scored) if scored else 0.0,
        "per_category": {
            k: {"recall": sum(v) / len(v), "n": len(v)} for k, v in sorted(per_cat.items())
        },
        "judged": judged,
        "split": split,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hypotheses", help="jsonl from ours_core, or the judged file")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--bench", default="locomo", choices=["locomo", "longmemeval"])
    args = ap.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.hypotheses).read_text().splitlines()
        if line.strip()
    ]
    s = score(rows, bench=args.bench)

    if args.json:
        print(json.dumps(s, indent=2))
        return 0

    print(f"\n{args.bench} evidence recall (no LLM calls)")
    print("-" * 60)
    print(f"  {'category':20s} {'recall':>8s}  {'n':>5s}")
    for cat, v in s["per_category"].items():
        print(f"  {cat:20s} {v['recall'] * 100:7.2f}  {v['n']:5d}")
    print("-" * 60)
    print(f"  {'overall':20s} {s['recall'] * 100:7.2f}  {s['scored']:5d}")
    print(f"  {'all-gold-present':20s} {s['full_recall_rate'] * 100:7.2f}%")
    if s["skipped_no_gold"]:
        print(f"  skipped (no gold evidence):   {s['skipped_no_gold']}")
    if s["skipped_no_provenance"]:
        print(
            f"  skipped (run predates provenance): {s['skipped_no_provenance']}"
            "  <- re-run to score these"
        )

    if s["judged"]:
        sp = s["split"]
        wrong = sp["retrieval_miss"] + sp["reader_miss"]
        print("\n  Attribution of WRONG answers")
        print("-" * 60)
        if wrong:
            print(
                f"  retrieval miss (gold absent): {sp['retrieval_miss']:5d}"
                f"  ({sp['retrieval_miss'] / wrong * 100:.1f}% of wrong)"
            )
            print(
                f"  reader miss (gold present):   {sp['reader_miss']:5d}"
                f"  ({sp['reader_miss'] / wrong * 100:.1f}% of wrong)"
            )
        print(f"  correct, all gold present:    {sp['correct_with_gold']:5d}")
        # LoCoMo's gold evidence is not always the only place an answer appears.
        print(f"  correct, gold incomplete:     {sp['correct_without_gold']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
