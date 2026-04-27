"""LongMemEval autoeval: LLM judge per the upstream eval_qa.py prompt.

Reads hypotheses jsonl produced by bench/common/runner.py, judges each against
the gold answer, and reports per-category accuracy. Default judge model is
gpt-5-mini (cheap, comparable to the gpt-4o-mini / gpt-4.1-mini judges used by
peer benchmarks like memU and Memori); override via --judge-model.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from memory.infra.llm import chat_json

JUDGE_SYSTEM = (
    "You are an evaluator. Decide whether the model's answer is semantically correct "
    "given the gold answer. Be strict on factual content, lenient on phrasing."
)

JUDGE_USER_TEMPLATE = (
    "Question category: {category}\n"
    "Gold answer: {gold}\n"
    "Model answer: {hyp}\n\n"
    "RULES:\n"
    "- If the gold answer is empty / 'null' / 'None' / 'not answerable' / 'no information', "
    "the question is unanswerable. Mark correct=true if the model also refuses or says the "
    "question cannot be answered from memory (e.g. 'I don't know', 'not answerable', "
    "'no information'). Mark correct=false if the model fabricates a substantive answer.\n"
    "- Otherwise: mark correct=true iff the model answer conveys the same fact(s) as the gold "
    "(allowing paraphrase, units, and equivalent dates/numbers). Mark correct=false if the "
    "model says it doesn't know when gold is a real answer.\n"
    "- For LongMemEval categories ending in '_abs' the gold is itself a refusal and the same "
    "rule applies: correct iff the model also refuses.\n"
    "Output JSON: {{\"correct\": true|false, \"reason\": \"<one short sentence>\"}}"
)


def _judge(hyp: dict, model: str) -> dict:
    user = JUDGE_USER_TEMPLATE.format(
        category=hyp.get("category", ""),
        gold=hyp.get("gold", ""),
        hyp=hyp.get("hypothesis", ""),
    )
    out = chat_json(model, JUDGE_SYSTEM, user, max_tokens=128)
    correct = bool(out.get("correct", False))
    return {**hyp, "correct": correct, "judge_reason": out.get("reason", "")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hypotheses", help="jsonl from runner")
    ap.add_argument("--judge-model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="write judged jsonl here")
    args = ap.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.hypotheses).read_text().splitlines()
        if line.strip()
    ]
    print(f"[eval] judging {len(rows)} rows with {args.judge_model}", file=sys.stderr)

    judged: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_judge, r, args.judge_model) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs), desc="judge"):
            judged.append(f.result())

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in judged))

    by_cat: dict[str, list[bool]] = collections.defaultdict(list)
    for r in judged:
        by_cat[r["category"]].append(r["correct"])
    overall = sum(r["correct"] for r in judged) / max(1, len(judged))

    print("\nLongMemEval results")
    print("-" * 50)
    for cat in sorted(by_cat):
        v = by_cat[cat]
        print(f"  {cat:32s} {sum(v)/len(v)*100:6.2f}%  ({sum(v)}/{len(v)})")
    print("-" * 50)
    print(f"  {'overall':32s} {overall*100:6.2f}%  ({sum(r['correct'] for r in judged)}/{len(judged)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
