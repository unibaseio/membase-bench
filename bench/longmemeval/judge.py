"""LongMemEval judge: the LoCoMo CORRECT/WRONG prompt with the benchmark's official per-type
grading rule prepended (verbatim from Membase's adapter). ``[NO_CONTEXT]`` is graded WRONG with
no model call; a row with no readable verdict after retries leaves the denominator, as the reference.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from bench.locomo.memori_official_eval import ACCURACY_PROMPT
from bench.longmemeval.adapter import load
from bench.longmemeval.runner import NO_CONTEXT
from memory.infra.llm import chat_text
from memory.infra.llm.membase_client import get_llm_client

_ABSTAIN = (
    "ADDITIONAL GRADING RULE (takes precedence): This question is UNANSWERABLE from the "
    "memories. Mark CORRECT if the generated answer correctly indicates the information is "
    "not available / incomplete / not mentioned (even if it also offers unrelated info). "
    "Mark WRONG only if it confidently answers as if the info existed."
)
_CLAUSES = {
    "temporal-reasoning": (
        "ADDITIONAL GRADING RULE (takes precedence): Do NOT penalize off-by-one errors in "
        "the number of days/weeks/months. If the question asks how many days/weeks/etc. and "
        "the generated answer is off by one (e.g. 19 vs gold 18, or 28 vs 29), still mark CORRECT."
    ),
    "knowledge-update": (
        "ADDITIONAL GRADING RULE (takes precedence): If the generated answer contains some "
        "previous/older value ALONG WITH the updated value, still mark CORRECT as long as "
        "the correct updated value is present."
    ),
    "single-session-preference": (
        "ADDITIONAL GRADING RULE (takes precedence): The gold is a rubric. Mark CORRECT as "
        "long as the answer recalls and uses the user's personal preference correctly; it "
        "need not cover every rubric point."
    ),
}


def parse_label(content: str) -> str | None:
    i, j = content.find("{"), content.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        label = str(json.loads(content[i : j + 1]).get("label", "")).strip().upper()
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in ("CORRECT", "WRONG") else None


def _judge_membase(row: dict, *, model: str, meta: dict, runs: int) -> dict:
    """Same prompt through membase's ``allm_judge``: ``runs`` parallel calls, majority vote."""
    from asgiref.sync import async_to_sync
    from membase_algo.testing.judge import allm_judge

    m = meta.get(row["question_id"], {})
    if row.get("hypothesis") == NO_CONTEXT or str(row.get("hypothesis", "")).startswith("ERROR"):
        return {
            **row,
            "question": m.get("question", ""),
            "label": "WRONG",
            "correct": False,
            "excluded": False,
        }
    clause = _ABSTAIN if m.get("is_abs") else _CLAUSES.get(row.get("category", ""), "")
    template = ACCURACY_PROMPT.replace("{generated_answer}", "{response}")
    if clause:
        template = f"{clause}\n\n{template}"
    try:
        res = async_to_sync(allm_judge)(
            question=m.get("question", ""),
            golden_answer=str(row.get("gold", "")),
            generated_answer=str(row.get("hypothesis", "")),
            judge_prompt=template,
            llm=get_llm_client(model),
            num_runs=runs,
            judge_model=model,
        )
    except Exception:  # noqa: BLE001 -- every retry failed: leaves the denominator, as below
        return {
            **row,
            "question": m.get("question", ""),
            "label": None,
            "correct": False,
            "excluded": True,
        }
    label = "CORRECT" if res.is_correct else "WRONG"
    return {
        **row,
        "question": m.get("question", ""),
        "label": label,
        "correct": res.is_correct,
        "excluded": False,
        "judge_runs": res.runs,
        "judge_reasoning": res.reasoning,
    }


def _judge(row: dict, *, model: str, meta: dict, retries: int = 4) -> dict:
    m = meta.get(row["question_id"], {})
    if row.get("hypothesis") == NO_CONTEXT or str(row.get("hypothesis", "")).startswith("ERROR"):
        return {
            **row,
            "question": m.get("question", ""),
            "label": "WRONG",
            "correct": False,
            "excluded": False,
        }
    clause = _ABSTAIN if m.get("is_abs") else _CLAUSES.get(row.get("category", ""), "")
    prompt = ACCURACY_PROMPT.format(
        question=m.get("question", ""),
        gold_answer=row.get("gold", ""),
        generated_answer=row.get("hypothesis", ""),
    )
    if clause:
        prompt = f"{clause}\n\n{prompt}"
    label = None
    for _ in range(retries):
        try:
            label = parse_label(chat_text(model, "", prompt, max_tokens=128) or "")
        except Exception:  # noqa: BLE001
            label = None
        if label:
            break
    return {
        **row,
        "question": m.get("question", ""),
        "label": label,
        "correct": label == "CORRECT",
        "excluded": label is None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hypotheses")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--membase-judge",
        action="store_true",
        help="Grade through membase_algo.testing.judge.allm_judge instead of the inline loop.",
    )
    ap.add_argument(
        "--judge-runs",
        type=int,
        default=1,
        help="With --membase-judge: independent judge calls per row, majority vote (default 3).",
    )
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.hypotheses).read_text().splitlines() if line.strip()]
    meta = {
        i.question.question_id: {
            "question": i.question.question,
            "is_abs": i.question.extra.get("is_abs", False),
        }
        for i in load()
    }
    judged: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.membase_judge:
            futs = [
                pool.submit(
                    _judge_membase, r, model=args.judge_model, meta=meta, runs=args.judge_runs
                )
                for r in rows
            ]
        else:
            futs = [pool.submit(_judge, r, model=args.judge_model, meta=meta) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs), desc="lme-judge"):
            judged.append(f.result())
    Path(args.out).write_text("\n".join(json.dumps(r) for r in judged))

    graded = [r for r in judged if not r["excluded"]]
    by = collections.defaultdict(list)
    for r in graded:
        by[r["category"]].append(r["correct"])
    print("\nLongMemEval_S judge (official per-type rules)")
    print("-" * 60)
    for cat in sorted(by):
        v = by[cat]
        print(f"  {cat:28s} {sum(v) / len(v) * 100:6.2f}%  n={len(v)}")
    print("-" * 60)
    print(
        f"  {'overall (micro)':28s} {sum(r['correct'] for r in graded) / max(1, len(graded)) * 100:6.2f}%  n={len(graded)}"
    )
    nc = sum(1 for r in judged if r.get("hypothesis") == NO_CONTEXT)
    ex = sum(1 for r in judged if r["excluded"])
    err = sum(1 for r in judged if str(r.get("hypothesis", "")).startswith("ERROR"))
    print(f"  [NO_CONTEXT]={nc}  ERROR={err}  judge-unreadable (excluded from denominator)={ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
