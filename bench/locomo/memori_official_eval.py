"""LoCoMo judge: Memori's notebook CORRECT/WRONG prompt."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from bench.common.llm import chat_text
from bench.locomo.adapter import load


ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def parse_judge_label(response: str) -> str | None:
    cleaned = re.sub(r"```(?:json)?\s*", "", response)
    cleaned = re.sub(r"```", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        label = str(parsed.get("label", "")).upper()
        if label in ("CORRECT", "WRONG"):
            return label
    except (json.JSONDecodeError, AttributeError):
        pass

    upper = response.upper()
    has_correct = "CORRECT" in upper
    has_wrong = "WRONG" in upper
    if has_correct and not has_wrong:
        return "CORRECT"
    if has_wrong and not has_correct:
        return "WRONG"
    return None


def f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def _question_lookup() -> dict[str, str]:
    return {
        inst.question.question_id: inst.question.question
        for inst in load(limit=None, include_adversarial=True)
    }


def _judge(row: dict, *, model: str, questions: dict[str, str]) -> dict:
    question = questions.get(row["question_id"], "")
    prompt = ACCURACY_PROMPT.format(
        question=question,
        gold_answer=row.get("gold", ""),
        generated_answer=row.get("hypothesis", ""),
    )
    response = chat_text(model, "", prompt, max_tokens=128)
    label = parse_judge_label(response)
    return {
        **row,
        "question": question,
        "judge_response": response,
        "label": label,
        "correct": label == "CORRECT",
        "f1": f1(row.get("hypothesis", ""), row.get("gold", "") or ""),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hypotheses", help="jsonl from runner")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.hypotheses).read_text().splitlines()
        if line.strip()
    ]
    questions = _question_lookup()

    judged: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_judge, r, model=args.judge_model, questions=questions) for r in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="memori-judge"):
            judged.append(fut.result())

    Path(args.out).write_text("\n".join(json.dumps(r) for r in judged))

    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for r in judged:
        by_cat[r["category"]].append(r)
    print("\nLoCoMo Memori-official judge")
    print("-" * 60)
    print(f"  {'category':20s} {'F1':>8s}  {'Acc':>8s}  {'n':>5s}")
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        f1_avg = sum(r["f1"] for r in vals) / len(vals) * 100
        acc = sum(r["correct"] for r in vals) / len(vals) * 100
        print(f"  {cat:20s} {f1_avg:7.2f}  {acc:7.2f}  {len(vals):5d}")
    print("-" * 60)
    overall_f1 = sum(r["f1"] for r in judged) / max(1, len(judged)) * 100
    overall_acc = sum(r["correct"] for r in judged) / max(1, len(judged)) * 100
    print(f"  {'overall (micro)':20s} {overall_f1:7.2f}  {overall_acc:7.2f}  {len(judged):5d}")
    if by_cat:
        macro = sum(
            sum(r["correct"] for r in vals) / len(vals) for vals in by_cat.values()
        ) / len(by_cat) * 100
        print(f"  {'overall (macro)':20s} {'':7s}  {macro:7.2f}  {len(by_cat):5d} cats")

    unreadable = sum(1 for r in judged if r["label"] is None)
    if unreadable:
        print(
            f"\n  WARNING: {unreadable} row(s) had no readable verdict and were graded "
            f"WRONG. The accuracy above is a floor, not a measurement."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
