"""Grade retrieval-only baseline output with mem0's reader and judge."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from baselines.common.config import default_model
from baselines.locomo.mem0_eval.prompts import (
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    preprocess_answer,
)


_NAME_TO_ID = {
    "multi_hop":    1,
    "temporal":     2,
    "open_domain":  3,
    "single_hop":   4,
    "adversarial":  5,
}


def _category_id(name_or_id: str | int) -> int:
    if isinstance(name_or_id, int):
        return name_or_id
    s = str(name_or_id).strip().lower().replace("-", "_")
    if s in _NAME_TO_ID:
        return _NAME_TO_ID[s]
    for cid, dname in CATEGORY_NAMES.items():
        if s in dname.lower().replace("-", "_"):
            return cid
    return 0


def _strip_answer_marker(generated: str) -> str:
    if "ANSWER:" in generated:
        return generated.rsplit("ANSWER:", 1)[-1].strip()
    return generated.strip()


def _generate_answer(model: str, question: str, memories: list[dict],
                     reference_date: str | None) -> str:
    from baselines.common.llm import chat_text
    prompt = get_answer_generation_prompt(
        question=question, search_results=memories, reference_date=reference_date,
    )
    return _strip_answer_marker(
        chat_text(model, system="", user=prompt, max_tokens=1024)
    )


def _judge_answer(model: str, category_id: int, question: str,
                  gold: str, hypothesis: str) -> tuple[bool, str]:
    from baselines.common.llm import chat_json
    prompt = get_judge_prompt(
        category=category_id, question=question,
        answer=preprocess_answer(category_id, gold), response=hypothesis,
    )
    data = chat_json(model, system=JUDGE_SYSTEM_PROMPT, user=prompt, max_tokens=512)
    label = str(data.get("label", "")).upper().strip()
    reasoning = str(data.get("reasoning", "")).strip()
    return (label == "CORRECT"), reasoning


def _process_row(row: dict, answerer: str, judge_model: str, top_k: int) -> dict:
    qid = row.get("question_id", "")
    question = row.get("question", "")
    category = row.get("category", "")
    cat_id = _category_id(category)
    gold = row.get("gold", "") or ""
    memories = (row.get("memories") or [])[:top_k]
    reference_date = row.get("reference_date") or "2023"

    try:
        generated = _generate_answer(answerer, question, memories, reference_date)
    except Exception as exc:
        return {
            "question_id": qid, "category": category, "question": question,
            "gold": gold, "generated_answer": f"ERROR: {type(exc).__name__}: {exc}",
            "judgment": "WRONG", "correct": False, "reasoning": "answerer failed",
        }
    try:
        correct, reasoning = _judge_answer(judge_model, cat_id, question, gold, generated)
    except Exception as exc:
        return {
            "question_id": qid, "category": category, "question": question,
            "gold": gold, "generated_answer": generated,
            "judgment": "WRONG", "correct": False,
            "reasoning": f"judge failed: {exc}",
        }
    return {
        "question_id": qid, "category": category, "question": question,
        "gold": gold, "generated_answer": generated,
        "judgment": "CORRECT" if correct else "WRONG", "correct": correct,
        "reasoning": reasoning,
    }


def _summarise(rows: list[dict]) -> None:
    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    headline_cats = ("single_hop", "multi_hop", "open_domain", "temporal")
    print()
    print(f"  {'category':<14s} {'Acc':>7s}  {'n':>5s}")
    print("  " + "-" * 32)
    main = {"correct": 0, "n": 0}
    for cat in headline_cats:
        rs = by_cat.get(cat, [])
        if not rs:
            continue
        c = sum(1 for r in rs if r["correct"])
        n = len(rs)
        main["correct"] += c
        main["n"] += n
        print(f"  {cat:<14s} {100*c/n:>6.2f}  {n:>5d}")
    if main["n"]:
        print("  " + "-" * 32)
        print(f"  {'overall':<14s} {100*main['correct']/main['n']:>6.2f}  {main['n']:>5d}")
    if "adversarial" in by_cat:
        rs = by_cat["adversarial"]
        c = sum(1 for r in rs if r["correct"])
        n = len(rs)
        print(f"  {'adversarial':<14s} {100*c/n:>6.2f}  {n:>5d}  (lower = better)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("retrieval", help="JSONL with one row per question, fields: "
                                       "question_id, category, question, gold, memories[], "
                                       "reference_date (optional)")
    ap.add_argument("--out", required=True, help="JSONL output (one row per judged question)")
    ap.add_argument("--answerer-model", default=default_model())
    ap.add_argument("--judge-model", default=default_model())
    ap.add_argument("--top-k", type=int, default=200,
                    help="How many of the supplied memories to give the Reader (mem0 default: 200)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="Skip questions already present (correctly) in --out")
    args = ap.parse_args(argv)

    rows_in = [
        json.loads(line) for line in Path(args.retrieval).read_text().splitlines() if line.strip()
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    kept: list[str] = []
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("generated_answer", "")).startswith("ERROR"):
                continue
            done.add(str(rec["question_id"]))
            kept.append(line)
        rows_in = [r for r in rows_in if r.get("question_id") not in done]

    print(f"[mem0-eval] {len(rows_in)} new questions, {len(done)} resumed",
          file=sys.stderr)

    t0 = time.time()
    judged: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("w") as fout:
        for line in kept:
            fout.write(line + "\n")
        futs = {pool.submit(_process_row, r, args.answerer_model,
                            args.judge_model, args.top_k): r["question_id"]
                for r in rows_in}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="mem0-eval"):
            row = fut.result()
            judged.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[mem0-eval] wrote {len(judged)} new + {len(done)} resumed in "
          f"{time.time() - t0:.0f}s -> {out_path}", file=sys.stderr)

    all_rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    _summarise(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
