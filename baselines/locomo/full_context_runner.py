"""Full-context ceiling: the whole conversation goes to the reader, with no retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from baselines.common.config import default_model
from baselines.common.runners import gold_text, speaker_line, stratified_sample as _stratified_sample
from bench.common.types import Hypothesis, Instance
from baselines.common.dataset import load


_PROMPT = """\
You are going to answer a question about a long conversation between two
people. The question is given first so you know what to look for; then
the full conversation; then answer.

Question (asked on {question_date}): {question}

Find the answer in the conversation below. Be concise: a few words or a
short phrase. If the question asks for a date, prefer the format that
appears in the conversation. Make your best inference from the
conversation; only answer "unknown" if there is genuinely no relevant
information at all.

Conversation:
{conversation}

Answer:"""


def _format_conversation(inst: Instance) -> str:
    blocks: list[str] = []
    for s in inst.sessions:
        blocks.append(f"--- Session @ {s.session_date} ---")
        for t in s.turns:
            blocks.append(speaker_line(t))
    return "\n".join(blocks)


def _answer(inst: Instance, model: str) -> Hypothesis:
    from openai import OpenAI
    client = OpenAI()
    convo = _format_conversation(inst)
    user = _PROMPT.format(
        conversation=convo,
        question_date=inst.question.question_date or "unknown",
        question=inst.question.question,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You answer questions about a long conversation."},
                {"role": "user", "content": user},
            ],
            max_tokens=128,
        )
        ans = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        ans = f"ERROR: {type(exc).__name__}: {exc}"
    return Hypothesis(
        question_id=inst.question.question_id,
        hypothesis=ans,
        category=inst.question.category,
        gold=gold_text(inst),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-category", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reader-model", default=default_model())
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    instances = load(limit=args.limit)
    if args.sample_per_category is not None:
        instances = _stratified_sample(instances, args.sample_per_category, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    kept_lines: list[str] = []
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("hypothesis", "")).startswith("ERROR"):
                continue
            done_ids.add(str(rec["question_id"]))
            kept_lines.append(line)
        instances = [inst for inst in instances if inst.question.question_id not in done_ids]

    print(f"[full-context] locomo: {len(instances)} instances", file=sys.stderr)

    t0 = time.time()
    results: list[Hypothesis] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("w") as fout:
        for line in kept_lines:
            fout.write(line + "\n")
        futs = {pool.submit(_answer, inst, args.reader_model): inst for inst in instances}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="full-context"):
            h = fut.result()
            results.append(h)
            fout.write(json.dumps(asdict(h)) + "\n")
            fout.flush()

    print(
        f"[full-context] wrote {len(results)} new + {len(done_ids)} resumed "
        f"in {time.time() - t0:.0f}s -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
