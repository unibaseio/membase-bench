"""DMR grader: Zep's/MemGPT's prompt verbatim, gpt-4o-mini, structured true/false.

An ERROR hypothesis is graded false without a model call and kept in the denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

SYSTEM = (
    "You are an expert grader that determines if answers to questions match a gold standard answer"
)
PROMPT = """I will give you a question, a correct answer, and a response from a model. Please answer true if the response contains the correct answer or touches on the same topic. Otherwise, answer false. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer true. If the response only contains a subset of the information required by the answer, answer false.

<QUESTION>
B: {question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
A: {response}
</RESPONSE>
"""


class Grade(BaseModel):
    is_correct: bool = Field(description="Whether or not the response is correct")


def _judge(row: dict, model: str, questions: dict[str, str], client: OpenAI) -> dict:
    hyp = str(row.get("hypothesis", ""))
    if hyp.startswith("ERROR"):
        return {**row, "correct": False, "excluded": False}
    for _ in range(3):
        try:
            r = client.beta.chat.completions.parse(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            question=questions.get(row["question_id"], ""),
                            gold_answer=row.get("gold", ""),
                            response=hyp,
                        ),
                    },
                ],
                response_format=Grade,
            )
            g = r.choices[0].message.parsed
            return {**row, "correct": bool(g.is_correct), "excluded": False}
        except Exception:  # noqa: BLE001
            continue
    return {**row, "correct": False, "excluded": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hypotheses")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    from bench.dmr.adapter import load

    questions = {i.question.question_id: i.question.question for i in load()}
    rows = [json.loads(line) for line in Path(a.hypotheses).read_text().splitlines() if line.strip()]
    client = OpenAI()
    judged = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_judge, r, a.judge_model, questions, client) for r in rows]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="dmr-judge"):
            judged.append(fu.result())
    Path(a.out).write_text("\n".join(json.dumps(r) for r in judged))
    graded = [r for r in judged if not r["excluded"]]
    k = sum(r["correct"] for r in graded)
    n = len(graded)
    err = sum(str(r.get("hypothesis", "")).startswith("ERROR") for r in judged)
    print(f"\nDMR (MSC-Self-Instruct) — Zep/MemGPT grader, {a.judge_model}")
    print("-" * 60)
    print(
        f"  accuracy {k / max(n, 1) * 100:.2f}%  ({k}/{n})   ERROR={err}  excluded={len(judged) - n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
