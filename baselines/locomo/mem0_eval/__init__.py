"""Mem0-equivalent LoCoMo evaluation harness.

This is a port of the prompts and grading logic from
https://github.com/mem0ai/memory-benchmarks (Apache 2.0 — see LICENSE in
this directory). It scores retrieval-only runs under the *same*
conditions mem0 uses to publish their headline numbers (top_k=200,
mem0's 7-step Reader prompt, mem0's category-aware judge with partial-
credit / paraphrase / date-tolerance rules).

Why it is a separate scorer:

- The Memori-notebook judge (``bench.locomo.memori_official_eval``) is
  stricter on dates and phrasing, so a system scored under it is not
  comparable with mem0 scored under mem0's judge.
- The mem0 paper's 91.6 LoCoMo number depends heavily on its prompt
  scaffold (200 memories shown chronologically, 7-step CoT in the
  Reader, partial credit in the judge). Numbers under any other
  configuration are not directly comparable.

Vendoring instead of importing keeps the comparison reproducible without
cloning a second tree.
"""

from baselines.locomo.mem0_eval.prompts import (
    ANSWER_GENERATION_PROMPT,
    CATEGORIES_TO_EVALUATE,
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    preprocess_answer,
)

__all__ = [
    "ANSWER_GENERATION_PROMPT",
    "CATEGORIES_TO_EVALUATE",
    "CATEGORY_NAMES",
    "JUDGE_SYSTEM_PROMPT",
    "get_answer_generation_prompt",
    "get_judge_prompt",
    "preprocess_answer",
]
