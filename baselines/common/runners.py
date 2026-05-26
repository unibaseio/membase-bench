"""Helpers shared by the LoCoMo runners (membase / mem0 / langmem / …).

Each runner is otherwise self-contained so a benchmark reproducer doesn't
need to depend on any membase internals — that constraint kept these
helpers from being shared earlier. The handful of functions below are
pure stdlib and don't break that property.
"""

from __future__ import annotations

import collections
import json
import random
from typing import Any

from bench.common.types import Instance


def stratified_sample(instances: list[Instance], n_per_cat: int, seed: int) -> list[Instance]:
    rng = random.Random(seed)
    by_cat: dict[str, list[Instance]] = collections.defaultdict(list)
    for inst in instances:
        by_cat[inst.question.category].append(inst)
    out: list[Instance] = []
    for cat in sorted(by_cat):
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        out.extend(pool[:n_per_cat])
    rng.shuffle(out)
    return out


def group_key(inst: Instance) -> str:
    if "-q" in inst.instance_id:
        return inst.instance_id.rsplit("-q", 1)[0]
    return "|".join(s.session_id for s in inst.sessions) or inst.instance_id


def safe_id(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)


def gold_text(inst: Instance) -> str:
    """Stringify the gold answer; LoCoMo gives both strings and lists."""
    ans = inst.question.answer
    return ans if isinstance(ans, str) else json.dumps(ans)


def make_retrieval_row(inst: Instance, memories: list[dict[str, Any]] | None,
                        error: str | None = None) -> dict[str, Any]:
    """Build the dict every retrieval-only runner emits. Keep one schema
    so `bench locomo eval` (mem0_eval) can read any of them."""
    row: dict[str, Any] = {
        "question_id": inst.question.question_id,
        "category": inst.question.category,
        "question": inst.question.question,
        "gold": gold_text(inst),
        "reference_date": inst.question.question_date,
        "memories": memories or [],
    }
    if error:
        row["error"] = error
    return row
