"""Helpers shared by the baseline runners."""

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
    return inst.instance_id


def safe_id(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)


def gold_text(inst: Instance) -> str:
    ans = inst.question.answer
    return ans if isinstance(ans, str) else json.dumps(ans)


def make_retrieval_row(inst: Instance, memories: list[dict[str, Any]] | None,
                        error: str | None = None) -> dict[str, Any]:
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


def speaker_of(turn: dict) -> str:
    return str(turn.get("speaker") or turn.get("role") or "user")


def speaker_line(turn: dict) -> str:
    speaker, content = speaker_of(turn), str(turn.get("content") or "").strip()
    return content if content.startswith(f"{speaker}: ") else f"{speaker}: {content}"


def chat_roles(inst: Instance) -> dict[str, str]:
    roles: dict[str, str] = {}
    for s in inst.sessions:
        for t in s.turns:
            speaker = speaker_of(t)
            if speaker in roles:
                continue
            low = speaker.lower()
            if low in ("user", "human"):
                roles[speaker] = "user"
            elif low in ("assistant", "ai"):
                roles[speaker] = "assistant"
            else:
                roles[speaker] = "user" if "user" not in roles.values() else "assistant"
    return roles
