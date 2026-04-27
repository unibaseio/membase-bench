"""LongMemEval loader.

Loads from the cleaned HF dataset `xiaowu0162/longmemeval-cleaned` (splits:
longmemeval_s, longmemeval_m, longmemeval_oracle).

Each row roughly looks like:
  {
    "question_id": "...",
    "question_type": "single-session-user" | "multi-session" | "temporal-reasoning" | ...,
    "question": "...",
    "answer": "...",
    "question_date": "YYYY/MM/DD (Day) HH:MM",
    "haystack_session_ids": [...],
    "haystack_dates": ["YYYY/MM/DD (Day) HH:MM", ...],
    "haystack_sessions": [[{role, content, has_answer?}, ...], ...],
    "answer_session_ids": [...]
  }
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from bench.common.types import Instance, Question, Session


def _coerce_turn(t: Any) -> dict[str, Any]:
    """Cleaned dataset stores each turn as a JSON string."""
    if isinstance(t, dict):
        return t
    if isinstance(t, str):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            return {"role": "user", "content": t}
    return {"role": "user", "content": str(t)}


def _norm_date(s: str | None) -> str:
    """LongMemEval dates look like '2023/05/14 (Sun) 14:42'. Return ISO 'YYYY-MM-DDTHH:MM'."""
    if not s:
        return "1970-01-01T00:00"
    m = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})(?:\s*\([^)]+\))?(?:\s+(\d{1,2}):(\d{2}))?", s)
    if not m:
        return s
    y, mo, d, hh, mm = m.group(1), m.group(2), m.group(3), m.group(4) or "00", m.group(5) or "00"
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T{int(hh):02d}:{int(mm):02d}"


_SPLIT_ALIASES = {
    "longmemeval_s": "longmemeval_s_cleaned",
    "longmemeval_m": "longmemeval_m_cleaned",
    "longmemeval_oracle": "longmemeval_oracle",
}


def load(split: str = "longmemeval_s", limit: int | None = None) -> list[Instance]:
    """Stream-load to dodge a known overflow when materializing the M split."""
    from datasets import load_dataset
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    real_split = _SPLIT_ALIASES.get(split, split)
    ds = load_dataset(
        "xiaowu0162/longmemeval-cleaned",
        split=real_split,
        cache_dir=cache_dir,
        streaming=True,
    )
    instances: list[Instance] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        instances.append(_row_to_instance(row))
    return instances


def _row_to_instance(row: dict[str, Any]) -> Instance:
    sids = row.get("haystack_session_ids") or []
    dates = row.get("haystack_dates") or []
    sessions_raw = row.get("haystack_sessions") or []
    sessions: list[Session] = []
    for sid, date, turns in zip(sids, dates, sessions_raw):
        ev_idx: set[int] = set()
        clean_turns = []
        for j, t in enumerate(turns or []):
            t = _coerce_turn(t)
            if t.get("has_answer"):
                ev_idx.add(j)
            clean_turns.append({"role": t.get("role", "user"), "content": t.get("content", "")})
        sessions.append(
            Session(
                session_id=str(sid),
                session_date=_norm_date(date),
                turns=clean_turns,
                evidence_turn_idx=ev_idx,
            )
        )
    q = Question(
        question_id=str(row["question_id"]),
        question=str(row["question"]),
        question_date=_norm_date(row.get("question_date")),
        answer=row.get("answer"),
        category=str(row.get("question_type", "unknown")),
        extra={"answer_session_ids": row.get("answer_session_ids") or []},
    )
    return Instance(instance_id=q.question_id, sessions=sessions, question=q)
