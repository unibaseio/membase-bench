"""LongMemEval_S loader: one instance (own haystack, own store) per question.

Sessions are named positionally, ``session_<k>`` for ``haystack_sessions[k]``; gold cites
``answer_session_ids`` and ``gold_sessions`` inverts that mapping (first occurrence wins).
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from bench.common.types import Instance, Question, Session

DEFAULT_PATH = Path(os.environ.get("BENCH_DATA_DIR", "data")) / "longmemeval_s.json"
DATA_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s"
EVAL_OWNER = "user"

_DATE_FORMATS = ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d")
# Undated sessions fall back to a synthetic clock, as the reference does; only order matters.
_BASE = datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    m = re.match(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m:
        return datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=UTC)
    return None


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M")


def load(limit: int | None = None, path: str | None = None) -> list[Instance]:
    src = Path(path or DEFAULT_PATH)
    if not src.exists():
        raise SystemExit(f"LongMemEval_S not found at {src}; download it with\n  curl -L -o {src} {DATA_URL}")
    data = json.loads(src.read_text())
    out: list[Instance] = []
    for i, q in enumerate(data):
        if limit is not None and i >= limit:
            break
        sid_prefix = f"lme-{i}"
        sessions: list[Session] = []
        dates = q.get("haystack_dates") or []
        for k, msgs in enumerate(q.get("haystack_sessions") or []):
            dt = parse_date(dates[k] if k < len(dates) else None)
            if dt is None:
                dt = _BASE.replace(day=min(28, 1 + k % 28), month=1 + (k // 28) % 12)
            turns = [
                {"role": str(m.get("role") or "user"), "speaker": str(m.get("role") or "user"),
                 "content": str(m.get("content") or "")}
                for m in (msgs or []) if (m or {}).get("content")
            ]
            if not turns:
                continue
            sessions.append(Session(
                session_id=f"{sid_prefix}-session_{k}",
                session_date=_iso(dt),
                turns=turns,
                evidence_turn_idx=set(),
            ))
        qdt = parse_date(q.get("question_date"))
        qid = str(q.get("question_id") or i)
        question = Question(
            question_id=f"{sid_prefix}-q0",
            question=str(q.get("question") or ""),
            question_date=_iso(qdt) if qdt else None,
            answer=str(q.get("answer") or ""),
            category=str(q.get("question_type") or "unknown"),
            extra={
                "raw_question_id": qid,
                "is_abs": qid.endswith("_abs"),
                "answer_session_ids": list(q.get("answer_session_ids") or []),
                "haystack_session_ids": list(q.get("haystack_session_ids") or []),
                "eval_owner": EVAL_OWNER,
            },
        )
        out.append(Instance(instance_id=question.question_id, sessions=sessions, question=question))
    return out


def gold_sessions(question: Question) -> set[str]:
    """``answer_session_ids`` → our positional session ids; first occurrence wins."""
    prefix = question.question_id.rsplit("-q", 1)[0]
    pos: dict[str, int] = {}
    for k, s in enumerate(question.extra.get("haystack_session_ids") or []):
        pos.setdefault(s, k)
    return {f"{prefix}-session_{pos[a]}" for a in question.extra.get("answer_session_ids") or [] if a in pos}
