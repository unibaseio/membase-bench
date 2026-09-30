"""LongMemEval loader.

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
(`longmemeval_s_cleaned.json` — 500 questions, the standard "short
haystack" split, ~115k tokens of distractor sessions per question).

Each question:
  {
    "question_id": "e47becba" | "0862e8bf_abs",   # _abs = abstention/trap
    "question_type": "multi-session" | "temporal-reasoning" |
                     "knowledge-update" | "single-session-user" |
                     "single-session-assistant" | "single-session-preference",
    "question": "...",
    "question_date": "2023/05/30 (Tue) 23:40",
    "answer": "Business Administration",
    "answer_session_ids": ["answer_280352e9"],
    "haystack_dates": ["2023/05/20 (Sat) 02:21", ...],     # one per session
    "haystack_session_ids": ["sharegpt_yywfIrx_0", ...],
    "haystack_sessions": [[{role, content, has_answer?}, ...], ...],
  }

We map each question to one `Instance`: every haystack session becomes a
`Session`, the question becomes a `Question`. The `_abs` suffix maps to
category ``abstention`` (the trap-question axis — the right answer is to
refuse); all other categories pass through as-is.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from baselines.common.types import Instance, Question, Session

# The cleaned short split. Override with BENCH_LONGMEMEVAL_PATH or pass path=.
DATA_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
DEFAULT_CACHE = Path(os.environ.get("UNIBASE_DATA_DIR", "bench/data")) / "longmemeval_s.json"

# "2023/05/30 (Tue) 23:40" -> "2023-05-30T23:40"
_DATE_RE = re.compile(
    r"(?P<y>\d{4})/(?P<mon>\d{2})/(?P<d>\d{2})\s*(?:\([A-Za-z]+\))?\s*"
    r"(?:(?P<h>\d{2}):(?P<mm>\d{2}))?"
)


def _norm_date(s: str | None) -> str:
    if not s:
        return "1970-01-01T00:00"
    m = _DATE_RE.search(s)
    if not m:
        return s
    h = m.group("h") or "00"
    mm = m.group("mm") or "00"
    return f"{m.group('y')}-{m.group('mon')}-{m.group('d')}T{h}:{mm}"


def _ensure_dataset(path: Path = DEFAULT_CACHE) -> Path:
    if path.exists():
        return path
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    # 265 MB — stream to disk.
    with urllib.request.urlopen(DATA_URL, timeout=600) as r, path.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return path


def _category(question_type: str, question_id: str) -> str:
    if question_id.endswith("_abs"):
        return "abstention"
    return question_type


def _build_sessions(qid: str, q: dict) -> list[Session]:
    sessions: list[Session] = []
    hs = q.get("haystack_sessions") or []
    dates = q.get("haystack_dates") or []
    sids = q.get("haystack_session_ids") or []
    for i, turns_raw in enumerate(hs):
        sid = sids[i] if i < len(sids) else f"{qid}-s{i}"
        date = _norm_date(dates[i] if i < len(dates) else None)
        turns: list[dict] = []
        evidence: set[int] = set()
        for ti, t in enumerate(turns_raw or []):
            content = (t.get("content") or "").strip()
            if not content:
                continue
            turns.append({"role": t.get("role", "user"), "content": content})
            if t.get("has_answer"):
                evidence.add(len(turns) - 1)
        if turns:
            sessions.append(Session(
                session_id=str(sid), session_date=date,
                turns=turns, evidence_turn_idx=evidence,
            ))
    return sessions


def load(limit: int | None = None, path: str | None = None) -> list[Instance]:
    p = Path(path) if path else _ensure_dataset()
    if not p.exists():
        p = _ensure_dataset(p)
    data = json.loads(p.read_text())

    instances: list[Instance] = []
    for q in data:
        qid = str(q["question_id"])
        sessions = _build_sessions(qid, q)
        question = Question(
            question_id=qid,
            question=q.get("question", ""),
            question_date=_norm_date(q.get("question_date")),
            answer=q.get("answer"),
            category=_category(q.get("question_type", "unknown"), qid),
            extra={"answer_session_ids": q.get("answer_session_ids", [])},
        )
        instances.append(Instance(
            instance_id=qid, sessions=sessions, question=question,
        ))
        if limit is not None and len(instances) >= limit:
            break
    return instances
