"""DMR (Deep Memory Retrieval) loader: MemGPT's MSC-Self-Instruct, 500 questions.

Zep's protocol: all five sessions are ingested, even-indexed messages are speaker A and odd
are B, the question is ``self_instruct["B"]`` and the gold reply is ``self_instruct["A"]``.
Zep passes no timestamps; dates are synthesised backwards from a fixed anchor.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from bench.common.types import Instance, Question, Session

DEFAULT_PATH = Path(os.environ.get("BENCH_DATA_DIR", "data")) / "dmr" / "msc_self_instruct.jsonl"
DATA_URL = "https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct/resolve/main/msc_self_instruct.jsonl"
EVAL_OWNER = "A"
_ANCHOR = datetime(2023, 11, 14, 22, 0)
_UNIT = {
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
    "week": "weeks",
    "weeks": "weeks",
    "month": "days",
    "months": "days",
    "year": "days",
    "years": "days",
}
_MULT = {"month": 30, "months": 30, "year": 365, "years": 365}


def _back(prev: dict) -> timedelta:
    n = float(prev.get("time_num") or 0)
    u = str(prev.get("time_unit") or "days").lower()
    n *= _MULT.get(u, 1)
    return timedelta(**{_UNIT.get(u, "days"): n})


def _turns(msgs: list[dict]) -> list[dict]:
    out = []
    for i, m in enumerate(msgs):
        text = str(m.get("text") or "").strip()
        if not text:
            continue
        who = "A" if i % 2 == 0 else "B"
        out.append({"role": who, "speaker": who, "content": text})
    return out


def load(limit: int | None = None, path: str | None = None) -> list[Instance]:
    src = Path(path or DEFAULT_PATH)
    if not src.exists():
        raise SystemExit(f"MSC-Self-Instruct not found at {src}; download it with\n  curl -L --create-dirs -o {src} {DATA_URL}")
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    out: list[Instance] = []
    for i, r in enumerate(rows):
        if limit is not None and i >= limit:
            break
        prefix = f"dmr-{i}"
        sessions: list[Session] = []
        prevs = r.get("previous_dialogs") or []
        # Dates run backwards from the anchor by each session's time_back.
        cursor = _ANCHOR
        dates: list[datetime] = []
        for p in reversed(prevs):
            cursor = cursor - _back(p)
            dates.append(cursor)
        dates.reverse()
        for k, p in enumerate(prevs):
            t = _turns(p.get("dialog") or [])
            if t:
                sessions.append(
                    Session(f"{prefix}-session_{k}", dates[k].strftime("%Y-%m-%dT%H:%M"), t, set())
                )
        cur = _turns(r.get("dialog") or [])
        if cur:
            sessions.append(
                Session(
                    f"{prefix}-session_{len(prevs)}", _ANCHOR.strftime("%Y-%m-%dT%H:%M"), cur, set()
                )
            )
        si = r.get("self_instruct") or {}
        out.append(
            Instance(
                instance_id=f"{prefix}-q0",
                sessions=sessions,
                question=Question(
                    question_id=f"{prefix}-q0",
                    question=str(si.get("B") or ""),
                    question_date=_ANCHOR.strftime("%Y-%m-%dT%H:%M"),
                    answer=str(si.get("A") or ""),
                    category="dmr",
                    extra={"eval_owner": EVAL_OWNER},
                ),
            )
        )
    return out
