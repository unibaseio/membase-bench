"""LoCoMo loader.

Source: https://github.com/snap-research/locomo (data/locomo10.json — 10 conversations,
each with multiple sessions, 300+ turns avg, and a `qa` list).

Each conversation roughly:
  {
    "sample_id": "...",
    "conversation": {
       "session_1": [{speaker, text, time?, ...}, ...],
       "session_1_date_time": "1:56 pm on 8 May, 2023",
       "session_2": [...], "session_2_date_time": "...",
       ...
    },
    "qa": [{"question": "...", "answer": "...", "category": 1..5, "evidence": [...]}, ...]
  }
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from baselines.common.types import Instance, Question, Session

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_CACHE = Path(os.environ.get("BENCH_DATA_DIR", "data")) / "locomo10.json"

CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


def _ensure_dataset(path: Path = DEFAULT_CACHE) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as r:
        path.write_bytes(r.read())
    return path


_DATE_RE = re.compile(
    r"(?P<h>\d{1,2})(?::(?P<mm>\d{2}))?\s*(?P<ap>am|pm)?\s+on\s+"
    r"(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+),\s+(?P<y>\d{4})",
    re.IGNORECASE,
)
_MONTHS = {m: i for i, m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], start=1)}


def _norm_date(s: str | None) -> str:
    if not s:
        return "1970-01-01T00:00"
    m = _DATE_RE.search(s)
    if not m:
        return s
    h = int(m.group("h"))
    mm = int(m.group("mm") or 0)
    ap = (m.group("ap") or "").lower()
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    d = int(m.group("d"))
    mon = _MONTHS.get(m.group("mon")[:3].lower(), 1)
    y = int(m.group("y"))
    return f"{y:04d}-{mon:02d}-{d:02d}T{h:02d}:{mm:02d}"


def load(limit: int | None = None, path: str | None = None) -> list[Instance]:
    p = Path(path) if path else _ensure_dataset()
    if not p.exists():
        p = _ensure_dataset(p)
    data = json.loads(p.read_text())

    instances: list[Instance] = []
    count = 0
    for conv in data:
        sample_id = str(conv.get("sample_id", f"conv{count}"))
        sessions = _build_sessions(sample_id, conv.get("conversation", {}))
        for j, q in enumerate(conv.get("qa", [])):
            if limit is not None and count >= limit:
                return instances
            cat_id = q.get("category", 0)
            cat = CATEGORY_NAMES.get(int(cat_id) if str(cat_id).isdigit() else 0, "unknown")
            ans = q.get("answer")
            if isinstance(ans, list):
                ans = "; ".join(str(a) for a in ans)
            question = Question(
                question_id=f"{sample_id}-q{j}",
                question=str(q.get("question", "")),
                question_date=_latest_session_date(sessions),
                answer=ans,
                category=cat,
                extra={"evidence": q.get("evidence", [])},
            )
            instances.append(Instance(
                instance_id=question.question_id,
                sessions=sessions,
                question=question,
            ))
            count += 1
    return instances


def _build_sessions(sample_id: str, conv: dict[str, Any]) -> list[Session]:
    sessions: list[Session] = []
    keys = sorted(
        (k for k in conv if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )
    for sk in keys:
        date_key = f"{sk}_date_time"
        date = _norm_date(conv.get(date_key))
        turns_raw = conv.get(sk) or []
        turns = []
        for t in turns_raw:
            speaker = t.get("speaker") or t.get("role") or "user"
            text = t.get("text") or t.get("content") or ""
            # LoCoMo turns may include shared images with captions.
            # Incorporate blip_caption / query so the pipeline can
            # retrieve and reason over visual content.
            blip = t.get("blip_caption") or ""
            query = t.get("query") or ""
            if blip or query:
                img_desc = query if query else blip
                if query and blip:
                    img_desc = f"{query} — {blip}"
                text = f"[Shared image: {img_desc}] {text}".strip()
            # LoCoMo is two-speaker dialogue. Use the speaker name as the role
            # so observer/reader can attribute facts. Keep the prefix in
            # content for BM25/embedding recall on the speaker name.
            turns.append({
                "role": speaker,
                "content": f"{speaker}: {text}",
                "speaker": speaker,
            })
        sessions.append(Session(
            session_id=f"{sample_id}-{sk}",
            session_date=date,
            turns=turns,
            evidence_turn_idx=set(),
        ))
    return sessions


def _latest_session_date(sessions: list[Session]) -> str:
    return max((s.session_date for s in sessions), default="1970-01-01T00:00")
