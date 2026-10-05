"""LoCoMo loader (snap-research/locomo)."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from bench.common.types import Instance, Question, Session

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_CACHE = Path(os.environ.get("BENCH_DATA_DIR", "data")) / "locomo10.json"

# Verified against the data: category 1 is multi-hop, 4 single-hop.
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

# Category 5 (adversarial) is left out, as in the mem0, Zep and Memori LoCoMo numbers.
EXCLUDED_CATEGORIES = frozenset({5})


def _ensure_dataset(path: Path) -> Path:
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


def load(
    limit: int | None = None,
    path: str | None = None,
    *,
    include_adversarial: bool = False,
) -> list[Instance]:
    p = _ensure_dataset(Path(path) if path else DEFAULT_CACHE)
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
            cat_int = int(cat_id) if str(cat_id).isdigit() else 0
            if not include_adversarial and cat_int in EXCLUDED_CATEGORIES:
                continue
            cat = CATEGORY_NAMES.get(cat_int, "unknown")
            ans = q.get("answer")
            if isinstance(ans, list):
                ans = "; ".join(str(a) for a in ans)
            question = Question(
                question_id=f"{sample_id}-q{j}",
                question=str(q.get("question", "")),
                question_date=_latest_session_date(sessions),
                answer=ans,
                category=cat,
                extra={
                    "evidence": q.get("evidence", []),
                    "eval_owner": str(conv.get("conversation", {}).get("speaker_a", "")),
                },
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
        date = _norm_date(conv.get(f"{sk}_date_time"))
        turns = []
        for t in conv.get(sk) or []:
            speaker = t.get("speaker") or t.get("role") or "user"
            text = t.get("text") or t.get("content") or ""
            blip = t.get("blip_caption") or ""
            query = t.get("query") or ""
            if blip or query:
                img_desc = f"{query} — {blip}" if query and blip else (query or blip)
                text = f"[Shared image: {img_desc}] {text}".strip()
            turns.append({
                "role": speaker,
                "content": f"{speaker}: {text}",
                "speaker": speaker,
            })
        sessions.append(Session(
            session_id=f"{sample_id}-{sk}",
            session_date=date,
            turns=turns,
        ))
    return sessions


def _latest_session_date(sessions: list[Session]) -> str:
    return max((s.session_date for s in sessions), default="1970-01-01T00:00")


_EVIDENCE_RE = re.compile(r"D(\d+):")


def gold_sessions(question: Question) -> set[str]:
    sample_id = question.question_id.rsplit("-q", 1)[0]
    out: set[str] = set()
    for ev in question.extra.get("evidence") or []:
        m = _EVIDENCE_RE.match(str(ev))
        if m:
            out.add(f"{sample_id}-session_{int(m.group(1))}")
    return out
