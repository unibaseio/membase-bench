"""BEAM loader.

Source: https://huggingface.co/datasets/Mohammadta/BEAM
(`data/<size>-*.parquet`, sizes 100K / 500K / 1M) and Mohammadta/BEAM-10M.
BEAM is mem0's long-context memory benchmark: per conversation a long
multi-session `chat` haystack plus `probing_questions` across 10 memory
abilities (abstention, contradiction_resolution, event_ordering,
information_extraction, instruction_following, knowledge_update,
multi_session_reasoning, preference_following, summarization,
temporal_reasoning).

Row schema (per conversation):
  {
    "conversation_id": "1",
    "conversation_seed": {category, theme, title, ...},
    "chat": [ [ {content, role, time_anchor, index, ...}, ... ], ... ],  # sessions
    "probing_questions": "<python-dict-repr str>"   # {category: [{question, ideal_response, ...}]}
    ...
  }

All probing questions of one conversation share that conversation's
haystack, so we sample whole **conversations** (every conversation
already contains all 10 categories) rather than per-question — N
conversations = N ingests covering every category. Set the sample with
``BEAM_CONVERSATIONS`` (default 6) and ``BEAM_SIZE`` (default 100K).
``instance_id`` is ``<conversation_id>-q<idx>`` so the runner's
group_key collapses every question back onto one shared haystack ingest.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
from pathlib import Path

from baselines.common.types import Instance, Question, Session

_HF_REPO = "Mohammadta/BEAM"
_HF_REPO_10M = "Mohammadta/BEAM-10M"
_SIZE_FILE = {
    "100K": "data/100K-00000-of-00001.parquet",
    "500K": "data/500K-00000-of-00001.parquet",
    "1M": "data/1M-00000-of-00001.parquet",
}
# "->-> 1,1" / "->-> 1,2" index markers BEAM appends to user turns.
_INDEX_MARKER = re.compile(r"\s*->->\s*[\d,]+\s*$")


def _size() -> str:
    return os.environ.get("BEAM_SIZE", "100K").strip()


def _cache_path(size: str) -> Path:
    base = Path(os.environ.get("UNIBASE_DATA_DIR", "bench/data"))
    return base / f"beam_{size}.parquet"


def _ensure_dataset(size: str) -> Path:
    path = _cache_path(size)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if size == "10M":
        url = f"https://huggingface.co/datasets/{_HF_REPO_10M}/resolve/main/data/10M-00000-of-00002.parquet"
    else:
        rel = _SIZE_FILE.get(size)
        if rel is None:
            raise ValueError(f"unknown BEAM_SIZE={size!r}; expected 100K/500K/1M/10M")
        url = f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/{rel}"
    import urllib.request
    with urllib.request.urlopen(url, timeout=600) as r, path.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return path


def _clean(text: str) -> str:
    return _INDEX_MARKER.sub("", (text or "").strip())


def _build_sessions(conv_id: str, chat: list) -> list[Session]:
    sessions: list[Session] = []
    for si, turns_raw in enumerate(chat or []):
        turns: list[dict] = []
        date = ""
        for t in turns_raw or []:
            if not isinstance(t, dict):
                continue
            content = _clean(t.get("content", ""))
            if not content:
                continue
            if not date:
                date = str(t.get("time_anchor") or "").strip()
            turns.append({"role": t.get("role", "user"), "content": content})
        if turns:
            sessions.append(Session(
                session_id=f"{conv_id}-s{si}",
                session_date=date or "2024-01-01",
                turns=turns, evidence_turn_idx=set(),
            ))
    return sessions


def _parse_probing(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return {}


def load(limit: int | None = None, path: str | None = None) -> list[Instance]:
    import pyarrow.parquet as pq

    size = _size()
    p = Path(path) if path else _ensure_dataset(size)
    table = pq.read_table(p)
    rows = [
        {c: table.column(c)[i].as_py() for c in table.column_names}
        for i in range(table.num_rows)
    ]

    # Sample whole conversations (each already spans all 10 categories).
    n_conv = int(os.environ.get("BEAM_CONVERSATIONS", "6"))
    seed = int(os.environ.get("BEAM_SEED", "42"))
    rng = random.Random(seed)
    if 0 < n_conv < len(rows):
        rows = rng.sample(rows, n_conv)

    instances: list[Instance] = []
    for row in rows:
        conv_id = str(row.get("conversation_id", len(instances)))
        sessions = _build_sessions(conv_id, row.get("chat") or [])
        if not sessions:
            continue
        pqd = _parse_probing(row.get("probing_questions"))
        qi = 0
        for category, qs in pqd.items():
            for q in (qs or []):
                if not isinstance(q, dict):
                    continue
                # BEAM stores the gold under a category-specific field:
                #   answer               event_ordering / information_extraction /
                #                        knowledge_update / multi_session_reasoning /
                #                        temporal_reasoning
                #   ideal_answer         contradiction_resolution
                #   ideal_summary        summarization
                #   ideal_response       abstention
                #   expected_compliance  instruction_following / preference_following
                gold = (
                    q.get("answer")
                    or q.get("ideal_answer")
                    or q.get("ideal_summary")
                    or q.get("ideal_response")
                    or q.get("expected_compliance")
                )
                question = Question(
                    question_id=f"{conv_id}-{category}-{qi}",
                    question=q.get("question", ""),
                    question_date=None,
                    answer=gold,
                    category=category,
                    extra={"difficulty": q.get("difficulty"),
                           "rubric": q.get("rubric")},
                )
                # instance_id "<conv>-q<n>" → group_key() returns "<conv>",
                # so all questions of a conversation share one ingest.
                instances.append(Instance(
                    instance_id=f"{conv_id}-q{qi}",
                    sessions=sessions, question=question,
                ))
                qi += 1
                if limit is not None and len(instances) >= limit:
                    return instances
    return instances
