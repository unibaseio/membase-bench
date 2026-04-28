"""Loader for the Memori `advanced_augmented_memories.json` dataset.

Used by the `--memory-only` ablation: skips our own Observer and ingests
Memori's pre-extracted memories directly into the observations table, so we
can compare retrieval quality on the *same* memory set Memori reports against.

Schema (top-level):
  {
    "conversations": [
      {
        "conv_id": "...",
        "sessions": [
          {"session_id": "...", "summary": "...", "timestamp": "...",
           "memories": [{"content": "...", "context": "...", ...}, ...]}
        ]
      }
    ]
  }
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

DATA_URL = (
    "https://raw.githubusercontent.com/lborro/memori-aa-data/"
    "refs/heads/main/advanced_augmented_memories.json"
)
DEFAULT_CACHE = (
    Path(os.environ.get("UNIBASE_DATA_DIR", "data"))
    / "advanced_augmented_memories.json"
)


def _ensure_dataset(path: Path = DEFAULT_CACHE) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=120) as r:
        path.write_bytes(r.read())
    return path


def conv_id_from_instance(instance_id: str) -> str:
    """LoCoMo Instance ids are '<sample_id>-q<idx>'. Drop the trailing -q* suffix."""
    base, sep, _q = instance_id.rpartition("-q")
    return base if sep else instance_id


def load(path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return {conv_id: [observation_dict]} keyed by LoCoMo sample_id.

    Each observation dict is shaped for memory.pipeline.ingest.materialise_observations.
    Match Memori's benchmark indexing text by combining `content + context`
    instead of indexing only the terse content string:
      {"text": "<content>. context: <context>. triple: <triple>",
       "type": "fact",
       "valid_at": "<timestamp>",
       "subject": None,
       "turn_indices": [],
       "_session_id": "<augmented session id>"}
    """
    p = Path(path) if path else _ensure_dataset()
    if not p.exists():
        p = _ensure_dataset(p)
    data = json.loads(p.read_text())

    out: dict[str, list[dict[str, Any]]] = {}
    for conv in data.get("conversations", []):
        conv_id = str(conv.get("conv_id", ""))
        if not conv_id:
            continue
        bucket: list[dict[str, Any]] = []
        for sess in conv.get("sessions", []):
            sess_id = str(sess.get("session_id", ""))
            ts = str(sess.get("timestamp", "") or "")
            summary = str(sess.get("summary", "") or "").strip()
            if summary:
                bucket.append({
                    "text": f"Session {sess_id} summary: {summary}",
                    "type": "summary",
                    "valid_at": ts or None,
                    "subject": None,
                    "turn_indices": [],
                    "_session_id": sess_id,
                })
            for mem in sess.get("memories", []):
                content = str(mem.get("content", "") or "").strip()
                context = str(mem.get("context", "") or "").strip()
                if not content and not context:
                    continue
                triple = _triple_text(mem.get("triple"))
                text_parts = [content or context]
                if context and context != content:
                    text_parts.append(f"context: {context}")
                if triple:
                    text_parts.append(f"triple: {triple}")
                bucket.append({
                    "text": "\n".join(text_parts),
                    "context": context,
                    "triple": mem.get("triple"),
                    "type": "fact",
                    "valid_at": ts or None,
                    "subject": None,
                    "turn_indices": [],
                    "_session_id": sess_id,
                })
        out[conv_id] = bucket
    return out


def _triple_text(raw: object) -> str:
    if not isinstance(raw, dict):
        return ""
    subj = raw.get("subject") or {}
    obj = raw.get("object") or {}
    if not isinstance(subj, dict) or not isinstance(obj, dict):
        return ""
    s = str(subj.get("name") or "").strip()
    p = str(raw.get("predicate") or "").strip()
    o = str(obj.get("name") or "").strip()
    if not (s and p and o):
        return ""
    return f"{s} {p} {o}"
