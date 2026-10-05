from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    session_date: str           # ISO8601 (YYYY-MM-DD or full datetime)
    turns: list[dict]           # [{"role": ..., "speaker": ..., "content": "..."}]


@dataclass
class Question:
    question_id: str
    question: str
    question_date: str | None
    answer: str | None
    category: str
    extra: dict                 # passthrough for benchmark-specific eval (e.g. answer_session_ids)


@dataclass
class Instance:
    """One eval instance: a haystack of sessions + one question."""
    instance_id: str
    sessions: list[Session]
    question: Question


@dataclass
class Hypothesis:
    question_id: str
    hypothesis: str
    category: str
    gold: str | None
    # Retrieval provenance for scoring evidence recall without the answer model; defaulted
    # because the baseline runners record none.
    retrieved_sessions: list[str] = field(default_factory=list)
    retrieved_observations: int = 0
