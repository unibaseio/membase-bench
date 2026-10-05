from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    session_date: str
    turns: list[dict]


@dataclass
class Question:
    question_id: str
    question: str
    question_date: str | None
    answer: str | None
    category: str
    extra: dict


@dataclass
class Instance:
    instance_id: str
    sessions: list[Session]
    question: Question


@dataclass
class Hypothesis:
    question_id: str
    hypothesis: str
    category: str
    gold: str | None
    retrieved_sessions: list[str] = field(default_factory=list)
    retrieved_observations: int = 0
