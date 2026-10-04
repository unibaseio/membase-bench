"""bench.longmemeval.judge --membase-judge path: parallel runs, majority vote, prompt rendering."""

from __future__ import annotations

import json
import threading

import pytest

from bench.longmemeval import judge as lme_judge


class FakeChat:
    """Stands in for ``bench.common.llm.chat_text``: replies in order, records each call."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        with self._lock:
            self.calls.append({"model": model, "system": system, "user": user})
            return self.replies.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def test_membase_judge_majority_and_rendering(monkeypatch):
    replies = [json.dumps({"label": "CORRECT", "reasoning": "yes"}),
               json.dumps({"label": "WRONG", "reasoning": "no"}),
               json.dumps({"label": "CORRECT", "reasoning": "yes"})]
    fake = FakeChat(replies)
    monkeypatch.setattr(lme_judge, "chat_text", fake)
    row = {"question_id": "q1", "gold": "Paris", "hypothesis": "It is Paris.", "category": "single-session-user"}
    meta = {"q1": {"question": "Where does the user live?", "is_abs": False}}
    out = lme_judge._judge_membase(row, model="fake-model", meta=meta, runs=3)
    assert out["label"] == "CORRECT" and out["correct"] is True and out["excluded"] is False
    assert sorted(out["judge_runs"]) == [False, True, True]
    assert fake.call_count == 3
    rendered = fake.calls[0]["user"]
    assert fake.calls[0]["system"] == ""
    assert "Where does the user live?" in rendered and "Paris" in rendered and "{response}" not in rendered


def test_membase_judge_abstention_clause_and_no_context(monkeypatch):
    fake = FakeChat([json.dumps({"label": "CORRECT"})])
    monkeypatch.setattr(lme_judge, "chat_text", fake)
    row = {"question_id": "q2", "gold": "n/a", "hypothesis": "I don't have that information.", "category": "x"}
    out = lme_judge._judge_membase(row, model="fake", meta={"q2": {"question": "Q?", "is_abs": True}}, runs=1)
    assert out["correct"] is True
    assert "UNANSWERABLE" in fake.calls[0]["user"]
    nc = lme_judge._judge_membase({"question_id": "q3", "gold": "g", "hypothesis": lme_judge.NO_CONTEXT},
                                  model="fake", meta={}, runs=1)
    assert nc["label"] == "WRONG" and fake.call_count == 1


def test_membase_judge_retries_unparseable_then_excludes(monkeypatch):
    monkeypatch.setattr(lme_judge.time, "sleep", lambda s: None)
    fake = FakeChat(["no json here", json.dumps({"label": "WRONG"})])
    monkeypatch.setattr(lme_judge, "chat_text", fake)
    row = {"question_id": "q4", "gold": "g", "hypothesis": "h", "category": "x"}
    out = lme_judge._judge_membase(row, model="fake", meta={}, runs=1)
    assert out["label"] == "WRONG" and fake.call_count == 2

    fake = FakeChat(["nope"] * 5)
    monkeypatch.setattr(lme_judge, "chat_text", fake)
    out = lme_judge._judge_membase(row, model="fake", meta={}, runs=1)
    assert out["excluded"] is True and out["label"] is None and fake.call_count == 5


@pytest.mark.parametrize("reply,label", [('{"label": "correct"}', "CORRECT"), ("WRONG", None)])
def test_inline_parse_label(reply, label):
    assert lme_judge.parse_label(reply) == label
