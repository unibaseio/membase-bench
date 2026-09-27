"""bench.longmemeval.judge --membase-judge path over membase_algo.testing.judge.allm_judge."""

from __future__ import annotations

import json

from membase_algo.testing import FakeLLMClient

from bench.longmemeval import judge as lme_judge


def test_membase_judge_majority_and_rendering(monkeypatch):
    replies = [json.dumps({"label": "CORRECT", "reasoning": "yes"}),
               json.dumps({"label": "WRONG", "reasoning": "no"}),
               json.dumps({"label": "CORRECT", "reasoning": "yes"})]
    fake = FakeLLMClient(responses=replies)
    monkeypatch.setattr(lme_judge, "get_llm_client", lambda model: fake)
    row = {"question_id": "q1", "gold": "Paris", "hypothesis": "It is Paris.", "category": "single-session-user"}
    meta = {"q1": {"question": "Where does the user live?", "is_abs": False}}
    out = lme_judge._judge_membase(row, model="fake-model", meta=meta, runs=3)
    assert out["label"] == "CORRECT" and out["correct"] is True and out["excluded"] is False
    assert out["judge_runs"] == [True, False, True]
    assert fake.call_count == 3
    rendered = fake.calls[0].messages[-1].content
    assert "Where does the user live?" in rendered and "Paris" in rendered and "{response}" not in rendered


def test_membase_judge_abstention_clause_and_no_context():
    fake = FakeLLMClient(responses=[json.dumps({"label": "CORRECT"})])
    import bench.longmemeval.judge as m
    orig = m.get_llm_client
    m.get_llm_client = lambda model: fake
    try:
        row = {"question_id": "q2", "gold": "n/a", "hypothesis": "I don't have that information.", "category": "x"}
        out = m._judge_membase(row, model="fake", meta={"q2": {"question": "Q?", "is_abs": True}}, runs=1)
        assert out["correct"] is True
        assert "UNANSWERABLE" in fake.calls[0].messages[-1].content
        nc = m._judge_membase({"question_id": "q3", "gold": "g", "hypothesis": m.NO_CONTEXT}, model="fake",
                               meta={}, runs=1)
        assert nc["label"] == "WRONG" and fake.call_count == 1
    finally:
        m.get_llm_client = orig
