"""bench.common.llm: request shape, temperature fallback, JSON parsing, retries (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from bench.common import llm


def _reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return _reply(out)


@pytest.fixture
def fake(monkeypatch):
    def install(*outcomes):
        completions = FakeCompletions(outcomes)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm, "_client", lambda: client)
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        llm._NO_TEMP_MODELS.clear()
        return completions

    return install


def _status_error(cls, status, message):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=req), body=None)


def test_chat_text_temperature_zero_and_no_empty_system(fake):
    c = fake("hello")
    assert llm.chat_text("gpt-4o-mini", "", "hi", max_tokens=128) == "hello"
    (call,) = c.calls
    assert call["temperature"] == 0.0 and call["max_tokens"] == 128
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_reasoning_model_drops_temperature_after_400(fake):
    err = _status_error(openai.BadRequestError, 400, "Unsupported value: 'temperature'")
    c = fake(err, "ok", "ok2")
    assert llm.chat_text("gpt-5.5", "sys", "q", max_tokens=100) == "ok"
    assert c.calls[0]["temperature"] == 0.0 and c.calls[0]["max_completion_tokens"] == 2048
    assert "temperature" not in c.calls[1]
    assert llm.chat_text("gpt-5.5", "sys", "q") == "ok2"
    assert "temperature" not in c.calls[2]  # remembered


def test_chat_json_parses_and_falls_back_to_empty(fake):
    c = fake('{"label": "CORRECT"}', "not json")
    assert llm.chat_json("gpt-4o-mini", "", "x") == {"label": "CORRECT"}
    assert c.calls[0]["response_format"] == {"type": "json_object"}
    assert c.calls[0]["messages"][0] == {"role": "system", "content": ""}
    assert llm.chat_json("gpt-4o-mini", "", "x") == {}


def test_transient_errors_are_retried(fake):
    c = fake(_status_error(openai.RateLimitError, 429, "slow down"), "done")
    assert llm.chat_text("gpt-4o-mini", "", "x") == "done"
    assert len(c.calls) == 2
