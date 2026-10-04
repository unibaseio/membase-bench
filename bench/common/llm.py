"""Bench-owned chat client for judges, the DMR reader and the competitor baselines.

Built on the official ``openai`` SDK so grading never depends on the engine under test. It
reproduces the calling convention every published judge run used:

- temperature 0, except for models that reject it (gpt-5*, o-series): those get the default
  temperature and ``max_completion_tokens`` instead of ``max_tokens``;
- transient errors (rate limit, connection, timeout, 5xx) retried up to 8 attempts with
  randomised exponential backoff (2-90 s), logged to stderr;
- ``chat_text`` omits an empty system message; ``chat_json`` asks for a JSON object and returns
  ``{}`` when the reply does not parse.

``OPENAI_API_KEY`` and, for an OpenAI-compatible endpoint, ``OPENAI_BASE_URL`` come from the
environment, as the SDK reads them.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import threading
import time
from functools import lru_cache
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

__all__ = ["chat_json", "chat_text"]

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_ATTEMPTS = 8
_WAIT_MIN, _WAIT_MAX = 2.0, 90.0

_log = logging.getLogger("membase_bench.llm")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[llm-retry] %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.WARNING)

# Models that answered 400 to ``temperature``; later calls skip it.
_NO_TEMP_MODELS: set[str] = set()
_NO_TEMP_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI()


def _is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _create(kwargs: dict[str, Any]):
    """One completion with retries; temperature 0 unless the model rejects it."""
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return _create_with_temp0(kwargs)
        except _RETRYABLE as exc:
            if attempt == _ATTEMPTS:
                raise
            # Same envelope as tenacity's wait_random_exponential(min=2, max=90).
            wait = random.uniform(0, min(_WAIT_MAX, 2.0 ** (attempt - 1)))
            wait = max(_WAIT_MIN, wait)
            _log.warning(
                "%s attempt %d/%d: %s; retrying in %.1fs",
                kwargs.get("model"), attempt, _ATTEMPTS, type(exc).__name__, wait,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def _create_with_temp0(kwargs: dict[str, Any]):
    model = kwargs.get("model", "")
    if "temperature" in kwargs or model in _NO_TEMP_MODELS:
        return _client().chat.completions.create(**kwargs)
    try:
        return _client().chat.completions.create(**{**kwargs, "temperature": 0.0})
    except BadRequestError as exc:
        if "temperature" not in str(exc).lower():
            raise
        with _NO_TEMP_LOCK:
            _NO_TEMP_MODELS.add(model)
        return _client().chat.completions.create(**kwargs)


def chat_text(model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    """Plain completion; returns the reply text ("" when the model sends none)."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = max(max_tokens * 4, 2048)
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_tokens
    resp = _create(kwargs)
    return resp.choices[0].message.content or ""


def chat_json(model: str, system: str, user: str, max_tokens: int = 2048) -> dict[str, Any]:
    """JSON-object completion; returns ``{}`` when the reply is not valid JSON."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = max(max_tokens * 8, 8192)
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_tokens
    resp = _create(kwargs)
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}
