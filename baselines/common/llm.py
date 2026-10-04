"""Judge and reader calls for the baselines, through the bench-owned OpenAI client.

Keeps the ``chat_text(cfg, model, system, user)`` shape the runners were written against;
``cfg`` is unused and the endpoint comes from the environment (``OPENAI_API_KEY``,
``OPENAI_BASE_URL``).
"""

from __future__ import annotations

from typing import Any

from bench.common.llm import chat_json as _chat_json
from bench.common.llm import chat_text as _chat_text


def chat_text(cfg: Any, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    return _chat_text(model, system, user, max_tokens=max_tokens)


def chat_json(cfg: Any, model: str, system: str, user: str, max_tokens: int = 2048) -> dict:
    return _chat_json(model, system, user, max_tokens=max_tokens)
