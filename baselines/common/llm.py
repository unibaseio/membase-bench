"""Judge and reader calls for the baselines, through the engine's provider layer.

Keeps the ``chat_text(cfg, model, system, user)`` shape the runners were written against;
``cfg`` is unused and the provider comes from the environment.
"""

from __future__ import annotations

from typing import Any

from memory.infra.llm import chat_json as _chat_json
from memory.infra.llm import chat_text as _chat_text


def chat_text(cfg: Any, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    return _chat_text(model, system, user, max_tokens=max_tokens)


def chat_json(cfg: Any, model: str, system: str, user: str, max_tokens: int = 2048) -> dict:
    return _chat_json(model, system, user, max_tokens=max_tokens)
