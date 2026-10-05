"""Judge and reader calls for the baselines, through the bench-owned OpenAI client.

The endpoint comes from the environment (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``).
"""

from __future__ import annotations

from bench.common.llm import chat_json, chat_text

__all__ = ["chat_json", "chat_text"]
