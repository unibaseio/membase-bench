"""Default model for the baseline readers and judges: ``MEMBASE_BENCH_MODEL``, else gpt-4o.

Providers come from the environment (``OPENAI_API_KEY`` etc.).
"""

from __future__ import annotations

import os


def default_model() -> str:
    return os.environ.get("MEMBASE_BENCH_MODEL", "gpt-4o")
