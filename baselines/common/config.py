"""Model defaults for the baseline runners and judges (``MEMBASE_BENCH_MODEL``, default gpt-4o).

Stands in for the SDK config these runners were written against; ``llm`` is kept for their call
signatures, and providers come from the environment (``OPENAI_API_KEY`` etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchConfig:
    answerer_model: str = "gpt-4o"
    judge_model: str = "gpt-4o"
    reader_model: str = "gpt-4o"


@dataclass(frozen=True)
class Config:
    bench: BenchConfig = field(default_factory=BenchConfig)
    llm: None = None


def load_config(_argv: list[str] | None = None) -> Config:
    model = os.environ.get("MEMBASE_BENCH_MODEL", "gpt-4o")
    return Config(bench=BenchConfig(answerer_model=model, judge_model=model, reader_model=model))
