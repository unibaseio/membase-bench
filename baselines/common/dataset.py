"""Baselines load through bench's loaders, so every system answers the same questions."""

from __future__ import annotations

import os

from bench.common.types import Instance


def load(limit: int | None = None) -> list[Instance]:
    name = os.environ.get("BENCH_DATASET", "locomo").strip().lower()
    if name in ("locomo", ""):
        from bench.locomo.adapter import load as _load
    elif name in ("longmemeval", "lme", "longmem"):
        from bench.longmemeval.adapter import load as _load
    else:
        raise ValueError(f"unknown BENCH_DATASET={name!r}; expected 'locomo' or 'longmemeval'")
    return _load(limit=limit)
