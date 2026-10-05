"""Dataset dispatch for the baseline runners: the same loaders as ``bench/``, so every system
answers the same questions over the same sessions and one judge grades them all.

``BENCH_DATASET`` picks the benchmark (default ``locomo``; ``longmemeval``).
"""

from __future__ import annotations

import os

from bench.common.types import Instance


def load(limit: int | None = None, path: str | None = None) -> list[Instance]:
    name = os.environ.get("BENCH_DATASET", "locomo").strip().lower()
    if name in ("locomo", ""):
        from bench.locomo.adapter import load as _load
    elif name in ("longmemeval", "lme", "longmem"):
        from bench.longmemeval.adapter import load as _load
    else:
        raise ValueError(f"unknown BENCH_DATASET={name!r}; expected 'locomo' or 'longmemeval'")
    return _load(limit=limit, path=path)
