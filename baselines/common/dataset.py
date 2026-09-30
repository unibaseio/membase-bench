"""Dataset dispatch for the LoCoMo-style runners.

Every runner used to hardcode ``from bench.locomo.adapter import load``.
To run the same runner against a different benchmark (LongMemEval, …)
without forking it, they now import ``load`` from here, and the dataset
is chosen by the ``BENCH_DATASET`` env var (default ``locomo``).

    BENCH_DATASET=locomo        membase bench locomo run --runner membase ...
    BENCH_DATASET=longmemeval   membase bench locomo run --runner membase ...

The adapters return the same ``list[Instance]`` shape, so the runners,
the stratified sampler, and the judge are all dataset-agnostic.
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
    elif name in ("beam",):
        from bench.beam.adapter import load as _load
    else:
        raise ValueError(
            f"unknown BENCH_DATASET={name!r}; expected 'locomo', 'longmemeval', or 'beam'"
        )
    return _load(limit=limit, path=path)
