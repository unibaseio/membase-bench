# membase-bench

Benchmark harnesses for [membase-core](https://pypi.org/project/membase-core/), the
Membase memory engine: **LoCoMo**, **LongMemEval_S** and **DMR**, plus competitor baselines on
the same data. The harness drives the engine through its public API only (`CoreMemoryEngine`
and the `MEMBASE_*` settings); judges and baselines call OpenAI through their own client
(`bench/common/llm.py`), never through the engine under test.

The harness is open source (MIT). membase-core itself is proprietary and is installed from PyPI as
compiled wheels (CPython 3.12 / 3.13 on Linux, macOS and Windows); its license allows running
these benchmarks.

## Published numbers

| Benchmark | Questions | Accuracy | Reader |
|---|---|---|---|
| LoCoMo | 1,540 | 93.12% | gpt-4.1-mini |
| LongMemEval_S | 500 | 92.60% | gpt-5.5 |
| DMR | 500 | 92.20% | gpt-4o-mini |

**These were measured on engine commit `c9d26ed`** (internal),
the pre-cleanup engine (then `unibase-supermem`, import `memory`, `SUPERMEM_*` settings). In that
engine the episode vector lane was diluted by observation, turn and session vectors in the same
index, and the reader context carried two session summaries after the episodes. membase-core
searches episodes only and packs episodes only, so the pinned engine is a different system:
**the numbers have not been re-measured on membase-core yet.** Configurations, costs and
ablations are in [bench/locomo/REPRODUCE.md](bench/locomo/REPRODUCE.md), which also records how
the published runs were produced on `c9d26ed`.

## Setup

```bash
uv sync --extra dev          # installs membase-core 0.2.0 from PyPI (Python 3.12+)
export OPENAI_API_KEY=...
```

`pyproject.toml` pins the engine version, so every result maps to one engine release.

Datasets are not redistributed. `bench/locomo/adapter.py` downloads LoCoMo into `data/` on
first use; the LongMemEval and DMR loaders document where to place their files
(`UNIBASE_DATA_DIR` overrides `data/`). Run outputs go to `runs/` and engine stores to
`.cache/`; both are git-ignored.

## Running

The engine reads its models from `MEMBASE_*` settings; the runners set the per-benchmark
answer prompt, date line and episode owner themselves.

```bash
export MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini
export MEMBASE_READER_MODEL=gpt-4.1-mini      # gpt-5.5 for LongMemEval

python -m bench.locomo.ours_core --out runs/locomo.jsonl --workdir .cache/locomo --resume
python -m bench.locomo.memori_official_eval runs/locomo.jsonl --out runs/locomo.judged.jsonl

python -m bench.longmemeval.runner --out runs/lme.jsonl --workdir .cache/lme --resume
python -m bench.longmemeval.judge runs/lme.jsonl --out runs/lme.judged.jsonl

python -m bench.dmr.runner --out runs/dmr.jsonl --workdir .cache/dmr --resume
python -m bench.dmr.judge runs/dmr.jsonl --out runs/dmr.judged.jsonl
```

Every runner takes `--limit N` for a cheap smoke run.

## Baselines

`baselines/` runs other memory systems on the same data: mem0, Memori, LangMem, Zep, Graphiti
and a full-context ceiling (LoCoMo), with loaders for LongMemEval and BEAM. Install the extra
for the system you run, e.g. `uv pip install -r pyproject.toml --extra mem0`, then see
[baselines/README.md](baselines/README.md). These runners carry their own judge copy
(`baselines/locomo/memori_official_eval.py`); grade every system with one judge before comparing
numbers.

## Layout

```
bench/
  common/        shared types and the OpenAI client used by judges and baselines
  locomo/        runner (ours_core.py), Memori-compatible judge, paired comparison, retrieval metrics
  longmemeval/   runner and judge (official per-type rules; --membase-judge for majority vote)
  dmr/           runner and judge under the Zep / MemGPT protocol
baselines/       competitor runners (mem0, Memori, LangMem, Zep, Graphiti, full context)
tests/           offline tests (no network)
```

## License

MIT, see [LICENSE](LICENSE), except the mem0-derived evaluation code listed in [NOTICE](NOTICE).
