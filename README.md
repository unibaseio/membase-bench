# membase-bench

Benchmark harnesses for [membase-algo](https://github.com/unibaseio/unibase-supermem), the
Membase memory engine: **LoCoMo**, **LongMemEval_S** and **DMR**. Every published number
comes from here, with the engine commit pinned in `pyproject.toml`.

| Benchmark | Questions | Accuracy | Reader |
|---|---|---|---|
| LoCoMo | 1,540 | 93.12% | gpt-4.1-mini |
| LongMemEval_S | 500 | 92.60% | gpt-5.5 |
| DMR | 500 | 92.20% | gpt-4o-mini |

Exact configurations, commands, costs and the ablations behind each number are in
[bench/locomo/REPRODUCE.md](bench/locomo/REPRODUCE.md).

## Setup

```bash
uv sync --extra dev          # installs the pinned engine (Python 3.12+)
export OPENAI_API_KEY=...
```

Datasets are not redistributed. `bench/locomo/adapter.py` downloads LoCoMo into `data/` on
first use; the LongMemEval and DMR loaders document where to place their files. Run outputs go
to `runs/` and engine stores to `.cache/`; both are git-ignored.

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
  common/        shared types
  locomo/        runner (ours_core.py), Memori-compatible judge, paired comparison, retrieval metrics
  longmemeval/   runner and judge (official per-type rules; --membase-judge for majority vote)
  dmr/           runner and judge under the Zep / MemGPT protocol
baselines/       competitor runners (mem0, Memori, LangMem, Zep, Graphiti, full context)
tests/           offline tests (fake LLM)
```

## License

MIT, see [LICENSE](LICENSE), except the mem0-derived evaluation code listed in [NOTICE](NOTICE).
