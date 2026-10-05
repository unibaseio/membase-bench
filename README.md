# membase-bench

Benchmark harnesses for [membase-core](https://pypi.org/project/membase-core/), the
Membase memory engine: **LoCoMo**, **LongMemEval_S** and **DMR**, plus competitor baselines on
the same data. The harness drives the engine through its public API only (`CoreMemoryEngine`
and the `MEMBASE_*` settings); judges and baselines call OpenAI through their own client
(`bench/common/llm.py`), never through the engine under test.

The harness is open source (MIT). membase-core itself is proprietary and is installed from PyPI as
compiled wheels (CPython 3.12 / 3.13 on Linux, macOS and Windows); its license allows running
these benchmarks.

## Results

| Benchmark | Questions | Accuracy | Reader | Judge (`gpt-4o-mini`) |
|---|---|---|---|---|
| LoCoMo | 1,540 (categories 1–4) | **93.12%** (run-to-run noise ≈ 0.9pp) | `gpt-4.1-mini` | Memori's CORRECT/WRONG prompt |
| LongMemEval_S | 500 | **92.60%** (95% CI 90.0–94.6) | `gpt-5.5` | official per-type rules |
| DMR | 500 | **92.20%** (95% CI 89.5–94.2) | `gpt-4o-mini` | MemGPT's prompt (Zep protocol) |

Episode extraction and the multi-round decider use `gpt-4.1-mini` in all three. On DMR, Zep
reports 94.8% and MemGPT 93.4% on the same 500 questions, both inside our interval. Per-category
tables, efficiency numbers and ablations are in [bench/locomo/REPRODUCE.md](bench/locomo/REPRODUCE.md).

## Quick start

```bash
uv sync                      # or: pip install -e .   (Python 3.12+; installs membase-core 0.2.1)
export OPENAI_API_KEY=...
export MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini MEMBASE_READER_MODEL=gpt-4.1-mini

python -m bench.locomo.ours_core --limit 1 --out runs/smoke.jsonl --workdir .cache/smoke
python -m bench.locomo.memori_official_eval runs/smoke.jsonl --out runs/smoke.judged.jsonl
```

This ingests one LoCoMo conversation and answers one question: a few minutes and a few cents.
Every runner takes `--limit N`.

## Datasets

Datasets are not redistributed. They live in `data/` (`BENCH_DATA_DIR` overrides it):

| Benchmark | File | Source |
|---|---|---|
| LoCoMo | `data/locomo10.json` | downloaded on first use from [snap-research/locomo](https://github.com/snap-research/locomo) |
| LongMemEval_S | `data/longmemeval_s.json` | `curl -L -o data/longmemeval_s.json https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s` |
| DMR | `data/dmr/msc_self_instruct.jsonl` | `curl -L --create-dirs -o data/dmr/msc_self_instruct.jsonl https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct/resolve/main/msc_self_instruct.jsonl` |

LongMemEval_S is the original release, not the later `longmemeval-cleaned` one.

## Full runs

The engine reads its models from `MEMBASE_*` settings (as in the quick start; set
`MEMBASE_READER_MODEL=gpt-5.5` for LongMemEval). The runners set the per-benchmark answer prompt,
date line and episode owner themselves. `--workdir` keeps the engine stores, so a rerun with
`--resume` or a second judge pass skips ingestion.

```bash
python -m bench.locomo.ours_core --out runs/locomo.jsonl --workdir .cache/locomo --workers 4 --question-workers 8 --resume
python -m bench.locomo.memori_official_eval runs/locomo.jsonl --out runs/locomo.judged.jsonl

python -m bench.longmemeval.runner --out runs/lme.jsonl --workdir .cache/lme --workers 8 --ingest-workers 16 --resume
python -m bench.longmemeval.judge runs/lme.jsonl --out runs/lme.judged.jsonl

python -m bench.dmr.runner --out runs/dmr.jsonl --workdir .cache/dmr --workers 12 --resume
python -m bench.dmr.judge runs/dmr.jsonl --out runs/dmr.judged.jsonl
```

Cost and time of the published runs (OpenAI list prices):

| | Build stores | Answer + judge | Wall clock |
|---|---|---|---|
| LoCoMo (10 conversations) | ≈ $5 | ≈ $8 | ≈ 20 min + 15 min |
| LongMemEval_S (500 stores) | ≈ $0.50 per question | ≈ $0.02 per question | ≈ 1 question/min |
| DMR (500 stores) | ≈ $8 in total | | ≈ 32 min |

LongMemEval is the expensive one: every question has its own ~47-session haystack.
`--stride 5` runs every 5th question (100 of 500) as a cheaper sample.

Also in `bench/locomo/`: `compare.py` (paired per-question flips between two judged runs) and
`retrieval_metrics.py` (evidence recall and miss attribution; `--bench longmemeval` for LongMemEval).

## Judges

Each benchmark is graded the way its best-known published numbers were, so results compare with
those numbers, not across benchmarks:

- **LoCoMo**: Memori's notebook judge, a lenient CORRECT/WRONG prompt. Category 5 (adversarial)
  is dropped, as in every published LoCoMo figure.
- **LongMemEval_S**: the official per-question-type rules; `--membase-judge` grades by majority vote.
- **DMR**: Zep's published harness with MemGPT's judge prompt; answer and judge `gpt-4o-mini` at
  temperature 0.

## Baselines

`baselines/` runs other memory systems on the same questions and sessions, through the loaders
above: mem0, Memori, LangMem, Zep, Graphiti and a full-context ceiling. Install the extra for the
system you run, e.g. `uv pip install -r pyproject.toml --extra mem0` (one at a time: their
dependency trees clash), then see [baselines/README.md](baselines/README.md). The same judges
grade them.

## Layout

```
bench/
  common/        shared types and the OpenAI client used by judges and baselines
  locomo/        runner (ours_core.py), Memori-compatible judge, paired comparison, retrieval metrics
  longmemeval/   runner and judge (official per-type rules; --membase-judge for majority vote)
  dmr/           runner and judge under the Zep / MemGPT protocol
baselines/       competitor runners (mem0, Memori, LangMem, Zep, Graphiti, full context)
tests/           offline tests, no network: uv sync --extra dev && pytest
```

## License

MIT, see [LICENSE](LICENSE), except the mem0-derived evaluation code listed in [NOTICE](NOTICE).
