<p align="center">
  <img src="assets/membase-logo.png" width="72" alt="Membase">
</p>

<h1 align="center">membase-bench</h1>

<p align="center">
  <b>Membase Benchmarks</b><br>
  Accuracy, context efficiency and latency of the Membase memory engine, measured end to end.
</p>

<p align="center">
  <a href="https://www.unibase.com/memory">Website</a> ·
  <a href="https://unibaseio.gitbook.io/unibase-docs/membase">Docs</a> ·
  <a href="https://pypi.org/project/membase-core/">membase-core</a> ·
  <a href="https://pypi.org/project/membase-ai/">membase-ai</a>
</p>

Harnesses for [membase-core](https://pypi.org/project/membase-core/), the engine of
[Membase](https://www.unibase.com/memory), Unibase's memory infrastructure for AI agents and apps:
**LoCoMo**, **LongMemEval_S** and **DMR**, plus competitor baselines on the same data. These are
the runs behind the numbers on [unibase.com/memory](https://www.unibase.com/memory). The harness
drives the engine through its public API only (`CoreMemoryEngine` and the `MEMBASE_*` settings).
Judges, the DMR reader and the baselines call OpenAI directly, never through the engine under test.

The harness is open source (MIT). membase-core itself is proprietary and is installed from PyPI as
compiled wheels (CPython 3.12 / 3.13 on Linux, macOS and Windows); its license allows running
these benchmarks.

## Results

| Benchmark | Questions | Accuracy | Context tokens per question | Reader | Judge (`gpt-4o-mini`) |
|---|---|---|---|---|---|
| LoCoMo | 1,540 (categories 1–4) | **93.12%** (run-to-run noise ≈ 0.9pp) | 6,562 | `gpt-4.1-mini` | Memori's CORRECT/WRONG prompt |
| LongMemEval_S | 500 | **92.60%** (95% CI 90.0–94.6) | 8,970 | `gpt-5.5` | Memori's prompt + LongMemEval's per-type rules |
| DMR | 500 | **92.20%** (95% CI 89.5–94.2) | 1,602 | `gpt-4o-mini` | MemGPT's prompt (Zep protocol) |

Context tokens are what the reader is given per question (mean over the efficiency sample, latency
alongside in REPRODUCE.md). Episode extraction and the multi-round decider use `gpt-4.1-mini` in
all three. For DMR, the Zep
paper ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956), Table 1) reports, with the same
`gpt-4o-mini` reader, 98.2% for Zep and 98.0% for the full conversation in context (with
`gpt-4-turbo`: Zep 94.8%, MemGPT 93.4%). Per-category tables, efficiency numbers and ablations are
in [bench/locomo/REPRODUCE.md](bench/locomo/REPRODUCE.md).

## Quick start

```bash
uv sync && source .venv/bin/activate    # or: pip install -e .   (Python 3.12 or 3.13; installs membase-core 0.2.3)
export OPENAI_API_KEY=...
export MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini MEMBASE_READER_MODEL=gpt-4.1-mini

python -m bench.locomo.ours_core --limit 1 --out runs/smoke.jsonl --workdir .cache/smoke
python -m bench.locomo.memori_official_eval runs/smoke.jsonl --out runs/smoke.judged.jsonl
```

This ingests one LoCoMo conversation and answers one question: a few minutes and well under a
dollar. Every runner takes `--limit N`. On Linux without a GPU,
`uv venv && uv pip install --torch-backend cpu -e .` avoids the multi-GB CUDA build of torch that
the engine's cross-encoder pulls in.

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
date line and episode owner themselves. `--workdir` keeps the engine stores, so a rerun (to
re-answer, or with `--resume` after an interruption) skips ingestion.

```bash
python -m bench.locomo.ours_core --out runs/locomo.jsonl --workdir .cache/locomo --workers 4 --question-workers 8 --resume
python -m bench.locomo.memori_official_eval runs/locomo.jsonl --out runs/locomo.judged.jsonl

python -m bench.longmemeval.runner --out runs/lme.jsonl --workdir .cache/lme --workers 8 --ingest-workers 16 --resume
python -m bench.longmemeval.judge runs/lme.jsonl --out runs/lme.judged.jsonl

python -m bench.dmr.runner --out runs/dmr.jsonl --workdir .cache/dmr --workers 12 --resume
python -m bench.dmr.judge runs/dmr.jsonl --out runs/dmr.judged.jsonl
```

Size of the published runs:

| | Ingest (LLM input tokens) | Wall clock |
|---|---|---|
| LoCoMo (10 conversations) | 1.8M | ≈ 20 min to build, ≈ 15 min to answer and judge |
| LongMemEval_S (500 stores) | 225M (449k per question) | ≈ 1 question/min |
| DMR (500 stores) | 12.7M | ≈ 32 min |

LongMemEval is the expensive one: every question has its own ~47-session haystack.
`--stride 5` runs every 5th question (100 of 500) as a cheaper sample.

Latency and context size, one question at a time on the stores a run kept:

```bash
python -m bench.efficiency locomo --workdir .cache/locomo --out runs/efficiency.json   # or longmemeval, dmr
```

Also in `bench/locomo/`: `compare.py` (paired per-question flips between two judged runs) and
`retrieval_metrics.py` (evidence recall and miss attribution; `--bench longmemeval` for LongMemEval).

## Judges

Each benchmark is graded the way its best-known published numbers were, so results compare with
those numbers, not across benchmarks:

- **LoCoMo**: Memori's notebook judge, a lenient CORRECT/WRONG prompt. Category 5 (adversarial)
  is dropped, as in the mem0, Zep and Memori LoCoMo figures.
- **LongMemEval_S**: Memori's CORRECT/WRONG prompt with LongMemEval's per-question-type rules
  (temporal off-by-one, knowledge update, preference rubric, abstention) prepended.
  `--membase-judge --judge-runs 3` grades each answer by a majority of three judge calls.
- **DMR**: Zep's published harness with MemGPT's judge prompt; answer and judge `gpt-4o-mini` at
  temperature 0.

## Baselines

`baselines/` runs other memory systems on the same LoCoMo and LongMemEval questions and sessions,
through the loaders above (DMR has no baseline runners): mem0, Memori, LangMem, Zep, Graphiti and a full-context ceiling. Install the extra for the
system you run, e.g. `uv sync --extra mem0` (one environment per system keeps their dependencies
apart), then see [baselines/README.md](baselines/README.md). The same judges grade them, except
`--retrieval-only` output, which mem0's own judge grades.

## Layout

```
bench/
  common/        shared types; the OpenAI client of the LoCoMo / LongMemEval judges and the DMR reader
  efficiency.py  serial latency and context size on kept stores
  locomo/        runner (ours_core.py), Memori-compatible judge, paired comparison, retrieval metrics
  longmemeval/   runner and judge (per-type rules; --membase-judge for a majority vote)
  dmr/           runner and judge under the Zep / MemGPT protocol
baselines/       competitor runners (mem0, Memori, LangMem, Zep, Graphiti, full context)
tests/           offline tests, no network: uv sync --extra dev && uv run pytest
```

## License

MIT, see [LICENSE](LICENSE), except the mem0-derived evaluation code listed in [NOTICE](NOTICE).
