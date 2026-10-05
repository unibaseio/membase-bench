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
| LoCoMo | 1,540 (categories 1–4) | **93.12%** | 6,562 | `gpt-4.1-mini` | Memori's CORRECT/WRONG prompt |
| LongMemEval_S | 500 | **92.60%** (95% CI 90.0–94.6) | 8,970 | `gpt-5.5` | Memori's prompt + LongMemEval's per-type rules |
| DMR | 500 | **92.20%** (95% CI 89.5–94.2) | 1,602 | `gpt-4o-mini` | MemGPT's prompt (Zep protocol) |

Episode extraction and the multi-round decider use `gpt-4.1-mini` in all three; embeddings are
`text-embedding-3-small`.

| LoCoMo category | n | Accuracy |
|---|---|---|
| multi_hop | 282 | 93.6% |
| open_domain | 96 | 83.3% |
| single_hop | 841 | 94.6% |
| temporal | 321 | 91.6% |

| LongMemEval_S type | n | Accuracy |
|---|---|---|
| knowledge-update | 78 | 97.4% |
| multi-session | 133 | 88.0% |
| single-session-assistant | 56 | 85.7% |
| single-session-preference | 30 | 100% |
| single-session-user | 70 | 98.6% |
| temporal-reasoning | 133 | 92.5% |

- **LoCoMo**: re-answering the same stores with `gpt-5.5` scores 93.18% (42 questions fixed, 41
  broken), so the cheaper `gpt-4.1-mini` stays the reader.
- **LongMemEval_S**: all 37 misses are reader misses: the gold sessions were in the context (recall
  99.95%). On a 100-question sample, `gpt-5.5` answered 96 correctly against 77–89 for six other
  OpenAI readers, which is why it reads here; it rejects `temperature=0` and runs at the default.
- **DMR**: each question's memory is 5–7 episodes, so the whole memory reaches the reader; the 39
  misses are details lost in the episode narrative. The Zep paper
  ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956), Table 1) reports, with the same
  `gpt-4o-mini` reader, 98.2% for Zep and 98.0% for the full conversation in context (with
  `gpt-4-turbo`: Zep 94.8%, MemGPT 93.4%).

Latency, one question at a time (search includes the multi-round decider's LLM calls):

| | LoCoMo (40 q) | LongMemEval_S (30 q) | DMR (30 q) |
|---|---|---|---|
| search p50 / p95 | 1.67 s / 7.02 s | 2.53 s / 6.11 s | 1.13 s / 1.71 s |
| answer p50 / p95 | 5.86 s / 10.9 s | 12.0 s / 28.1 s | 2.06 s / 4.63 s |
| total p50 / p95 | 8.30 s / 18.0 s | 14.7 s / 30.2 s | 3.21 s / 6.34 s |
| answer output tokens (mean) | 798 | 697 | 101 |
| store: episodes per unit | 32 per conversation | 54 per question | 5 per question |

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
`MEMBASE_READER_MODEL=gpt-5.5` for LongMemEval). The runners set the per-benchmark choices
themselves:

| Runner | Answer prompt | "Current Date" line | Episode and search owner |
|---|---|---|---|
| `bench.locomo.ours_core` | `answer_locomo` | off (LoCoMo has no per-question date) | each conversation's `speaker_a` |
| `bench.longmemeval.runner` | `answer_longmemeval`, 16,384-token budget | on | `user` |
| `bench.dmr.runner` | Zep's prompt, outside the engine | n/a | `A` |

`--workdir` keeps the engine stores, so a rerun (to re-answer, or with `--resume` after an
interruption) skips ingestion.

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

Analysis on judged runs and kept stores:

```bash
python -m bench.locomo.compare runs/a.judged.jsonl runs/b.judged.jsonl   # paired per-question flips
python -m bench.retrieval_metrics runs/locomo.judged.jsonl                # evidence recall, retrieval vs reader misses
python -m bench.retrieval_metrics --bench longmemeval runs/lme.judged.jsonl
python -m bench.efficiency locomo --workdir .cache/locomo                 # latency and context size; or longmemeval, dmr
```

## Judges

Each benchmark is graded the way its best-known published numbers were, so results compare with
those numbers, not across benchmarks:

- **LoCoMo**: Memori's notebook judge, a lenient CORRECT/WRONG prompt. Category 5 (adversarial)
  is dropped, as in the mem0, Zep and Memori LoCoMo figures.
- **LongMemEval_S**: Memori's CORRECT/WRONG prompt with LongMemEval's per-question-type rules
  (temporal off-by-one, knowledge update, preference rubric, abstention) prepended. An empty
  retrieval is answered `[NO_CONTEXT]` and graded wrong; unreadable verdicts leave the
  denominator. `--membase-judge --judge-runs 3` grades each answer by a majority of three calls.
- **DMR**: Zep's published harness
  ([`zep_memgpt_eval.ipynb`](https://github.com/getzep/zep-papers/blob/main/kg_architecture_agent_memory/zep_memgpt_eval.ipynb))
  verbatim: all five sessions ingested, even turns speaker A and odd turns B, question
  `self_instruct.B`, gold `self_instruct.A`, answered in A's first person with Zep's prompt, judged
  with MemGPT's prompt; answer and judge `gpt-4o-mini` at temperature 0.

## Baselines

`baselines/` runs other memory systems on the same LoCoMo and LongMemEval questions and sessions,
through the loaders above (DMR has no baseline runners): mem0, Memori, LangMem, Zep, Graphiti and
a full-context ceiling. Install the extra for the system you run, e.g. `uv sync --extra mem0` (one
environment per system keeps their dependencies apart), then see
[baselines/README.md](baselines/README.md). The same judges grade them.

## Layout

```
bench/
  common/              shared types; the OpenAI client of the LoCoMo / LongMemEval judges and the DMR reader
  locomo/              runner (ours_core.py), Memori-compatible judge, paired comparison
  longmemeval/         runner and judge (per-type rules; --membase-judge for a majority vote)
  dmr/                 runner and judge under the Zep / MemGPT protocol
  retrieval_metrics.py evidence recall and miss attribution
  efficiency.py        serial latency and context size on kept stores
baselines/             competitor runners (mem0, Memori, LangMem, Zep, Graphiti, full context)
tests/                 offline tests, no network: uv sync --extra dev && uv run pytest
```

## License

MIT, see [LICENSE](LICENSE). Evaluation prompts used verbatim from Memori, Zep and MemGPT are
listed in [NOTICE](NOTICE).
