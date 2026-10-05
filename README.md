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

[Membase](https://www.unibase.com/memory) is Unibase's memory infrastructure for AI agents and
apps. This repository measures its engine, [membase-core](https://pypi.org/project/membase-core/),
on three long-term memory benchmarks — **LoCoMo**, **LongMemEval_S** and **DMR** — and runs
competing memory systems on the same questions. It produced the numbers published on
[unibase.com/memory](https://www.unibase.com/memory), and anyone with an OpenAI key can rerun it.

## Results

| Benchmark | Questions | Accuracy | Context tokens per question |
|---|---|---|---|
| LoCoMo | 1,540 | **93.12%** | 6,562 |
| LongMemEval_S | 500 | **92.60%** (95% CI 90.0–94.6) | 8,970 |
| DMR | 500 | **92.20%** (95% CI 89.5–94.2) | 1,602 |

- **Retrieval finds the evidence.** On LongMemEval_S the gold sessions were in the context for
  every one of the 37 misses (recall 99.95%): what is left is the reader's. On LoCoMo a stronger
  reader does not move the score (`gpt-5.5` 93.18% against `gpt-4.1-mini` 93.12%, 42 questions
  fixed and 41 broken).
- **The reader sees a few thousand tokens.** Retrieval hands it the relevant episodes, not the
  conversation history.
- **The cost is search latency.** Search runs a multi-round LLM decider, about 1.1–2.5 s at the
  median (table below).

<details>
<summary>Accuracy by category and type, and latency</summary>

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

Serial timing, one question at a time:

| | LoCoMo (40 q) | LongMemEval_S (30 q) | DMR (30 q) |
|---|---|---|---|
| search p50 / p95 | 1.67 s / 7.02 s | 2.53 s / 6.11 s | 1.13 s / 1.71 s |
| answer p50 / p95 | 5.86 s / 10.9 s | 12.0 s / 28.1 s | 2.06 s / 4.63 s |
| total p50 / p95 | 8.30 s / 18.0 s | 14.7 s / 30.2 s | 3.21 s / 6.34 s |
| answer output tokens (mean) | 798 | 697 | 101 |
| episodes in the store | 32 per conversation | 54 per question | 5 per question |

</details>

## How it is measured

Every question goes through the same four steps, using only membase-core's public API
(`CoreMemoryEngine` and the `MEMBASE_*` settings):

1. **Ingest.** The question's conversation history is stored; the engine splits each session into
   topic-coherent cells and writes one dated episode per cell (`gpt-4.1-mini`).
2. **Search.** Multi-round retrieval: BM25 and vector recall over the episodes, fused per
   sub-query, and an LLM decider (`gpt-4.1-mini`) that picks the core episodes over up to three
   rounds.
3. **Answer.** A reader model answers from the selected episodes.
4. **Judge.** An LLM judge (`gpt-4o-mini`) grades the answer against the gold answer.

Each benchmark keeps the protocol of its best-known published numbers, so results compare with
those numbers:

| | LoCoMo | LongMemEval_S | DMR |
|---|---|---|---|
| Questions | categories 1–4; category 5 (adversarial) dropped, as in the mem0, Zep and Memori figures | all 500 | all 500 |
| Memory per question | one store per conversation, episodes of `speaker_a` | one store per question, episodes of `user` | one store per question, all five sessions, episodes of speaker A |
| Reader | `gpt-4.1-mini`, `answer_locomo` prompt | `gpt-5.5`, `answer_longmemeval` prompt, a "Current Date" line | `gpt-4o-mini` with Zep's prompt, outside the engine |
| Judge | Memori's lenient CORRECT/WRONG prompt | Memori's prompt with LongMemEval's per-type rules (temporal off-by-one, knowledge update, preference rubric, abstention) | MemGPT's prompt, temperature 0 |

`gpt-5.5` reads LongMemEval_S because, on a 100-question sample, it answered 96 correctly against
77–89 for six other OpenAI readers; it rejects `temperature=0` and runs at the default. DMR follows
Zep's published harness
([`zep_memgpt_eval.ipynb`](https://github.com/getzep/zep-papers/blob/main/kg_architecture_agent_memory/zep_memgpt_eval.ipynb))
verbatim: even turns are speaker A, the question is `self_instruct.B` and the gold `self_instruct.A`.

**Against published numbers.** On DMR the Zep paper
([arXiv:2501.13956](https://arxiv.org/abs/2501.13956), Table 1) reports, with the same
`gpt-4o-mini` reader, 98.2% for Zep and 98.0% for the full conversation in context (with
`gpt-4-turbo`: Zep 94.8%, MemGPT 93.4%). Each DMR memory is only 5–7 episodes, so the whole memory
reaches our reader; the 39 misses are details the episode narrative lost.

## Run it yourself

**Install.** Python 3.12 or 3.13; membase-core comes from PyPI as compiled wheels for Linux,
macOS and Windows (proprietary, licensed for running these benchmarks).

```bash
uv sync && source .venv/bin/activate    # or: pip install -e .
export OPENAI_API_KEY=...
export MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini MEMBASE_READER_MODEL=gpt-4.1-mini
```

On Linux without a GPU, `uv venv && uv pip install --torch-backend cpu -e .` avoids the
multi-GB CUDA build of torch that the engine's cross-encoder pulls in.

**Data.** Datasets are not redistributed; they live in `data/` (`BENCH_DATA_DIR` overrides it).
LoCoMo downloads itself on first use. LongMemEval_S (the original release, not
`longmemeval-cleaned`) and DMR:

```bash
curl -L -o data/longmemeval_s.json https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
curl -L --create-dirs -o data/dmr/msc_self_instruct.jsonl https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct/resolve/main/msc_self_instruct.jsonl
```

**Try one question** — a few minutes and well under a dollar:

```bash
python -m bench.locomo.ours_core --limit 1 --out runs/smoke.jsonl --workdir .cache/smoke
python -m bench.locomo.memori_official_eval runs/smoke.jsonl --out runs/smoke.judged.jsonl
```

**Full runs.** Each benchmark is a runner and a judge. `--workdir` keeps the engine stores, so a
rerun (to re-answer, or with `--resume` after an interruption) skips ingestion; every runner also
takes `--limit N`.

```bash
python -m bench.locomo.ours_core --out runs/locomo.jsonl --workdir .cache/locomo --workers 4 --question-workers 8 --resume
python -m bench.locomo.memori_official_eval runs/locomo.jsonl --out runs/locomo.judged.jsonl

MEMBASE_READER_MODEL=gpt-5.5 python -m bench.longmemeval.runner --out runs/lme.jsonl --workdir .cache/lme --workers 8 --ingest-workers 16 --resume
python -m bench.longmemeval.judge runs/lme.jsonl --out runs/lme.judged.jsonl

python -m bench.dmr.runner --out runs/dmr.jsonl --workdir .cache/dmr --workers 12 --resume
python -m bench.dmr.judge runs/dmr.jsonl --out runs/dmr.judged.jsonl
```

| | Ingest (LLM input tokens) | Wall clock |
|---|---|---|
| LoCoMo (10 conversations) | 1.8M | ≈ 20 min to build, ≈ 15 min to answer and judge |
| LongMemEval_S (500 stores) | 225M (449k per question) | ≈ 1 question/min |
| DMR (500 stores) | 12.7M | ≈ 32 min |

LongMemEval_S is the expensive one, since every question has its own ~47-session history;
`--stride 5` runs every 5th question (100 of 500) as a cheaper sample.

**Analyse.**

```bash
python -m bench.locomo.compare runs/a.judged.jsonl runs/b.judged.jsonl   # paired per-question flips between two runs
python -m bench.retrieval_metrics runs/locomo.judged.jsonl                # evidence recall; retrieval vs reader misses
python -m bench.retrieval_metrics --bench longmemeval runs/lme.judged.jsonl
python -m bench.efficiency locomo --workdir .cache/locomo                 # latency and context size (or longmemeval, dmr)
```

`--membase-judge --judge-runs 3` on the LongMemEval judge grades each answer by a majority of three
calls.

## Competitor baselines

[baselines/](baselines/README.md) runs mem0, Memori, LangMem, Zep, Graphiti and a full-context
ceiling on the same LoCoMo and LongMemEval_S questions and sessions, through the same loaders, and
the same judges grade them. Each system has its own extra (`uv sync --extra mem0`, …); DMR has no
baseline runners.

## Repository

```
bench/
  locomo/              runner (ours_core.py), judge (memori_official_eval.py), paired comparison
  longmemeval/         runner and judge
  dmr/                 runner and judge
  common/              shared types and the OpenAI client of the judges and the DMR reader
  retrieval_metrics.py evidence recall and miss attribution
  efficiency.py        serial latency and context size
baselines/             competitor runners
tests/                 offline tests: uv sync --extra dev && uv run pytest
```

Judges, the DMR reader and the baselines call OpenAI directly, never through the engine under test.

## License

MIT, see [LICENSE](LICENSE). Evaluation prompts used verbatim from Memori, Zep and MemGPT are
listed in [NOTICE](NOTICE).
