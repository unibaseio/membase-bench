<p align="center">
  <img src="assets/membase-logo.png" width="72" alt="Membase">
</p>

<h1 align="center">membase-bench</h1>

<p align="center">
  <b>Long-term memory benchmarks for <a href="https://www.unibase.com/memory">Membase</a></b><br>
  LoCoMo · LongMemEval · DMR — reproducible end to end, with competitor baselines on the same data.
</p>

<p align="center">
  <a href="https://pypi.org/project/membase-core/"><img src="https://img.shields.io/pypi/v/membase-core?label=membase-core" alt="membase-core"></a>
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python 3.12 | 3.13">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"></a>
</p>

<p align="center">
  <img src="assets/results.svg" width="840" alt="LoCoMo 93.1%, LongMemEval_S 92.6%, DMR 92.2%">
</p>

Membase hands the reader a few thousand tokens per question instead of the whole history, and it
finds the evidence: on LongMemEval_S the gold sessions were in the context for every miss.

## Quickstart

```bash
git clone https://github.com/unibaseio/membase-bench && cd membase-bench
uv venv --python 3.12 && uv pip install --torch-backend cpu -e ".[dev]" && source .venv/bin/activate
export OPENAI_API_KEY=sk-... MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini MEMBASE_READER_MODEL=gpt-4.1-mini

python -m bench.locomo.ours_core --limit 1 --out runs/smoke.jsonl --workdir .cache/smoke
python -m bench.locomo.memori_official_eval runs/smoke.jsonl --out runs/smoke.judged.jsonl
```

One LoCoMo question, end to end, in a few minutes. Needs Python 3.12 or 3.13 on Linux, macOS or
Windows, and an OpenAI key.

## The benchmarks

Each benchmark keeps the protocol of its best-known published numbers.

| | What it tests | Reader | Judge (`gpt-4o-mini`) |
|---|---|---|---|
| **LoCoMo** | 10 long two-person conversations, 1,540 questions (categories 1–4) | `gpt-4.1-mini` | Memori's CORRECT/WRONG prompt |
| **LongMemEval_S** | 500 questions, each over its own ~47-session chat history | `gpt-5.5` | Memori's prompt + LongMemEval's per-type rules |
| **DMR** | 500 questions on MemGPT's MSC-Self-Instruct, Zep's harness verbatim | `gpt-4o-mini` | MemGPT's prompt |

Each question runs the same path through [membase-core](https://pypi.org/project/membase-core/)'s
public API: **ingest** the history into dated episodes → **search** with a multi-round LLM
decider → **answer** with the reader → **judge** against the gold answer. Episode extraction and
the decider use `gpt-4.1-mini`.

## Reproduce

```bash
python -m bench.locomo.ours_core --out runs/locomo.jsonl --workdir .cache/locomo --workers 4 --question-workers 8 --resume
python -m bench.locomo.memori_official_eval runs/locomo.jsonl --out runs/locomo.judged.jsonl

MEMBASE_READER_MODEL=gpt-5.5 python -m bench.longmemeval.runner --out runs/lme.jsonl --workdir .cache/lme --workers 8 --ingest-workers 16 --resume
python -m bench.longmemeval.judge runs/lme.jsonl --out runs/lme.judged.jsonl

python -m bench.dmr.runner --out runs/dmr.jsonl --workdir .cache/dmr --workers 12 --resume
python -m bench.dmr.judge runs/dmr.jsonl --out runs/dmr.judged.jsonl
```

LoCoMo takes about 35 minutes and DMR about 30. LongMemEval_S builds one store per question, about
one question a minute at `--workers 8`; `--stride 5` runs a 100-question sample. `--workdir` keeps
the stores, so a rerun skips ingestion.

<details>
<summary><b>Data</b></summary>

Datasets are not redistributed; they live in `data/` (`BENCH_DATA_DIR` overrides it). LoCoMo
downloads itself on first use. LongMemEval_S is the original release, not `longmemeval-cleaned`.

```bash
curl -L -o data/longmemeval_s.json https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
curl -L --create-dirs -o data/dmr/msc_self_instruct.jsonl https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct/resolve/main/msc_self_instruct.jsonl
```

</details>

<details>
<summary><b>Install with pip, or with a GPU</b></summary>

```bash
python3.12 -m venv .venv && source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux without a GPU; skip on macOS / Windows
pip install -e ".[dev]"
```

The CPU build of torch (which the engine's cross-encoder needs) keeps the environment at about
1.5 GB; with uv, drop `--torch-backend cpu` for the CUDA build (about 6 GB on Linux). `pytest`
checks the install offline.

</details>

<details>
<summary><b>Results in detail</b></summary>

| LoCoMo category | n | Accuracy |
|---|---|---|
| multi_hop | 282 | 93.6% |
| open_domain | 96 | 83.3% |
| single_hop | 841 | 94.6% |
| temporal | 321 | 91.6% |
| **all** | **1,540** | **93.12%** |

| LongMemEval_S type | n | Accuracy |
|---|---|---|
| knowledge-update | 78 | 97.4% |
| multi-session | 133 | 88.0% |
| single-session-assistant | 56 | 85.7% |
| single-session-preference | 30 | 100% |
| single-session-user | 70 | 98.6% |
| temporal-reasoning | 133 | 92.5% |
| **all** | **500** | **92.60%** (95% CI 90.0–94.6) |

DMR: **92.20%** (461/500, 95% CI 89.5–94.2).

- **LongMemEval_S** — recall 99.95%; all 37 misses are the reader's. `gpt-5.5` reads because, on a
  100-question sample, it answered 96 against 77–89 for six other OpenAI readers.
- **LoCoMo** — a stronger reader does not help: `gpt-5.5` scores 93.18% on the same stores.
- **DMR** — each memory is only 5–7 episodes, so all of it reaches the reader. With the same
  `gpt-4o-mini` reader, the [Zep paper](https://arxiv.org/abs/2501.13956) (Table 1) reports 98.2%
  for Zep and 98.0% for the full conversation in context.

</details>

<details>
<summary><b>Latency and context size</b></summary>

One question at a time; search includes the decider's LLM calls.

| | LoCoMo | LongMemEval_S | DMR |
|---|---|---|---|
| search p50 / p95 | 1.67 s / 7.02 s | 2.53 s / 6.11 s | 1.13 s / 1.71 s |
| total p50 / p95 | 8.30 s / 18.0 s | 14.7 s / 30.2 s | 3.21 s / 6.34 s |
| context tokens per question | 6,562 | 8,970 | 1,602 |

`python -m bench.efficiency {locomo,longmemeval,dmr} --workdir <kept stores>` measures them on your
own run.

</details>

<details>
<summary><b>Analysis tools</b></summary>

```bash
python -m bench.locomo.compare runs/a.judged.jsonl runs/b.judged.jsonl   # paired per-question flips
python -m bench.retrieval_metrics runs/locomo.judged.jsonl                # evidence recall; retrieval vs reader misses
python -m bench.longmemeval.judge runs/lme.jsonl --out x.jsonl --membase-judge --judge-runs 3   # majority of three judges
```

</details>

## Baselines

mem0, Memori, LangMem, Zep, Graphiti and a full-context ceiling, on the same LoCoMo and
LongMemEval_S questions, graded by the same judges — see [baselines/](baselines/README.md).

## License

MIT. The harness drives the proprietary membase-core engine through its public API only; judges and
readers call OpenAI directly. Prompts used verbatim from Memori, Zep and MemGPT are listed in
[NOTICE](NOTICE).
