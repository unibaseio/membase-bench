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

The harness behind Membase's published scores: three public long-term memory benchmarks, run end to
end through [membase-core](https://pypi.org/project/membase-core/)'s public API.

<p align="center">
  <img src="assets/example.svg" width="840" alt="One LoCoMo question from history to judged answer">
</p>

## Benchmarks

| | Questions | History per question | Reader | Judge (`gpt-4o-mini`) |
|---|---|---|---|---|
| [**LoCoMo**](https://github.com/snap-research/locomo) | 1,540 over 10 conversations | ~27 sessions, ~20k tokens | `gpt-4.1-mini` | Memori's CORRECT/WRONG prompt |
| [**LongMemEval_S**](https://huggingface.co/datasets/xiaowu0162/longmemeval) | 500 | ~50 sessions, ~103k tokens | `gpt-5.5` | Memori's prompt + LongMemEval's per-type rules |
| [**DMR**](https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct) | 500 | 5 sessions, ~1.6k tokens | `gpt-4o-mini` | MemGPT's prompt (Zep's harness) |

Each benchmark keeps the protocol of its best-known published numbers. Episode extraction and the
decider use `gpt-4.1-mini`.

## Results

<img src="assets/categories.svg" width="840" alt="Accuracy by category and question type">

<img src="assets/context.svg" width="840" alt="Context tokens per question against the full history">

LoCoMo **93.12%** · LongMemEval_S **92.60%** (95% CI 90.0–94.6) · DMR **92.20%** (95% CI 89.5–94.2),
with a few thousand tokens of context per question instead of the whole history.

<details>
<summary><b>Retrieval, readers and latency</b></summary>

<img src="assets/readers.svg" width="840" alt="Accuracy by reader on a LongMemEval_S sample">

The gold session reaches the reader for 98.1% of LoCoMo and 99.95% of LongMemEval_S evidence, so
most wrong answers are the reader's: 89 of LoCoMo's 105 and all 37 of LongMemEval_S's. That is why
LongMemEval_S uses `gpt-5.5`; on LoCoMo the reader barely matters (`gpt-5.5` scores 93.18% on the
same stores). Every LoCoMo conversation scores between 89.9% and 95.1%.

<img src="assets/latency.svg" width="840" alt="Search and total latency">

Search includes the multi-round decider's 1–3 LLM calls, which keep the context small.

DMR histories are short enough to reach the reader whole; with the same `gpt-4o-mini` reader the
[Zep paper](https://arxiv.org/abs/2501.13956) (Table 1) reports 98.2% for Zep and 98.0% for the
full conversation.

</details>

## Reproduce

Python 3.12 or 3.13 and an OpenAI key.

```bash
git clone https://github.com/unibaseio/membase-bench && cd membase-bench
uv venv --python 3.12 && uv pip install --torch-backend cpu -e ".[dev]" && source .venv/bin/activate
export OPENAI_API_KEY=sk-... MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini MEMBASE_READER_MODEL=gpt-4.1-mini

# LoCoMo downloads itself; the other two go in data/
curl -L -o data/longmemeval_s.json https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
curl -L --create-dirs -o data/dmr/msc_self_instruct.jsonl https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct/resolve/main/msc_self_instruct.jsonl
```

Smoke test, one LoCoMo question in a few minutes:

```bash
python -m bench.locomo.ours_core --limit 1 --out runs/smoke.jsonl --workdir .cache/smoke
python -m bench.locomo.memori_official_eval runs/smoke.jsonl --out runs/smoke.judged.jsonl
```

Full runs:

```bash
python -m bench.locomo.ours_core --out runs/locomo.jsonl --workdir .cache/locomo --workers 4 --question-workers 8 --resume
python -m bench.locomo.memori_official_eval runs/locomo.jsonl --out runs/locomo.judged.jsonl

MEMBASE_READER_MODEL=gpt-5.5 python -m bench.longmemeval.runner --out runs/lme.jsonl --workdir .cache/lme --workers 8 --ingest-workers 16 --resume
python -m bench.longmemeval.judge runs/lme.jsonl --out runs/lme.judged.jsonl

python -m bench.dmr.runner --out runs/dmr.jsonl --workdir .cache/dmr --workers 12 --resume
python -m bench.dmr.judge runs/dmr.jsonl --out runs/dmr.judged.jsonl
```

LoCoMo takes about 35 minutes, DMR about 30, LongMemEval_S about a minute per question
(`--stride 5` runs a 100-question sample). Ingest is the expensive step and runs once: `--workdir`
keeps the stores for later runs.

<details>
<summary><b>Install with pip, or with a GPU</b></summary>

```bash
python3.12 -m venv .venv && source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux without a GPU; skip on macOS / Windows
pip install -e ".[dev]"
```

The CPU build of torch keeps the environment at about 1.5 GB; with uv, drop `--torch-backend cpu`
for the CUDA build (about 6 GB on Linux).

</details>

## Baselines

mem0, Memori, LangMem, Zep, Graphiti and a full-context ceiling run on the same LoCoMo and
LongMemEval_S questions and are graded by the same judges: see [baselines/](baselines/README.md).

<details>
<summary><b>Add your own memory system</b></summary>

1. Load the questions with `baselines.common.dataset.load()` (`BENCH_DATASET=locomo` or
   `longmemeval`): each instance has `sessions` and a `question`.
2. Ingest the sessions into your system and answer each question.
3. Write one JSON line per question: `{question_id, hypothesis, category, gold}`.
4. Grade with `python -m bench.locomo.memori_official_eval` (or `bench.longmemeval.judge`).

[`baselines/locomo/full_context_runner.py`](baselines/locomo/full_context_runner.py) is the
smallest working example.

</details>

<details>
<summary><b>FAQ</b></summary>

**Why a lenient judge on LoCoMo?** Memori's CORRECT/WRONG prompt is the one behind the published
LoCoMo numbers. A stricter judge lowers every system; grade all systems with one judge.

**Why leave out LoCoMo category 5?** Its adversarial questions have no answer and need a
refusal-aware judge; published LoCoMo figures leave it out too. `--include-adversarial` keeps it.

**Why the original LongMemEval_S?** The published 92.6 was measured on it; `longmemeval-cleaned` is
different data.

**Other model providers?** Readers, judges and embeddings call OpenAI; `OPENAI_BASE_URL` points them
at any OpenAI-compatible endpoint.

</details>

## License and citation

MIT. The harness drives the proprietary membase-core engine through its public API only. The
datasets are downloaded from their sources, not shipped here: LoCoMo
([paper](https://arxiv.org/abs/2402.17753), CC BY-NC 4.0), LongMemEval
([paper](https://arxiv.org/abs/2410.10813), MIT), MSC-Self-Instruct
([MemGPT](https://arxiv.org/abs/2310.08560), Apache-2.0; DMR protocol from
[Zep](https://arxiv.org/abs/2501.13956)). Prompts used verbatim are listed in [NOTICE](NOTICE).

```bibtex
@misc{membase-bench,
  title        = {membase-bench: Long-term memory benchmarks for Membase},
  author       = {{Unibase}},
  year         = {2026},
  howpublished = {\url{https://github.com/unibaseio/membase-bench}}
}
```
