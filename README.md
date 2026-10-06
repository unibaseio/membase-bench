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

Membase hands the reader a few thousand tokens per question instead of the whole history (3× fewer
than the full conversation on LoCoMo, 11× on LongMemEval_S), and the evidence is in them: on
LongMemEval_S the gold session was retrieved for every miss. The SDK is
[membase-ai](https://github.com/unibaseio/membase-ai); more at [unibase.com/memory](https://www.unibase.com/memory).

## One question, end to end

<p align="center">
  <img src="assets/example.svg" width="840" alt="One LoCoMo question from history to judged answer">
</p>

## The benchmarks

| | Questions | History per question | Reader | Judge (`gpt-4o-mini`) |
|---|---|---|---|---|
| **LoCoMo** | 1,540 over 10 conversations (categories 1–4) | ~27 sessions, ~20k tokens | `gpt-4.1-mini` | Memori's CORRECT/WRONG prompt |
| **LongMemEval_S** | 500 | ~50 sessions, ~103k tokens | `gpt-5.5` | Memori's prompt + LongMemEval's per-type rules |
| **DMR** | 500, Zep's harness verbatim | 5 sessions, ~1.6k tokens | `gpt-4o-mini` | MemGPT's prompt |

Each benchmark keeps the protocol of its best-known published numbers. Every question takes the
same path through [membase-core](https://pypi.org/project/membase-core/)'s public API: ingest the
history, search, answer, judge. Episode extraction and the decider use `gpt-4.1-mini`.

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

## Reproduce

Datasets are not redistributed; they live in `data/` (`BENCH_DATA_DIR` overrides it). LoCoMo
downloads itself; LongMemEval_S is the original release, not `longmemeval-cleaned`.

```bash
curl -L -o data/longmemeval_s.json https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
curl -L --create-dirs -o data/dmr/msc_self_instruct.jsonl https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct/resolve/main/msc_self_instruct.jsonl
```

```bash
python -m bench.locomo.ours_core --out runs/locomo.jsonl --workdir .cache/locomo --workers 4 --question-workers 8 --resume
python -m bench.locomo.memori_official_eval runs/locomo.jsonl --out runs/locomo.judged.jsonl

MEMBASE_READER_MODEL=gpt-5.5 python -m bench.longmemeval.runner --out runs/lme.jsonl --workdir .cache/lme --workers 8 --ingest-workers 16 --resume
python -m bench.longmemeval.judge runs/lme.jsonl --out runs/lme.judged.jsonl

python -m bench.dmr.runner --out runs/dmr.jsonl --workdir .cache/dmr --workers 12 --resume
python -m bench.dmr.judge runs/dmr.jsonl --out runs/dmr.judged.jsonl
```

LoCoMo takes about 35 minutes and DMR about 30. LongMemEval_S builds one store per question, about
one question a minute at `--workers 8`; `--stride 5` runs a 100-question sample. Ingest is the
expensive step (1.8M LLM input tokens for LoCoMo, 225M for LongMemEval_S, 12.7M for DMR) and runs
once: `--workdir` keeps the stores, so later answer or judge passes reuse them.

## Results

### Accuracy

<img src="assets/categories.svg" width="840" alt="Accuracy by category and question type">

LoCoMo **93.12%** · LongMemEval_S **92.60%** (95% CI 90.0–94.6) · DMR **92.20%** (95% CI 89.5–94.2).
Every LoCoMo conversation scores between 89.9% and 95.1%, and questions whose evidence spans three
or more sessions score as well as single-session ones.

### Retrieval or reader?

The gold session reaches the reader for 98.1% of LoCoMo and 99.95% of LongMemEval_S evidence, so
most wrong answers are the reader's: 89 of LoCoMo's 105 and all 37 of LongMemEval_S's had the gold
session in front of it (`python -m bench.retrieval_metrics` on a judged run). DMR
histories are short enough to reach the reader whole; with the same `gpt-4o-mini` reader
the [Zep paper](https://arxiv.org/abs/2501.13956) (Table 1) reports 98.2% for Zep and 98.0% for the
full conversation.

<img src="assets/readers.svg" width="840" alt="Accuracy by reader on a LongMemEval_S sample">

On LongMemEval_S the reader matters, most on multi-session and temporal questions, hence `gpt-5.5`.
On LoCoMo it does not: `gpt-5.5` scores 93.18% on the same stores.

### Context and latency

<img src="assets/context.svg" width="840" alt="Context tokens per question against the full history">

<img src="assets/latency.svg" width="840" alt="Search and total latency">

Search includes the multi-round decider's 1–3 LLM calls; that is what keeps the context small.
`python -m bench.efficiency {locomo,longmemeval,dmr} --workdir <kept stores>` measures both on your
own run.

## Baselines

mem0, Memori, LangMem, Zep, Graphiti and a full-context ceiling, on the same LoCoMo and
LongMemEval_S questions, graded by the same judges — see [baselines/](baselines/README.md).
`python -m bench.locomo.compare a.judged.jsonl b.judged.jsonl` pairs two runs question by question.

<details>
<summary><b>Add your own memory system</b></summary>

A system is a runner that writes the same rows; the judges do the rest.

1. Load the questions with `baselines.common.dataset.load()` (`BENCH_DATASET=locomo` or
   `longmemeval`). Each instance has `sessions` (each with `session_date` and `turns` of `role`,
   `speaker`, `content`) and a `question` (`question`, `question_date`, `answer`, `category`).
2. Ingest the sessions into your system, then answer each question.
3. Write `{question_id, hypothesis, category, gold}` per question, as above.
4. Grade with `python -m bench.locomo.memori_official_eval` (or `bench.longmemeval.judge`).

[`baselines/locomo/full_context_runner.py`](baselines/locomo/full_context_runner.py) is the
smallest working example.

Runners write one JSON object per question; judges copy it and add `correct`:

| Field | |
|---|---|
| `question_id`, `category` | from the dataset (`conv-26-q3`, `multi_hop`) |
| `hypothesis` | the answer; `ERROR: …` when the question failed, `[NO_CONTEXT]` when LongMemEval retrieval was empty |
| `gold` | the dataset's answer |
| `retrieved_sessions`, `retrieved_observations` | the sessions of the episodes handed to the reader, and how many episodes |

Judges copy each row and add `correct` and, by benchmark, `label` (`CORRECT` / `WRONG`; LoCoMo and
LongMemEval), `question`, `judge_response` and `f1` (LoCoMo token F1), or `excluded` (an unreadable
verdict, left out of the denominator; LongMemEval and DMR).

</details>

## FAQ

<details>
<summary><b>Why a lenient judge on LoCoMo?</b></summary>

Memori's CORRECT/WRONG prompt is the one behind the published LoCoMo numbers this is compared with.
A stricter judge lowers every system; grade all systems with one judge before comparing them.

</details>

<details>
<summary><b>Why leave out LoCoMo category 5?</b></summary>

Its adversarial questions have no answer and need a refusal-aware judge; published LoCoMo figures
leave it out too. `--include-adversarial` keeps it.

</details>

<details>
<summary><b>Why the original LongMemEval_S?</b></summary>

The published 92.6 was measured on it; `longmemeval-cleaned` is different data, so scores on it are
not comparable.

</details>

<details>
<summary><b>Can I use other model providers?</b></summary>

The engine also runs on Anthropic, Ollama or any OpenAI-compatible endpoint
(`MEMBASE_LLM_PROVIDER`), but this harness's readers, judges and the engine's embeddings call
OpenAI; `OPENAI_BASE_URL` points them at an OpenAI-compatible endpoint.

</details>

## Datasets and citation

| Dataset | Paper | License |
|---|---|---|
| [LoCoMo](https://github.com/snap-research/locomo) | [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) | CC BY-NC 4.0 |
| [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval) | [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) | MIT |
| [MSC-Self-Instruct](https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct) (DMR) | [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560); protocol from [Zep](https://arxiv.org/abs/2501.13956) | Apache-2.0 |

The datasets are downloaded from their sources, not shipped here; LoCoMo's license is
non-commercial. To cite this harness:

```bibtex
@misc{membase-bench,
  title        = {membase-bench: Long-term memory benchmarks for Membase},
  author       = {{Unibase}},
  year         = {2026},
  howpublished = {\url{https://github.com/unibaseio/membase-bench}}
}
```

## License

MIT. The harness drives the proprietary membase-core engine through its public API only; judges and
readers call OpenAI directly. Prompts used verbatim from Memori, Zep and MemGPT are listed in
[NOTICE](NOTICE). The figures are drawn by `scripts/figures.py` from the published runs.
