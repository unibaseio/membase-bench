# Membase 2.0 — Benchmarks

Reproducible LoCoMo eval against Membase, mem0, Memori, LangMem, Zep, and
a Full-Context (no-retrieval) ceiling — all scored by the same
Memori-compatible LLM judge.

## Layout

```
baselines/
├── common/types.py                Shared dataclasses (Instance, Question, Hypothesis)
└── locomo/
    ├── adapter.py                 Loads `baselines/data/locomo10.json` (auto-downloads)
    ├── membase_runner.py          Membase 2.0 — Recovery + Runtime
    ├── mem0_runner.py             mem0 baseline (mem0ai)
    ├── memori_runner.py           Memori baseline (memori / memorisdk)
    ├── langmem_runner.py          LangMem baseline (langmem + langgraph)
    ├── zep_runner.py              Zep baseline (zep-cloud, needs ZEP_API_KEY)
    ├── full_context_runner.py     Full-Context ceiling (no retrieval)
    ├── memori_official_eval.py    LLM judge (Memori notebook prompt) + F1
    └── report.py                  Aggregates judged.jsonl files into a
                                   per-category × method table
```

Each runner produces a `hypotheses.jsonl` with the same schema
(`question_id`, `hypothesis`, `category`, `gold`). The judge takes that
file, asks an LLM "CORRECT / WRONG" per question against the gold answer,
and reports per-category accuracy and F1. The `report` command then
aggregates multiple judged files into a single table:

```
Method                  Single-hop (%)  Multi-hop (%)  Open-domain (%)  Temporal (%)  Overall (%)
Membase                          XX.XX          XX.XX            XX.XX         XX.XX         XX.XX
Mem0                             ...
LangMem                          ...
Zep                              ...
Full-Context (Ceiling)           ...
```

## Setup

```bash
pip install -e '.[recovery,runtime]'    # for the membase runner
pip install mem0ai                      # for the mem0 runner
pip install memorisdk                   # for the memori runner
pip install langmem langgraph           # for the langmem runner
pip install zep-cloud                   # for the zep runner (needs ZEP_API_KEY)
```

`OPENAI_API_KEY` must be in the environment (load from `.env`).

## Run

The dataset is downloaded on first use to `baselines/data/locomo10.json` (155 MB).

### Membase 2.0

```bash
membase bench locomo run --runner membase \
  --out runs/membase.jsonl --workers 4
```

Tunables (env vars read by the membase pipeline):

* `MEMBASE_RECOVERY_OBSERVER_MODEL` (default `gpt-4o-mini`)
* `MEMBASE_RECOVERY_LINKER_MODEL` (default `gpt-4o-mini`)
* `MEMBASE_RUNTIME_READER_MODEL` (default `gpt-4o`)

### mem0

```bash
membase bench locomo run --runner mem0 \
  --out runs/mem0.jsonl --workers 2
```

mem0's vector store has occasional thread-safety issues; keep `--workers`
low. The reader model used to compose answers from retrieved memories
defaults to `gpt-4o`; override with `--reader-model`.

### Memori

```bash
membase bench locomo run --runner memori \
  --out runs/memori.jsonl --workers 2
```

Memori captures memories as a side effect of LLM chat calls. The runner
replays each LoCoMo turn through Memori's `LlmProviderOpenAi` wrapper so
the conversation is captured, then calls `memori.recall(question)` and
synthesizes an answer with a follow-up OpenAI call.

### LangMem

```bash
membase bench locomo run --runner langmem \
  --out runs/langmem.jsonl --workers 2
```

LangMem extracts memories via `create_memory_store_manager` and queries
with the langgraph `BaseStore.search`. We use an in-process
`InMemoryStore` per LoCoMo conversation; both ingest and recall are
self-contained with no external service.

### Zep

```bash
ZEP_API_KEY=... membase bench locomo run --runner zep \
  --out runs/zep.jsonl --workers 2
```

Zep is cloud-only; the runner creates a session per conversation, posts
messages, and queries with `client.memory.search_sessions`. Requires
`ZEP_API_KEY`; sessions are deleted after each haystack to keep the
account tidy.

### Full-Context (ceiling)

```bash
membase bench locomo run --runner full_context \
  --out runs/full_context.jsonl --workers 4
```

No memory store, no retrieval — the entire LoCoMo conversation is sent to
the reader LLM in one shot. This is the LoCoMo paper's ceiling baseline:
any retrieval-based system has to approach this without paying its
per-query token cost.

## Judge

```bash
membase bench locomo judge runs/membase.jsonl --out runs/membase.judged.jsonl
membase bench locomo judge runs/mem0.jsonl    --out runs/mem0.judged.jsonl
membase bench locomo judge runs/memori.jsonl  --out runs/memori.judged.jsonl
```

The judge prints a per-category and overall summary table at the end.

Two metrics are reported:

* **Accuracy (Acc)** — LLM judge labels CORRECT / WRONG. This is the
  number Memori's published LoCoMo notebook uses; lenient on date format
  and phrasing.
* **F1** — token-level F1 between the hypothesis and the gold answer.
  Tighter on phrasing, often misleadingly low when systems answer in a
  different format than the gold (e.g. `2023-05-07` vs `7 May 2023`).
  Treat F1 as supplementary; Accuracy is the primary metric.

## Report (combine multiple runs)

After running the judge on each runner's hypotheses, aggregate the
results into a per-category × method table:

```bash
membase bench locomo report \
  --label Membase=runs/membase.judged.jsonl \
  --label Mem0=runs/mem0.judged.jsonl \
  --label LangMem=runs/langmem.judged.jsonl \
  --label Zep=runs/zep.judged.jsonl \
  --label "Full-Context (Ceiling)"=runs/full_context.judged.jsonl \
  runs/membase.judged.jsonl runs/mem0.judged.jsonl runs/langmem.judged.jsonl \
  runs/zep.judged.jsonl runs/full_context.judged.jsonl
```

Output (matching the LoCoMo paper format):

```
Method                  Single-hop (%)  Multi-hop (%)  Open-domain (%)  Temporal (%)  Overall (%)
Membase                          XX.XX          XX.XX            XX.XX         XX.XX         XX.XX
Mem0                             ...
LangMem                          ...
Zep                              ...
Full-Context (Ceiling)           ...
```

Add `--show-adversarial` to also report the LoCoMo "trick question"
category (`gold = null`) — these are reported separately because their
scoring axis is inverted (a wrong answer here means the system
hallucinated rather than recognising the question is unanswerable).

## Tiny smoke

Run one question through Membase end-to-end (~3 minutes for an LLM-heavy
ingest of one ~400-turn conversation):

```bash
membase bench locomo run --runner membase --out /tmp/smoke.jsonl --limit 1 --workers 1
membase bench locomo judge /tmp/smoke.jsonl --out /tmp/smoke.judged.jsonl
```

## Sample sizes

LoCoMo has ~1500 question-instances grouped into 10 conversations
(haystacks). A full run on one runner against `gpt-4o`-class reader models
takes a couple of hours on this hardware and noticeable token spend.

For development, prefer:

```bash
# 5 questions per category (5 categories × 5 = 25 questions)
membase bench locomo run --runner membase --out runs/membase.sample.jsonl \
  --sample-per-category 5 --seed 42 --workers 4
```

## Reproducibility

* `--seed` controls stratified sampling. Same seed across runners means
  same question subset.
* Each runner writes the dataset's full `question_id` into its hypotheses
  file, so `judge` is order-independent.
* The judge's prompt is byte-identical to Memori's published notebook
  (see `memori_official_eval.ACCURACY_PROMPT`).
