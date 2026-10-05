# Baselines

Competitor memory systems on the same data as `bench/`: mem0, Memori, LangMem, Zep, Graphiti
and a full-context ceiling. The runners load questions through `bench/`'s own loaders
(`BENCH_DATASET=locomo`, the default, or `longmemeval`), so every system answers the same
questions over the same sessions; LoCoMo category 5 is dropped, as for Membase. Every runner
writes a `hypotheses.jsonl` with the same schema (`question_id`, `hypothesis`, `category`,
`gold`), so `bench/`'s judge grades every system. For `BENCH_DATASET=longmemeval`, grade with
`python -m bench.longmemeval.judge` (`report.py` is LoCoMo-only).

Membase itself runs through `bench/locomo/ours_core.py` (see the top-level README).

## Layout

```
baselines/
├── common/          dataset dispatch, model defaults (config.py), OpenAI calls (llm.py), sampling
└── locomo/
    ├── mem0_runner.py             mem0 (mem0ai)
    ├── mem0_eval/                 mem0's own LoCoMo evaluation scaffold (Apache-2.0, see NOTICE)
    ├── memori_runner.py           Memori
    ├── langmem_runner.py          LangMem (langmem + langgraph)
    ├── zep_runner.py              Zep Cloud (needs ZEP_API_KEY)
    ├── graphiti_runner.py         Graphiti (FalkorDB)
    ├── full_context_runner.py     full-context ceiling, no retrieval
    └── report.py                  per-category x method table from judged files
```

## Setup

Each system has its own extra; one environment per system keeps their dependencies apart:

```bash
uv sync --extra mem0                               # or memori, langmem, zep, graphiti
export OPENAI_API_KEY=...
export MEMBASE_BENCH_MODEL=gpt-4o                  # the runners' default reader model
```

The baseline readers call OpenAI directly (`chat.completions.create`, max_tokens 128, the
provider's default temperature); the judge and mem0_eval go through `bench/common/llm.py`
(temperature 0, retries on transient errors). `OPENAI_BASE_URL` points them at an
OpenAI-compatible endpoint. Each system's own extraction model is fixed in its runner
(`gpt-4o-mini`; Graphiti `gpt-4o`); `--reader-model` sets only the answering model.

## Run

```bash
python -m baselines.locomo.mem0_runner         --out runs/mem0.jsonl         --workers 2
python -m baselines.locomo.memori_runner       --out runs/memori.jsonl       --workers 2
python -m baselines.locomo.langmem_runner      --out runs/langmem.jsonl      --workers 2
ZEP_API_KEY=... python -m baselines.locomo.zep_runner --out runs/zep.jsonl   --workers 2
python -m baselines.locomo.graphiti_runner     --out runs/graphiti.jsonl     --workers 2
python -m baselines.locomo.full_context_runner --out runs/full_context.jsonl --workers 4
```

Common flags: `--limit N` (first N questions), `--sample-per-category N --seed S` (stratified
sample; the same seed gives every runner the same subset), `--reader-model`, `--resume`.
mem0, Memori, LangMem and Graphiti also take `--retrieval-only` and `--retrieval-top-k`: the
runner then writes the retrieved memories instead of an answer, to be read and graded by
`python -m baselines.locomo.mem0_eval.run` (mem0's own reader prompt and judge).

- **mem0**: its vector store has occasional thread-safety issues; keep `--workers` low.
- **Memori** captures memories as a side effect of chat calls: the runner replays each turn
  through Memori's OpenAI wrapper, then answers from `memori.recall(question)`.
- **LangMem** extracts with `create_memory_store_manager` into an in-process `InMemoryStore`
  per conversation.
- **Zep** is cloud-only: one session per conversation, deleted after the haystack.
- **Graphiti** needs a FalkorDB server (`docker run -p 6379:6379 falkordb/falkordb`);
  `GRAPHITI_FALKORDB_HOST` / `GRAPHITI_FALKORDB_PORT` override localhost:6379.
- **Full context** sends the whole conversation to the reader in one shot: the LoCoMo paper's
  ceiling.

## Judge and report

```bash
python -m bench.locomo.memori_official_eval runs/mem0.jsonl --out runs/mem0.judged.jsonl
python -m baselines.locomo.report \
  --label Mem0=runs/mem0.judged.jsonl --label Zep=runs/zep.judged.jsonl \
  runs/mem0.judged.jsonl runs/zep.judged.jsonl
```

The judge is `bench/`'s: Memori's published LoCoMo notebook prompt (`ACCURACY_PROMPT`), accuracy
(primary) and token F1 (supplementary), `--judge-model gpt-4o-mini` as for the published Membase
numbers. `report --show-adversarial` adds LoCoMo category 5, reported separately because its
scoring axis is inverted; only `bench.locomo.ours_core --include-adversarial` keeps it, the
baseline runners always drop it. `--retrieval-only` output is graded by mem0_eval instead, with
`--judge-model` defaulting to `MEMBASE_BENCH_MODEL` (gpt-4o).
