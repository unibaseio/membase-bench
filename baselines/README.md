# Baselines

Competitor memory systems on the same LoCoMo data as `bench/`: mem0, Memori, LangMem, Zep,
Graphiti and a full-context ceiling, plus loaders for LongMemEval (`baselines/longmemeval`) and
BEAM (`baselines/beam`). Every runner writes a `hypotheses.jsonl` with the same schema
(`question_id`, `hypothesis`, `category`, `gold`), so one judge grades every system.

Membase itself runs through `bench/locomo/ours_core.py` (see the top-level README).

## Layout

```
baselines/
├── common/          shared types, dataset helpers, model defaults (config.py), OpenAI calls (llm.py)
└── locomo/
    ├── adapter.py                 loads baselines/data/locomo10.json (downloaded on first use)
    ├── mem0_runner.py             mem0 (mem0ai)
    ├── mem0_eval/                 mem0's own LoCoMo evaluation scaffold (Apache-2.0, see NOTICE)
    ├── memori_runner.py           Memori
    ├── langmem_runner.py          LangMem (langmem + langgraph)
    ├── zep_runner.py              Zep Cloud (needs ZEP_API_KEY)
    ├── graphiti_runner.py         Graphiti (FalkorDB)
    ├── full_context_runner.py     full-context ceiling, no retrieval
    ├── memori_official_eval.py    LLM judge (Memori notebook prompt) + F1
    └── report.py                  per-category x method table from judged files
```

## Setup

Each system has its own extra, because their dependency trees clash; use one environment per
system:

```bash
uv pip install -r pyproject.toml --extra mem0      # or memori, langmem, zep, graphiti
export OPENAI_API_KEY=...
export MEMBASE_BENCH_MODEL=gpt-4o                  # the runners' default reader model
```

The judge and readers call OpenAI through `bench/common/llm.py` (temperature 0, retries on
transient errors); `OPENAI_BASE_URL` points them at an OpenAI-compatible endpoint.

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
mem0, Memori, LangMem and Graphiti also take `--retrieval-only` and `--retrieval-top-k`.

- **mem0**: its vector store has occasional thread-safety issues; keep `--workers` low.
- **Memori** captures memories as a side effect of chat calls: the runner replays each turn
  through Memori's OpenAI wrapper, then answers from `memori.recall(question)`.
- **LangMem** extracts with `create_memory_store_manager` into an in-process `InMemoryStore`
  per conversation.
- **Zep** is cloud-only: one session per conversation, deleted after the haystack.
- **Full context** sends the whole conversation to the reader in one shot: the LoCoMo paper's
  ceiling.

## Judge and report

```bash
python -m baselines.locomo.memori_official_eval runs/mem0.jsonl --out runs/mem0.judged.jsonl
python -m baselines.locomo.report \
  --label Mem0=runs/mem0.judged.jsonl --label Zep=runs/zep.judged.jsonl \
  runs/mem0.judged.jsonl runs/zep.judged.jsonl
```

The judge prompt is byte-identical to Memori's published LoCoMo notebook
(`ACCURACY_PROMPT`) and reports accuracy (primary) and token F1 (supplementary). This copy
defaults to `--judge-model gpt-4.1-mini`; `bench/locomo/memori_official_eval.py` is the same
prompt defaulting to `gpt-4o-mini`, which the published Membase numbers use. Grade every system
with the same judge model before comparing. `report --show-adversarial` adds LoCoMo category 5, reported separately
because its scoring axis is inverted.
