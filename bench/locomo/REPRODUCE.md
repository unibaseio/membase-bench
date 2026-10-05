# Reproducing the benchmark numbers

## Running

Episodes are always extracted and `multiround` is the default retrieval mode, so only the models
need setting. The runners pass the per-benchmark choices to `CoreMemoryEngine` themselves:

| | answer prompt | "Current Date" line | episode owner / search owner |
|---|---|---|---|
| LoCoMo (`bench/locomo/ours_core.py`) | `answer_locomo` | off | each conversation's `speaker_a` |
| LongMemEval (`bench/longmemeval/runner.py`) | `answer_longmemeval` (16384-token budget) | on | `user` |
| DMR (`bench/dmr/runner.py`) | Zep's prompt, outside the engine (`gpt-4o-mini`) | n/a | `A` |

```bash
export MEMBASE_EPISODE_MODEL=gpt-4.1-mini
export MEMBASE_DECIDER_MODEL=gpt-4.1-mini
export MEMBASE_READER_MODEL=gpt-4.1-mini      # LoCoMo; gpt-5.5 for LongMemEval; unused by DMR
```

Commands for each benchmark are in its section below. Add `--limit N` to any runner for a smoke run.

---

# LoCoMo

**93.12% micro accuracy on the 1,540 graded LoCoMo questions** (categories 1–4; category 5
dropped, as in the mem0, Zep and Memori LoCoMo figures). Measured 2026-09-18 on a fresh store.

| category    | n    | ours   |
|-------------|------|--------|
| multi_hop   | 282  | 93.6%  |
| open_domain | 96   | 83.3%  |
| single_hop  | 841  | 94.6%  |
| temporal    | 321  | 91.6%  |
| **micro**   | 1540 | **93.12%** |

Run-to-run noise on this benchmark is about 0.9pp; differences
inside that band are not differences.

## Exact configuration

Every model is `gpt-4.1-mini` except the judge (`gpt-4o-mini`) — the standard published LoCoMo
config. Embedding is `text-embedding-3-small`. Retrieval is `multiround` over **episodes only**, owner-scoped to each conversation's
`speaker_a`, one owner partition per conversation.

```bash
export MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini
export MEMBASE_READER_MODEL=gpt-4.1-mini
# The runner extracts episodes for speaker_a only, scopes search to that owner, and answers
# with answer_locomo and no date line.

python -m bench.locomo.ours_core --out runs/<arm>.jsonl --workdir .cache/<arm> \
    --workers 4 --question-workers 8 --resume
python -m bench.locomo.memori_official_eval runs/<arm>.jsonl --out runs/<arm>.judged.jsonl
python -m bench.locomo.retrieval_metrics runs/<arm>.judged.jsonl   # recall + miss attribution
python -m bench.locomo.compare runs/a.judged.jsonl runs/b.judged.jsonl  # paired per-question flips
```

Wall clock ≈ 20 min to build the 10 stores, ≈ 15 min to answer and judge. `--workdir` keeps the
stores; a second arm on the same stores skips ingest.

## What the number depends on (engine design choices, measured)

- **The generic whole-cell episode prompt**, not a user-centred one. The user-centred prompt
  produced episodes 61% as long, dropped the other speaker's facts and emitted Spanish for some
  speaker names; the choice was worth ~2pp overall.
- **Episodes only in the context.** Adding per-fact observations alongside the episodes scored
  91.56%: their `(on <session date>)` stamp is the date a fact was *said* and mis-anchors
  temporal answers (temporal 82.9% vs 91.6%).
- **No "Current Date" line.** LoCoMo has no per-question date; the loader's fabricated one
  mis-anchors relative-date arithmetic.
- **Image captions stripped** from extraction input (`[Shared image: …]`). With captions in, 20% of episodes came back non-English.
- **Two-pass boundary detection**: an open-ended pass, then a final pass on the tail. A single
  final-mode call never splits a session. Output is stochastic at temperature 0.
- **Serialised reads on the retrieval path**: without them, 8 worker threads on one store
  produced crashed answers graded wrong.

Reader upgrade does NOT move LoCoMo: the same stores re-answered with `gpt-5.5` score 93.18
(paired vs 93.12: 42 fixed / 41 broken, net +1 — pure noise), while abstentions rise 3 → 26.
Keep `gpt-4.1-mini` as the LoCoMo reader (same score, far cheaper).


---

# LongMemEval_S — full 500 (2026-09-21)

**92.60% micro (463/500), Wilson 95% CI 90.0–94.6**. Reader `gpt-5.5`, extractor/decider
`gpt-4.1-mini`, same harness as below.

| type | n | acc |
|---|---|---|
| knowledge-update | 78 | 97.4 |
| multi-session | 133 | 88.0 |
| single-session-assistant | 56 | 85.7 |
| single-session-preference | 30 | 100 |
| single-session-user | 70 | 98.6 |
| temporal-reasoning | 133 | 92.5 |

Retrieval: recall 99.95%, all gold present for 99.8%; `_abs` 29/30; 0 `[NO_CONTEXT]`, 0 errors.
**All 37 misses are reader misses** (gold in context). Episode extraction retries malformed-JSON
replies up to three times and skips a cell that still fails, rather than failing the question.
Wall clock: ~3 h for the 400 stores not already built by the 100-question sample
(`--workers 12 --ingest-workers 16`), 0 rate-limit retries.

# LongMemEval_S (100-question sample, 2026-09-18)

Harness: `bench/longmemeval/{adapter,runner,judge}.py`. Semantics: one
store and one owner per question, positional `session_<k>` ids, gold from `answer_session_ids`,
a `Current Date` line (LongMemEval supplies one), `[NO_CONTEXT]` on empty retrieval (graded wrong
without a judge call), LongMemEval's per-type leniency clauses in the judge, unreadable verdicts excluded from
the denominator (count printed). Dataset: HF `xiaowu0162/longmemeval`, file `longmemeval_s`.

**Sample: every 5th question (100 of 500).** Not the full set; 95% CI is roughly ±8pp.

| reader model  | micro  | knowledge-update | multi-session | ss-assistant | ss-preference | ss-user | temporal |
|---------------|--------|------|------|------|------|-------|------|
| gpt-4.1-mini  | 83.0%  | 86.7 | 70.4 | 81.8 | 100  | 100   | 81.5 |
| gpt-5-mini    | 87.0%  | 100  | 74.1 | 81.8 | 83.3 | 100   | 88.9 |
| gpt-4.1       | 86.0%  | 100  | 74.1 | 81.8 | 83.3 | 92.9  | 88.9 |
| gpt-5         | 87.0%  | 93.3 | 77.8 | 72.7 | 100  | 100   | 88.9 |
| gpt-5.4-mini  | 77.0%  | 86.7 | 63.0 | 63.6 | 100  | 100   | 74.1 |
| gpt-5.4       | 89.0%  | 93.3 | 88.9 | 72.7 | 83.3 | 100   | 88.9 |
| **gpt-5.5**   | **96.0%** | 100 | 92.6 | 81.8 | 100  | 100   | 100  |

Same stores, re-answer only, so these are paired. 63 questions are right under all seven readers,
3 are wrong under all seven (two cross-session aggregation questions and one
assistant-recommendation recall), and the union of correct answers is 97/100. Most readers land at
77–89; **gpt-5.5 reaches 96** (+13 net vs gpt-4.1-mini, 0 broken) and is the reader to use.
gpt-5.5 rejects `temperature=0`, so it runs at the default.

Retrieval is not the gap: recall 99.75%, all gold sessions present for 99% of questions, all six
sampled `_abs` questions correct, 0 `[NO_CONTEXT]`, 0 errors. With the gpt-4.1-mini reader, 16 of
17 misses had the gold in the context (with gpt-5.5, all 4). The misses are aggregation over dense narratives — counting instances across sessions
(user-did vs assistant-suggested, de-duplication) and date subtraction — and the episodes were
checked to be faithful to the raw text on three of them. The remaining lever is the answer model.

```bash
export MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini
export MEMBASE_READER_MODEL=gpt-5.5          # 96.0 on the sample (gpt-5.4 89.0, gpt-5-mini 87.0, gpt-4.1-mini 83.0)
python -m bench.longmemeval.runner --out runs/lme.jsonl --workers 8 --ingest-workers 16 \
    --workdir .cache/lme --resume                  # add --stride 5 for the 100-q sample
python -m bench.longmemeval.judge runs/lme.jsonl --out runs/lme.judged.jsonl
python -m bench.locomo.retrieval_metrics --bench longmemeval runs/lme.judged.jsonl
```

Time: ingest is ≈ 47 sessions per question; at 8 questions × 16 ingest workers ≈ 1 question/min,
with no rate-limit retries observed. Re-answering existing stores takes ~7 s/question.


---

# DMR — Deep Memory Retrieval (MSC-Self-Instruct, full 500, 2026-09-21)

**92.20% (461/500), Wilson 95% CI 89.5–94.2**. The Zep paper
([arXiv:2501.13956](https://arxiv.org/abs/2501.13956), Table 1) reports, with the same
`gpt-4o-mini` reader, 98.2% for Zep and 98.0% for the full conversation in context; with
`gpt-4-turbo`, Zep 94.8% and MemGPT 93.4%.

Protocol is Zep's published harness
([`zep_memgpt_eval.ipynb`](https://github.com/getzep/zep-papers/blob/main/kg_architecture_agent_memory/zep_memgpt_eval.ipynb))
verbatim: all five
sessions ingested (four `previous_dialogs` + current `dialog`), even turns = speaker A / odd = B,
question `self_instruct.B`, gold `self_instruct.A`, answered in A's first person with Zep's prompt,
answer and judge both `gpt-4o-mini` at temperature 0, judge prompt = MemGPT's ("contains the correct
answer or touches on the same topic … subset → false"). Only the memory system differs: ours is
episodes (gpt-4.1-mini, generic prompt) + multi-round decider, owner-scoped to A.

Harness: `bench/dmr/{adapter,runner,judge}.py`; dataset HF `MemGPT/MSC-Self-Instruct`
(`data/dmr/msc_self_instruct.jsonl`). Each question has 5 short sessions → 5–7 episodes, so
top_k=20 injects the whole memory and retrieval does no filtering; the 39 misses are detail lost in
the per-session narrative ("work at a gas station" vs gold "manager at a gas station"; "vanilla ice
cream and cats" absent). Only 4 misses are abstentions. 0 errors, 0 skipped cells.
Wall clock 32 min at 12 workers.

```bash
export MEMBASE_EPISODE_MODEL=gpt-4.1-mini MEMBASE_DECIDER_MODEL=gpt-4.1-mini
python -m bench.dmr.runner --out runs/dmr_full.jsonl --workers 12 --workdir .cache/dmr --resume
python -m bench.dmr.judge runs/dmr_full.jsonl --out runs/dmr_full.judged.jsonl
```


---

# Efficiency metrics (serial timing, 2026-09-21)

Final configs; one question at a time so latency is not flattered by concurrency. Search = FAISS +
FTS + multi-round decider LLM calls. Context tokens = tiktoken (o200k) over the reader's context.
Reproduce with `python -m bench.efficiency {locomo,longmemeval,dmr} --workdir <kept stores>` (40 / 30 /
30 questions spread over the stores by default). For LoCoMo and LongMemEval, answer time is the
end-to-end time less a separate search; DMR's reader runs outside the engine, so both are direct.

| | LoCoMo (40 q) | LongMemEval (30 q) | DMR (30 q) |
|---|---|---|---|
| reader | gpt-4.1-mini | gpt-5.5 | gpt-4o-mini (Zep protocol) |
| search p50 / p95 | 1.67 s / 7.02 s | 2.53 s / 6.11 s | 1.13 s / 1.71 s |
| answer p50 / p95 | 5.86 s / 10.9 s | 12.0 s / 28.1 s | 2.06 s / 4.63 s |
| total p50 / p95 | 8.30 s / 18.0 s | 14.7 s / 30.2 s | 3.21 s / 6.34 s |
| context tokens/q (mean) | 6,562 | 8,970 | 1,602 |
| answer output tokens (mean) | 798 | 697 | 101 |
| store: episodes / sessions per unit | 32 / 27 per conv | 54 / 48 per q | 5 / 5 per q |
| store size (tokens) | ~11.5k per conv | ~28.7k per q | ~1.7k per q |
| ingest LLM input tokens | 176k per conv (1.8M total) | 449k per q (225M total) | 25k per q (12.7M total) |

For reference, mem0's paper (OSS, LoCoMo): 1,764 memory tokens/q, search p50 0.148 s / p95 0.200 s,
total p50 0.708 s / p95 1.44 s; its 2026 platform: 6,956 tokens/q. Our search latency is an order of
magnitude higher because the decider is 1–3 LLM round-trips per query; context volume is on par.

