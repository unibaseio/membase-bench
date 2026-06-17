"""Run LoCoMo through Graphiti (https://github.com/getzep/graphiti).

Graphiti is Zep's open-source temporal knowledge-graph memory engine —
the local, reproducible core under the Zep cloud product. It extracts
entities + relationships from each episode via an LLM and stores them as
a time-aware graph; recall is a hybrid (semantic + BM25 + graph) search
over the edges (facts).

This is the closest architectural comparison to Membase (graph vs graph),
which is why it's worth a fair, apples-to-apples run.

Mapping (mirrors mem0_runner so the same judge scores both):
  - Each LoCoMo conversation = one Graphiti ``group_id`` (isolated graph).
  - Each session is added with ``add_episode`` as a ``message`` episode,
    tagged with the session's ``reference_time`` so the graph is temporally
    grounded (Graphiti's headline feature).
  - Each question runs ``search(query, group_ids=[gid])`` → fact edges,
    then composes an answer with one OpenAI call over the retrieved facts.

Fairness: the extractor + reader are pinned to gpt-4o-mini /
text-embedding-3-small — the same models the mem0 / memori / langmem
runners use — so the only variable is the memory architecture.

Backend: a graph DB. Easiest is FalkorDB (a Redis module):

    docker run -d -p 6379:6379 falkordb/falkordb:latest

Install:
    pip install 'graphiti-core[falkordb]'

Graphiti expects OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import datetime
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from unibase_membase.config import load_config
from bench.common.runners import (
    gold_text,
    group_key as _group_key,
    make_retrieval_row,
    safe_id as _safe_id,
    stratified_sample as _stratified_sample,
)
from bench.common.types import Hypothesis, Instance
from bench.common.dataset import load


_ANSWER_PROMPT = """\
Answer the question using the supplied memory facts. Be concise: a few
words or a short phrase. If the question asks for a date, prefer the
format that appears in the facts.

Facts:
{memories}

Question: {question}

Answer:"""

# Graphiti's entity/edge extraction needs a capable model: gpt-4o-mini
# degenerates into a runaway integer sequence on longer episodes, emitting
# malformed JSON that fails ExtractedEntities validation. gpt-4o is the
# realistic Graphiti extractor (Zep ships 4o-class). Embedder + the answer
# reader stay aligned with the other runners.
_EXTRACT_MODEL = "gpt-4o"
_EMBED_MODEL = "text-embedding-3-small"


def _compose_answer(model: str, query: str, facts: list[str]) -> str:
    from openai import OpenAI
    client = OpenAI()
    body = "\n".join(f"- {m}" for m in facts) if facts else "(none)"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer questions from a long-term memory store."},
            {"role": "user", "content": _ANSWER_PROMPT.format(memories=body, question=query)},
        ],
        max_tokens=128,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_dt(raw: str) -> datetime.datetime:
    """LoCoMo session_date is free-form (e.g. '7 May 2023', '2023-05-07').
    Fall back to a fixed epoch so add_episode always gets a tz-aware dt."""
    raw = (raw or "").strip()
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(raw, fmt).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)


def _make_graphiti():
    """Construct a Graphiti client wired to FalkorDB + gpt-4o-mini extraction.
    Connection params come from env (GRAPHITI_FALKORDB_HOST/PORT) with
    localhost:6379 defaults."""
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    host = os.environ.get("GRAPHITI_FALKORDB_HOST", "localhost")
    port = int(os.environ.get("GRAPHITI_FALKORDB_PORT", "6379"))
    driver = FalkorDriver(host=host, port=port)
    llm = OpenAIClient(config=LLMConfig(
        model=_EXTRACT_MODEL, small_model=_EXTRACT_MODEL, temperature=0,
    ))
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(embedding_model=_EMBED_MODEL))
    return Graphiti(graph_driver=driver, llm_client=llm, embedder=embedder)


async def _run_group_async(
    group_key: str, instances: list[Instance], reader_model: str,
    retrieval_only: bool, retrieval_top_k: int,
) -> list:
    try:
        from graphiti_core.nodes import EpisodeType
    except ImportError as e:
        raise ImportError(
            "graphiti baseline requires `pip install 'graphiti-core[falkordb]'`"
        ) from e

    # Unique group per run so re-runs don't read a half-built graph.
    gid = f"{_safe_id(group_key)}-{uuid.uuid4().hex[:8]}"
    g = _make_graphiti()
    try:
        await g.build_indices_and_constraints()

        # Ingest in small episodes. Graphiti runs LLM entity+edge
        # extraction per episode; feeding a whole LoCoMo session (20-50
        # turns) makes gpt-4o-mini emit a huge entity JSON that overflows
        # and fails to parse. Batch ~6 turns per episode so each
        # extraction stays small and well-formed. reference_time anchors
        # every chunk to the session date (Graphiti's temporal feature).
        _TURNS_PER_EPISODE = 6
        for s in instances[0].sessions:
            ref = _parse_dt(s.session_date)
            lines = []
            for t in s.turns:
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                speaker = t.get("speaker") or t.get("role") or "user"
                lines.append(f"{speaker}: {content}")
            if not lines:
                continue
            for ci in range(0, len(lines), _TURNS_PER_EPISODE):
                chunk = lines[ci:ci + _TURNS_PER_EPISODE]
                try:
                    await g.add_episode(
                        name=f"{group_key}:{s.session_id}:{ci}",
                        episode_body="\n".join(chunk),
                        source=EpisodeType.message,
                        source_description="locomo session",
                        reference_time=ref,
                        group_id=gid,
                    )
                except Exception as exc:
                    print(f"[graphiti] add_episode failed on {s.session_id}#{ci}: {exc}",
                          file=sys.stderr)

        if retrieval_only:
            out_rows: list[dict] = []
            for inst in instances:
                try:
                    edges = await g.search(
                        inst.question.question, group_ids=[gid],
                        num_results=retrieval_top_k,
                    )
                    memories = [{
                        "memory": getattr(e, "fact", str(e)),
                        "score": 0.0,
                        "id": str(getattr(e, "uuid", "")),
                        "created_at": (
                            getattr(e, "valid_at", "") or getattr(e, "created_at", "") or ""
                        ).__str__(),
                        "source": "graphiti",
                    } for e in (edges or [])]
                    out_rows.append(make_retrieval_row(inst, memories=memories))
                except Exception as exc:
                    out_rows.append(make_retrieval_row(
                        inst, memories=None, error=f"{type(exc).__name__}: {exc}"))
            return out_rows

        out: list[Hypothesis] = []
        for inst in instances:
            try:
                edges = await g.search(
                    inst.question.question, group_ids=[gid], num_results=10,
                )
                facts = [getattr(e, "fact", str(e)) for e in (edges or [])]
                ans = _compose_answer(reader_model, inst.question.question, facts)
            except Exception as exc:
                ans = f"ERROR: {type(exc).__name__}: {exc}"
            out.append(Hypothesis(
                question_id=inst.question.question_id,
                hypothesis=ans,
                category=inst.question.category,
                gold=gold_text(inst),
            ))
        return out
    finally:
        try:
            await g.close()
        except Exception:
            pass


def _answer_group(group_key, instances, reader_model, *,
                  retrieval_only=False, retrieval_top_k=200):
    # One event loop per worker thread.
    return asyncio.run(_run_group_async(
        group_key, instances, reader_model, retrieval_only, retrieval_top_k))


def main(argv: list[str] | None = None) -> int:
    cfg = load_config([])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to write hypotheses jsonl")
    ap.add_argument("--workers", type=int, default=2,
                    help="haystacks in parallel; Graphiti extraction is LLM-heavy, keep low")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-category", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reader-model", default=cfg.bench.reader_model)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="Emit retrieval rows (facts) instead of LLM-composed answers.")
    ap.add_argument("--retrieval-top-k", type=int, default=200)
    args = ap.parse_args(argv)

    instances = load(limit=args.limit)
    if args.sample_per_category is not None:
        instances = _stratified_sample(instances, args.sample_per_category, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    kept_lines: list[str] = []
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("hypothesis", "")).startswith("ERROR"):
                continue
            done_ids.add(str(rec["question_id"]))
            kept_lines.append(line)
        instances = [i for i in instances if i.question.question_id not in done_ids]

    grouped: dict[str, list[Instance]] = collections.defaultdict(list)
    for inst in instances:
        grouped[_group_key(inst)].append(inst)

    print(f"[graphiti] locomo: {len(instances)} instances, {len(grouped)} haystacks",
          file=sys.stderr)

    t0 = time.time()
    results: list = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("w") as fout:
        for line in kept_lines:
            fout.write(line + "\n")
        futs = {
            pool.submit(_answer_group, key, group, args.reader_model,
                        retrieval_only=args.retrieval_only,
                        retrieval_top_k=args.retrieval_top_k): key
            for key, group in grouped.items()
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="graphiti:haystacks"):
            group_results = fut.result()
            results.extend(group_results)
            for h in group_results:
                rec = h if isinstance(h, dict) else asdict(h)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[graphiti] wrote {len(results)} new + {len(done_ids)} resumed "
          f"in {time.time() - t0:.0f}s -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
