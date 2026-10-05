"""Graphiti baseline (https://github.com/getzep/graphiti)."""

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

from baselines.common.config import default_model
from baselines.common.runners import (
    gold_text,
    group_key as _group_key,
    safe_id as _safe_id,
    speaker_line,
    stratified_sample as _stratified_sample,
)
from bench.common.types import Hypothesis, Instance
from baselines.common.dataset import load


_ANSWER_PROMPT = """\
Answer the question using the supplied memory facts. Be concise: a few
words or a short phrase. If the question asks for a date, prefer the
format that appears in the facts.

Facts:
{memories}

Question: {question}

Answer:"""

# gpt-4o-mini emits malformed entity JSON on Graphiti's extraction prompts.
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
    raw = (raw or "").strip()
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(raw, fmt).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)


def _make_graphiti():
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
) -> list[Hypothesis]:
    try:
        from graphiti_core.nodes import EpisodeType
    except ImportError as e:
        raise ImportError(
            "graphiti baseline requires `pip install 'graphiti-core[falkordb]'`"
        ) from e

    gid = f"{_safe_id(group_key)}-{uuid.uuid4().hex[:8]}"
    g = _make_graphiti()
    try:
        await g.build_indices_and_constraints()

        # Short episodes keep Graphiti's per-episode extraction JSON well-formed.
        _TURNS_PER_EPISODE = 6
        for s in instances[0].sessions:
            ref = _parse_dt(s.session_date)
            lines = []
            for t in s.turns:
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                lines.append(speaker_line(t))
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


def _answer_group(group_key, instances, reader_model):
    return asyncio.run(_run_group_async(group_key, instances, reader_model))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to write hypotheses jsonl")
    ap.add_argument("--workers", type=int, default=2,
                    help="haystacks in parallel; Graphiti extraction is LLM-heavy, keep low")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-category", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reader-model", default=default_model())
    ap.add_argument("--resume", action="store_true")
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
            pool.submit(_answer_group, key, group, args.reader_model): key
            for key, group in grouped.items()
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="graphiti:haystacks"):
            group_results = fut.result()
            results.extend(group_results)
            for h in group_results:
                fout.write(json.dumps(asdict(h), ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[graphiti] wrote {len(results)} new + {len(done_ids)} resumed "
          f"in {time.time() - t0:.0f}s -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
