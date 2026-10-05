"""LangMem baseline (https://github.com/langchain-ai/langmem)."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from baselines.common.config import default_model
from baselines.common.runners import (
    chat_roles,
    gold_text,
    group_key as _group_key,
    safe_id as _safe_ns,
    speaker_of,
    stratified_sample as _stratified_sample,
)
from bench.common.types import Hypothesis, Instance
from baselines.common.dataset import load


_ANSWER_PROMPT = """\
Answer the question using the supplied recalled memories. Be concise: a
few words or a short phrase. If the question asks for a date, prefer the
format that appears in the memories.

Memories:
{memories}

Question: {question}

Answer:"""


def _compose_answer(model: str, query: str, memories: list[str]) -> str:
    from openai import OpenAI
    client = OpenAI()
    body = "\n".join(f"- {m}" for m in memories) if memories else "(none)"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You answer questions from a long-term memory store."},
            {"role": "user", "content": _ANSWER_PROMPT.format(memories=body, question=query)},
        ],
        max_tokens=128,
    )
    return (resp.choices[0].message.content or "").strip()


def _ingest_with_langmem(manager, namespace: tuple, turns: list[dict], roles: dict[str, str]) -> None:
    from langchain_core.messages import HumanMessage, AIMessage
    msgs = []
    for t in turns:
        content = (t.get("content") or "").strip()
        if not content:
            continue
        if roles[speaker_of(t)] == "user":
            msgs.append(HumanMessage(content=content))
        else:
            msgs.append(AIMessage(content=content))
    if not msgs:
        return
    try:
        manager.invoke({"messages": msgs},
                       config={"configurable": {"langgraph_user_id": namespace[-1]}})
    except Exception as exc:
        print(f"[langmem] manager.invoke failed: {exc}", file=sys.stderr)


def _answer_group(group_key: str, instances: list[Instance], reader_model: str) -> list[Hypothesis]:
    if not instances:
        return []
    try:
        from langmem import create_memory_store_manager
        from langgraph.store.memory import InMemoryStore
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise ImportError(
            "langmem baseline requires `pip install langmem langgraph langchain-openai`"
        ) from e

    # The default manager keeps only 1-2 summaries per session; extract facts instead.
    class Fact(BaseModel):
        """A specific, atomic fact extracted from the conversation."""
        subject: str = Field(description="Who or what the fact is about")
        fact: str = Field(description="The fact in one sentence; preserve "
                                       "specific names, dates, places, quantities verbatim")
        when: str | None = Field(default=None,
                                  description="When this fact applies, ISO date if stated")

    INSTRUCTIONS = (
        "You are extracting atomic, verifiable facts from a conversation "
        "between two people. Be exhaustive and concrete: one Fact per "
        "durable claim — names, dates, places, items, decisions, "
        "quantities. Preserve specific words from the conversation when "
        "possible (e.g. 'Pomodoro technique', '21 October 2023'). Don't "
        "summarise multi-fact claims into one Fact — split them. Don't "
        "repeat facts already stored."
    )

    namespace = ("memories", _safe_ns(group_key))
    store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})
    manager = create_memory_store_manager(
        ChatOpenAI(model="gpt-4o-mini"),
        schemas=[Fact],
        instructions=INSTRUCTIONS,
        namespace=namespace,
        store=store,
    )

    def _hit_to_text(h) -> str:
        val = h.get("value", h) if isinstance(h, dict) else getattr(h, "value", h)
        if not isinstance(val, dict):
            return str(val)
        inner = val.get("content")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, dict):
            subj = inner.get("subject", "")
            fact = inner.get("fact") or inner.get("content") or ""
            when = inner.get("when")
            return f"{subj}: {fact}" + (f" (when: {when})" if when else "")
        return json.dumps(val)

    out: list[Hypothesis] = []
    roles = chat_roles(instances[0])
    for s in instances[0].sessions:
        _ingest_with_langmem(manager, namespace, s.turns, roles)

    for inst in instances:
        try:
            hits = store.search(namespace, query=inst.question.question, limit=10)
            memories = [_hit_to_text(h) for h in (hits or [])]
            ans = _compose_answer(reader_model, inst.question.question, memories)
        except Exception as exc:
            ans = f"ERROR: {type(exc).__name__}: {exc}"
        out.append(Hypothesis(
            question_id=inst.question.question_id,
            hypothesis=ans,
            category=inst.question.category,
            gold=gold_text(inst),
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
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
        instances = [inst for inst in instances if inst.question.question_id not in done_ids]

    grouped: dict[str, list[Instance]] = collections.defaultdict(list)
    for inst in instances:
        grouped[_group_key(inst)].append(inst)

    print(
        f"[langmem] locomo: {len(instances)} instances, {len(grouped)} haystacks",
        file=sys.stderr,
    )

    t0 = time.time()
    results: list[Hypothesis] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("w") as fout:
        for line in kept_lines:
            fout.write(line + "\n")
        futs = {
            pool.submit(_answer_group, key, group, args.reader_model): key
            for key, group in grouped.items()
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="langmem:haystacks"):
            group_results = fut.result()
            results.extend(group_results)
            for h in group_results:
                fout.write(json.dumps(asdict(h), ensure_ascii=False) + "\n")
            fout.flush()

    print(
        f"[langmem] wrote {len(results)} new + {len(done_ids)} resumed "
        f"in {time.time() - t0:.0f}s -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
