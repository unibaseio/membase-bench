"""Run LoCoMo through mem0 (https://github.com/mem0ai/mem0).

Mirrors the membase_runner output format so the same memori_official_eval
judge can score both side-by-side.

Mapping:
  - Each LoCoMo conversation = one mem0 user_id (one isolated memory).
  - Sessions are added with `Memory.add(messages, user_id=...)`; mem0
    extracts and stores facts internally.
  - Each question runs `Memory.search(query, user_id=...)` then asks an
    OpenAI Chat completion to compose the final answer over the retrieved
    memories. This matches mem0's published LoCoMo recipe.

Install:
  pip install mem0ai

mem0 expects OPENAI_API_KEY in the environment.
"""

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

from unibase_membase.config import load_config
from bench.common.runners import (
    gold_text,
    group_key as _group_key,
    make_retrieval_row,
    safe_id as _safe_user_id,
    stratified_sample as _stratified_sample,
)
from bench.common.types import Hypothesis, Instance
from bench.common.dataset import load


_ANSWER_PROMPT = """\
Answer the question using the supplied memories. Be concise: a few words or
a short phrase. If the question asks for a date, prefer the format that
appears in the memories.

Memories:
{memories}

Question: {question}

Answer:"""


def _compose_answer(model: str, query: str, memories: list[str]) -> str:
    """Use an OpenAI chat call to synthesize an answer from retrieved memories.
    Mirrors mem0's published LoCoMo recipe."""
    from openai import OpenAI
    client = OpenAI()
    body = "\n".join(f"- {m}" for m in memories) if memories else "(none)"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer questions from a long-term memory store."},
            {"role": "user", "content": _ANSWER_PROMPT.format(memories=body, question=query)},
        ],
        max_tokens=128,
    )
    return resp.choices[0].message.content.strip()


def _answer_group(group_key: str, instances: list[Instance], reader_model: str,
                   *, retrieval_only: bool = False, retrieval_top_k: int = 200,
                   persist: bool = False,
                   ) -> list[Hypothesis] | list[dict]:
    if not instances:
        return []
    try:
        from mem0 import Memory
    except ImportError as e:
        raise ImportError("mem0 baseline requires `pip install mem0ai`") from e

    user_id = _safe_user_id(group_key)
    # IMPORTANT: explicit gpt-4o-mini config. mem0 v2.0.1 defaults to a
    # model whose OpenAI API requires `max_completion_tokens`, but mem0
    # still sends `max_tokens` — extraction silently fails with HTTP 400
    # and the store stays empty. Pinning gpt-4o-mini avoids this until
    # mem0 fixes upstream.
    mem = Memory.from_config({
        "llm": {"provider": "openai",
                "config": {"model": "gpt-4o-mini", "temperature": 0}},
        "embedder": {"provider": "openai",
                     "config": {"model": "text-embedding-3-small"}},
    })
    try:
        if persist:
            existing = mem.get_all(filters={"user_id": user_id}, top_k=1)
            existing_rows = existing.get("results", existing) if isinstance(existing, dict) else existing
            already_ingested = bool(existing_rows)
        else:
            already_ingested = False

        # mem0's parse_messages drops any role that isn't user/assistant/system,
        # so LoCoMo speaker names like "Caroline" → empty string → 400 from
        # OpenAI embed. Normalize to a 2-speaker user/assistant alternation,
        # keeping the speaker name as a content prefix for retrieval.
        sessions_to_ingest = [] if already_ingested else instances[0].sessions
        for s in sessions_to_ingest:
            speaker_to_role: dict[str, str] = {}
            messages = []
            for t in s.turns:
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                speaker = t.get("speaker") or t.get("role") or "user"
                if speaker not in speaker_to_role:
                    # First speaker → user, second → assistant.
                    speaker_to_role[speaker] = (
                        "user" if not speaker_to_role else "assistant"
                    )
                messages.append({
                    "role": speaker_to_role[speaker],
                    "content": content,
                })
            if not messages:
                continue
            # mem0's fact-extraction prompt hard-codes datetime.now() as
            # "today", so without an explicit per-session date its LLM
            # rewrites every dated reference into the wall-clock year.
            # Prefix the session timestamp so the extractor anchors facts
            # to LoCoMo's actual conversation date.
            session_date = (s.session_date or "").strip()
            if session_date and messages:
                messages[0] = dict(messages[0])
                messages[0]["content"] = (
                    f"[Session date: {session_date}] " + messages[0]["content"]
                )
            try:
                mem.add(messages, user_id=user_id)
            except Exception as exc:
                print(f"[mem0] add failed on {s.session_id}: {exc}",
                      file=sys.stderr)

        if retrieval_only:
            out_rows: list[dict] = []
            for inst in instances:
                try:
                    results = mem.search(
                        inst.question.question,
                        filters={"user_id": user_id},
                        top_k=retrieval_top_k,
                    )
                    rows = results.get("results", results) if isinstance(results, dict) else results
                    memories = [{
                        "memory": r.get("memory") or r.get("text") or str(r),
                        "score": float(r.get("score", 0.0)) if isinstance(r, dict) else 0.0,
                        "id": str(r.get("id", "")) if isinstance(r, dict) else "",
                        "created_at": r.get("created_at", "") if isinstance(r, dict) else "",
                        "source": "mem0",
                    } for r in (rows or [])]
                except Exception as exc:
                    out_rows.append(make_retrieval_row(
                        inst, memories=None,
                        error=f"{type(exc).__name__}: {exc}",
                    ))
                    continue
                out_rows.append(make_retrieval_row(inst, memories=memories))
            return out_rows

        out: list[Hypothesis] = []
        for inst in instances:
            try:
                results = mem.search(
                    inst.question.question,
                    filters={"user_id": user_id},
                    top_k=10,
                )
                rows = results.get("results", results) if isinstance(results, dict) else results
                memories = []
                for r in rows or []:
                    memories.append(r.get("memory") or r.get("text") or str(r))
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
    finally:
        if not persist:
            try:
                # mem0 keeps state across calls; flush this user's data.
                mem.delete_all(user_id=user_id)
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    cfg = load_config([])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to write hypotheses jsonl")
    ap.add_argument("--workers", type=int, default=2,
                    help="haystacks in parallel (mem0's default vector store may not be thread-safe; keep low)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-category", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reader-model", default=cfg.bench.reader_model)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="Emit retrieval rows (memories list) instead of LLM-composed answers.")
    ap.add_argument("--retrieval-top-k", type=int, default=200,
                    help="When --retrieval-only, how many top memories to emit per question.")
    ap.add_argument("--persist", action="store_true",
                    help="Keep mem0's qdrant store between runs; skip ingest if user_id already populated.")
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
        f"[mem0] locomo: {len(instances)} instances, {len(grouped)} haystacks",
        file=sys.stderr,
    )

    t0 = time.time()
    results: list[Hypothesis] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("w") as fout:
        for line in kept_lines:
            fout.write(line + "\n")
        futs = {
            pool.submit(_answer_group, key, group, args.reader_model,
                         retrieval_only=args.retrieval_only,
                         retrieval_top_k=args.retrieval_top_k,
                         persist=args.persist): key
            for key, group in grouped.items()
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="mem0:haystacks"):
            group_results = fut.result()
            results.extend(group_results)
            for h in group_results:
                rec = h if isinstance(h, dict) else asdict(h)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(
        f"[mem0] wrote {len(results)} new + {len(done_ids)} resumed "
        f"in {time.time() - t0:.0f}s -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
