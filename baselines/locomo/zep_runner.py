"""Zep Cloud baseline (https://www.getzep.com/): one Zep session per conversation, messages posted
with ``memory.add``, each question answered over ``memory.search_sessions`` hits.

Zep is cloud-only: ``ZEP_API_KEY=... python -m baselines.locomo.zep_runner --out runs/zep.jsonl``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
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
    safe_id as _safe_session_id,
    speaker_of,
    stratified_sample as _stratified_sample,
)
from bench.common.types import Hypothesis, Instance
from baselines.common.dataset import load


_ANSWER_PROMPT = """\
Answer the question using the supplied recalled facts. Be concise: a few
words or a short phrase. If the question asks for a date, prefer the
format that appears in the recalled facts.

Recalled facts:
{facts}

Question: {question}

Answer:"""


def _compose_answer(model: str, query: str, facts: list[str]) -> str:
    from openai import OpenAI
    client = OpenAI()
    body = "\n".join(f"- {f}" for f in facts) if facts else "(none)"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You answer questions from a long-term memory store."},
            {"role": "user", "content": _ANSWER_PROMPT.format(facts=body, question=query)},
        ],
        max_tokens=128,
    )
    return (resp.choices[0].message.content or "").strip()


def _answer_group(group_key: str, instances: list[Instance], reader_model: str) -> list[Hypothesis]:
    if not instances:
        return []
    api_key = os.environ.get("ZEP_API_KEY")
    if not api_key:
        raise RuntimeError("ZEP_API_KEY is not set; cannot run the Zep baseline.")
    try:
        from zep_cloud.client import Zep
        from zep_cloud.types import Message
    except ImportError as e:
        raise ImportError("zep baseline requires `pip install zep-cloud`") from e

    client = Zep(api_key=api_key)
    session_id = _safe_session_id(group_key)
    user_id = f"locomo-{session_id}"

    try:
        client.user.add(user_id=user_id)
    except Exception:
        pass  # may already exist
    try:
        client.memory.add_session(session_id=session_id, user_id=user_id)
    except Exception:
        pass

    out: list[Hypothesis] = []
    roles = chat_roles(instances[0])
    try:
        for s in instances[0].sessions:
            messages = []
            for t in s.turns:
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                messages.append(Message(role_type=roles[speaker_of(t)], content=content))
            if messages:
                try:
                    client.memory.add(session_id=session_id, messages=messages)
                except Exception as exc:
                    print(f"[zep] memory.add failed on {s.session_id}: {exc}", file=sys.stderr)

        for inst in instances:
            try:
                hits = client.memory.search_sessions(
                    session_ids=[session_id],
                    text=inst.question.question,
                    limit=10,
                )
                facts: list[str] = []
                # The Zep SDK returns search results with a `.message.content`
                # or similar structure depending on version; extract robustly.
                items = getattr(hits, "results", None) or hits or []
                for r in items:
                    msg = getattr(r, "message", None)
                    if msg is not None:
                        facts.append(getattr(msg, "content", str(msg)))
                    else:
                        facts.append(str(r))
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
            client.memory.delete(session_id=session_id)
        except Exception:
            pass


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
        f"[zep] locomo: {len(instances)} instances, {len(grouped)} haystacks",
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
        for fut in tqdm(as_completed(futs), total=len(futs), desc="zep:haystacks"):
            group_results = fut.result()
            results.extend(group_results)
            for h in group_results:
                fout.write(json.dumps(asdict(h)) + "\n")
            fout.flush()

    print(
        f"[zep] wrote {len(results)} new + {len(done_ids)} resumed "
        f"in {time.time() - t0:.0f}s -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
