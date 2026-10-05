"""Run LoCoMo through Memori v3 (https://github.com/GibsonAI/memori).

Memori captures memories as a side effect of OpenAI chat calls. Its v3
SDK patches an ``openai.OpenAI`` client through ``Memori().llm.register``;
afterwards every ``client.chat.completions.create(...)`` writes facts
to Memori's storage in the background, and ``mem.recall(query)``
returns those facts as ``FactSearchResult`` rows.

We run Memori in BYODB mode against an isolated per-haystack sqlite
file so it never touches the cloud (no ``MEMORI_API_KEY`` needed).
The schema is created once via ``Builder.execute()`` because v3's
``Memori.__init__`` doesn't auto-migrate when a connection factory is
passed.

Mirrors the membase / mem0 / langmem runners' jsonl output so the
``mem0_eval`` judge can score all of them side-by-side.

Install:
  pip install memori                  # v3 (preferred)
  # or, if only `memorisdk` is on PyPI in your environment, alias the
  # dist-info: cp -r memorisdk-3.X.Y.dist-info memori-3.X.Y.dist-info
  # then sed -i 's/Name: memorisdk/Name: memori/' memori-3.X.Y.dist-info/METADATA

Memori reads ``OPENAI_API_KEY`` from the environment.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import tempfile
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

# Memori prints a deprecation warning about the legacy package name on import;
# silence it so bench output stays clean.
warnings.filterwarnings("ignore", message=".*legacy package name.*")

from tqdm import tqdm

from baselines.common.config import load_config
from baselines.common.runners import (
    group_key as _group_key,
    stratified_sample as _stratified_sample,
)
from bench.common.types import Hypothesis, Instance
from bench.locomo.adapter import load


_ANSWER_PROMPT = """\
Answer the question using the supplied recalled facts. Be concise: a few words
or a short phrase. If the question asks for a date, prefer the format that
appears in the recalled facts.

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
            {"role": "system", "content": "You answer questions from a long-term memory store."},
            {"role": "user", "content": _ANSWER_PROMPT.format(facts=body, question=query)},
        ],
        max_tokens=128,
    )
    return resp.choices[0].message.content.strip()


def _ingest_turns_via_provider(client, turns: list[dict],
                                session_date: str | None = None) -> None:
    """Drive the Memori-patched OpenAI client with each turn so it captures
    the conversation. The patch writes facts to Memori as a side effect of
    ``chat.completions.create``.

    LoCoMo carries free-form speaker names ("John"/"Maria"); OpenAI's chat
    API only accepts user/assistant/system/... so we normalise to a 2-speaker
    alternation, keep the original speaker name as a content prefix, and
    prefix the session date so memori's fact-extraction LLM anchors to the
    correct historical date instead of datetime.now().
    """
    speaker_to_role: dict[str, str] = {}
    history: list[dict] = []
    for i, t in enumerate(turns):
        content = (t.get("content") or "").strip()
        if not content:
            continue
        speaker = t.get("speaker") or t.get("role") or "user"
        if speaker not in speaker_to_role:
            speaker_to_role[speaker] = "user" if not speaker_to_role else "assistant"
        msg_content = f"{speaker}: {content}"
        if i == 0 and session_date:
            msg_content = f"[Session date: {session_date}] {msg_content}"
        history.append({"role": speaker_to_role[speaker], "content": msg_content})
        try:
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=history[-8:],  # short context window for the fake reply
                max_tokens=8,
            )
        except Exception as exc:
            print(f"[memori] provider chat failed: {exc}", file=sys.stderr)


def _answer_group(group_key: str, instances: list[Instance], reader_model: str,
                  retrieval_only: bool, retrieval_top_k: int) -> list:
    if not instances:
        return []
    try:
        from openai import OpenAI
        from memori import Memori
        from memori.storage._builder import Builder
    except ImportError as e:
        raise ImportError("memori baseline requires `pip install memori`") from e

    namespace = f"locomo-{group_key}-{uuid.uuid4().hex[:8]}"
    db_path = tempfile.mktemp(prefix=f"memori-{group_key}-", suffix=".db")
    mem = Memori(conn=lambda: sqlite3.connect(db_path))
    mem.set_session(namespace)
    # BYODB doesn't auto-migrate; Builder.execute() creates the schema.
    Builder(mem.config).disable_banner().execute()

    # v3 patches an openai client to capture memories on every chat call.
    client = OpenAI()
    mem.llm.register(client).attribution(entity_id="locomo-user", process_id="locomo-proc")

    try:
        for s in instances[0].sessions:
            _ingest_turns_via_provider(client, s.turns, session_date=s.session_date)

        out = []
        for inst in instances:
            try:
                rows = mem.recall(
                    inst.question.question,
                    limit=retrieval_top_k if retrieval_only else 10,
                ) or []
                facts: list[str] = []
                memories: list[dict] = []
                for r in rows:
                    if hasattr(r, "content"):
                        text = r.content
                        score = float(getattr(r, "rank_score", getattr(r, "similarity", 0.0)) or 0.0)
                        rid = str(getattr(r, "id", ""))
                        created = str(getattr(r, "date_created", ""))
                    elif isinstance(r, dict):
                        text = r.get("content") or r.get("fact") or r.get("text") or json.dumps(r)
                        score = float(r.get("rank_score") or r.get("score") or 0.0)
                        rid = str(r.get("id", ""))
                        created = str(r.get("date_created", ""))
                    else:
                        text = str(r); score = 0.0; rid = ""; created = ""
                    facts.append(text)
                    memories.append({"memory": text, "score": score, "id": rid,
                                     "created_at": created, "source": "memori"})

                if retrieval_only:
                    out.append({
                        "question_id": inst.question.question_id,
                        "category": inst.question.category,
                        "question": inst.question.question,
                        "gold": (
                            inst.question.answer
                            if isinstance(inst.question.answer, str)
                            else json.dumps(inst.question.answer)
                        ),
                        "reference_date": inst.question.question_date,
                        "memories": memories,
                    })
                else:
                    ans = _compose_answer(reader_model, inst.question.question, facts)
                    out.append(asdict(Hypothesis(
                        question_id=inst.question.question_id,
                        hypothesis=ans,
                        category=inst.question.category,
                        gold=(
                            inst.question.answer
                            if isinstance(inst.question.answer, str)
                            else json.dumps(inst.question.answer)
                        ),
                    )))
            except Exception as exc:
                if not retrieval_only:
                    out.append(asdict(Hypothesis(
                        question_id=inst.question.question_id,
                        hypothesis=f"ERROR: {type(exc).__name__}: {exc}",
                        category=inst.question.category,
                        gold=str(inst.question.answer),
                    )))
        return out
    finally:
        try:
            mem.close()
        except Exception:
            pass
        for path in (db_path, db_path + "-wal", db_path + "-shm"):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    cfg = load_config([])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-category", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reader-model", default=cfg.bench.reader_model)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="Emit retrieval rows (memories list) instead of LLM answers.")
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
        instances = [inst for inst in instances if inst.question.question_id not in done_ids]

    grouped: dict[str, list[Instance]] = collections.defaultdict(list)
    for inst in instances:
        grouped[_group_key(inst)].append(inst)

    print(
        f"[memori] locomo: {len(instances)} instances, {len(grouped)} haystacks",
        file=sys.stderr,
    )

    t0 = time.time()
    results: list = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("w") as fout:
        for line in kept_lines:
            fout.write(line + "\n")
        futs = {
            pool.submit(_answer_group, key, group, args.reader_model,
                        args.retrieval_only, args.retrieval_top_k): key
            for key, group in grouped.items()
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="memori:haystacks"):
            group_results = fut.result()
            results.extend(group_results)
            for h in group_results:
                fout.write(json.dumps(h) + "\n")
            fout.flush()

    print(
        f"[memori] wrote {len(results)} new + {len(done_ids)} resumed "
        f"in {time.time() - t0:.0f}s -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
