"""Run DMR through the Membase engine (membase-core public API): retrieve with
``engine.search`` and answer with Zep's prompt verbatim on gpt-4o-mini, outside the engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from bench.common.types import Hypothesis, Instance
from bench.common.llm import chat_text
from bench.dmr.adapter import EVAL_OWNER, load
from membase_core import CoreMemoryEngine, RetrievalResult

ANSWER_MODEL = os.environ.get("DMR_ANSWER_MODEL", "gpt-4o-mini")
SYSTEM = (
    "You are speaker A and should respond to all questions in the the first person perspective of A"
)
PROMPT = """Your task is to briefly answer the question. You are given the following context from the previous conversation. If you don't know how to answer the question, abstain from answering.
    <CONTEXT>
    {context}
    </CONTEXT>
    <QUESTION>
    {question}
    </QUESTION>

Respond with an ANSWER section containing your answer. As well as an EVIDENCE section containing the context that help you came to your conclusion, and an explanation of why that context is relevant.
"""


def _context(result: RetrievalResult) -> str:
    lines = []
    for c in result.observations_top:
        lines.append(f"- {c.subject or ''}: {c.text}" if c.source == "episode" else f"- {c.text}")
    return "\n".join(lines)


def _answer(inst: Instance, work_dir: str, keep: bool, ingest_workers: int) -> Hypothesis:
    key = inst.instance_id.rsplit("-q", 1)[0]
    db = os.path.join(work_dir, f"{key}.db")
    marker = db + ".ingested"
    q = inst.question
    try:
        with CoreMemoryEngine(db_path=db, index_path=db + ".faiss") as engine:
            if not (keep and os.path.exists(marker)):
                engine.ingest_sessions_parallel(
                    [
                        {
                            "session_id": s.session_id,
                            "session_date": s.session_date,
                            "turns": s.turns,
                        }
                        for s in inst.sessions
                    ],
                    max_workers=ingest_workers,
                    episode_owner=EVAL_OWNER,
                )
                if keep:
                    Path(marker).write_text("ok")
            result = engine.search(q.question, q.question_date, owner=EVAL_OWNER)
            ctx = _context(result)
            ans = (
                chat_text(
                    ANSWER_MODEL,
                    SYSTEM,
                    PROMPT.format(context=ctx, question=q.question),
                    max_tokens=512,
                )
                or ""
            )
            return Hypothesis(
                q.question_id,
                ans,
                q.category,
                q.answer,
                [c.session_id for c in result.observations_top if c.session_id],
                len(result.observations_top),
            )
    except Exception as exc:  # noqa: BLE001
        return Hypothesis(
            q.question_id, f"ERROR: {type(exc).__name__}: {exc}", q.category, q.answer, [], 0
        )
    finally:
        if not keep:
            for p in (db, db + ".faiss", db + "-wal", db + "-shm", marker):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ingest-workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workdir")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    inst = load(limit=a.limit)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}
    if a.resume and out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not str(r.get("hypothesis", "")).startswith("ERROR"):
                    done[r["question_id"]] = line
    todo = [i for i in inst if i.question.question_id not in done]
    print(f"[dmr] {len(inst)} questions, {len(todo)} to run", file=sys.stderr)
    wd = a.workdir or tempfile.mkdtemp(prefix="unibase-dmr-")
    Path(wd).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool, out.open("w") as f:
        for line in done.values():
            f.write(line + "\n")
        futs = {pool.submit(_answer, i, wd, bool(a.workdir), a.ingest_workers): i for i in todo}
        for fu in tqdm(as_completed(futs), total=len(futs), desc="dmr"):
            f.write(json.dumps(asdict(fu.result())) + "\n")
            f.flush()
            n += 1
    print(
        f"[dmr] wrote {n} new + {len(done)} resumed in {time.time() - t0:.0f}s -> {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
