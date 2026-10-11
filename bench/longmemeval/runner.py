"""Run LongMemEval_S through membase-core, one store per question."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from bench.common.types import Hypothesis, Instance
from bench.longmemeval.adapter import EVAL_OWNER, load
from membase_core import CoreMemoryEngine

NO_CONTEXT = "[NO_CONTEXT]"
ANSWER_PROMPT = "answer_longmemeval"


def _answer(
    inst: Instance, work_dir: str, keep_store: bool, ingest_workers: int = 16
) -> Hypothesis:
    key = inst.instance_id.rsplit("-q", 1)[0]
    db_path = os.path.join(work_dir, f"{key}.db")
    marker = db_path + ".ingested"
    try:
        with CoreMemoryEngine(db_path=db_path, index_path=db_path + ".faiss") as engine:
            if not (keep_store and os.path.exists(marker)):
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
                if keep_store:
                    Path(marker).write_text("ok")
            q = inst.question
            ans = engine.answer_detail(
                q.question,
                q.question_date,
                owner=EVAL_OWNER,
                answer_prompt=ANSWER_PROMPT,
                reader_date_line=True,
            )
            if not ans.retrieval.observations_top:
                return Hypothesis(q.question_id, NO_CONTEXT, q.category, q.answer, [], 0)
            return Hypothesis(
                q.question_id,
                ans.answer,
                q.category,
                q.answer,
                list(ans.context.context_session_ids),
                len(ans.context.observation_ids),
            )
    except Exception as exc:  # noqa: BLE001
        return Hypothesis(
            inst.question.question_id,
            f"ERROR: {type(exc).__name__}: {exc}",
            inst.question.category,
            inst.question.answer,
            [],
            0,
        )
    finally:
        if not keep_store:
            for p in (db_path, db_path + ".faiss", db_path + "-wal", db_path + "-shm", marker):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4, help="questions (stores) in parallel")
    ap.add_argument(
        "--ingest-workers",
        type=int,
        default=16,
        help="LLM calls in flight per question during ingest",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--stride", type=int, default=1, help="take every Nth question (cheap spread sample)"
    )
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    os.environ.setdefault("MEMBASE_ANSWER_MAX_TOKENS", "16384")

    instances = load()
    if args.stride > 1:
        instances = instances[:: args.stride]
    if args.limit is not None:
        instances = instances[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not str(r.get("hypothesis", "")).startswith("ERROR"):
                    done[r["question_id"]] = line
    todo = [i for i in instances if i.question.question_id not in done]
    print(
        f"[lme] {len(instances)} questions, {len(todo)} to run, "
        f"{sum(len(i.sessions) for i in todo)} sessions to ingest",
        file=sys.stderr,
    )

    work_dir = args.workdir or tempfile.mkdtemp(prefix="membase-lme-")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("w") as fout:
            for line in done.values():
                fout.write(line + "\n")
            futs = {
                pool.submit(_answer, i, work_dir, bool(args.workdir), args.ingest_workers): i
                for i in todo
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="lme:questions"):
                h = fut.result()
                n += 1
                fout.write(json.dumps(asdict(h)) + "\n")
                fout.flush()
    finally:
        if not args.workdir:
            shutil.rmtree(work_dir, ignore_errors=True)
    print(
        f"[lme] wrote {n} new + {len(done)} resumed in {time.time() - t0:.0f}s -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
