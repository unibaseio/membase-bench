"""Run LoCoMo through the extracted core memory engine."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from bench.common.types import Hypothesis, Instance
from bench.locomo.adapter import load
from memory import CoreMemoryEngine


def _stratified_sample(instances: list[Instance], n_per_cat: int, seed: int) -> list[Instance]:
    rng = random.Random(seed)
    by_cat: dict[str, list[Instance]] = collections.defaultdict(list)
    for inst in instances:
        by_cat[inst.question.category].append(inst)
    out: list[Instance] = []
    for cat in sorted(by_cat):
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        out.extend(pool[:n_per_cat])
    rng.shuffle(out)
    return out


def _group_key(inst: Instance) -> str:
    if "-q" in inst.instance_id:
        return inst.instance_id.rsplit("-q", 1)[0]
    return "|".join(s.session_id for s in inst.sessions) or inst.instance_id


def _safe_key(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)


def _answer_group(group_key: str, instances: list[Instance], work_dir: str) -> list[Hypothesis]:
    if not instances:
        return []
    db_path = os.path.join(work_dir, f"{_safe_key(group_key)}.db")
    idx_path = db_path + ".faiss"
    try:
        with CoreMemoryEngine(db_path=db_path, index_path=idx_path) as engine:
            engine.ingest_sessions_parallel(
                [
                    {
                        "session_id": s.session_id,
                        "session_date": s.session_date,
                        "turns": s.turns,
                    }
                    for s in instances[0].sessions
                ]
            )
            out: list[Hypothesis] = []
            for inst in instances:
                try:
                    ans = engine.answer(
                        inst.question.question,
                        query_date=inst.question.question_date,
                    )
                except Exception as exc:
                    ans = f"ERROR: {type(exc).__name__}: {exc}"
                out.append(Hypothesis(
                    question_id=inst.question.question_id,
                    hypothesis=ans,
                    category=inst.question.category,
                    gold=(
                        inst.question.answer
                        if isinstance(inst.question.answer, str)
                        else json.dumps(inst.question.answer)
                    ),
                ))
            return out
    finally:
        for path in (db_path, idx_path, db_path + "-wal", db_path + "-shm"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to write hypotheses jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-per-category", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qids-file", default=None)
    ap.add_argument("--qids-key", default="all_failed")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    instances = load(limit=args.limit)
    if args.qids_file:
        data = json.loads(Path(args.qids_file).read_text())
        qids = data[args.qids_key] if isinstance(data, dict) else data
        wanted = set(qids)
        instances = [inst for inst in instances if inst.question.question_id in wanted]
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
        f"[ours-core] locomo: {len(instances)} instances, {len(grouped)} haystacks",
        file=sys.stderr,
    )

    work_dir = args.workdir or tempfile.mkdtemp(prefix="unibase-locomo-core-")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results: list[Hypothesis] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("w") as fout:
            for line in kept_lines:
                fout.write(line + "\n")
            futs = {
                pool.submit(_answer_group, key, group, work_dir): key
                for key, group in grouped.items()
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="ours-core:haystacks"):
                group_results = fut.result()
                results.extend(group_results)
                for h in group_results:
                    fout.write(json.dumps(asdict(h)) + "\n")
                fout.flush()
    finally:
        if not args.workdir:
            shutil.rmtree(work_dir, ignore_errors=True)

    print(
        f"[ours-core] wrote {len(results)} new + {len(done_ids)} resumed "
        f"in {time.time() - t0:.0f}s -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
