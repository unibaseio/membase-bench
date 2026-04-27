"""Bench runner: per instance, build a fresh Memory, ingest sessions, answer the question."""

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


def _load_instances(bench: str, limit: int | None) -> list[Instance]:
    if bench in ("longmemeval_s", "longmemeval_m", "longmemeval_oracle"):
        from bench.longmemeval.adapter import load as load_lme
        split = {
            "longmemeval_s": "longmemeval_s",
            "longmemeval_m": "longmemeval_m",
            "longmemeval_oracle": "longmemeval_oracle",
        }[bench]
        return load_lme(split=split, limit=limit)
    if bench == "locomo":
        from bench.locomo.adapter import load as load_loco
        return load_loco(limit=limit)
    raise SystemExit(f"unknown bench: {bench}")


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
    print(f"[runner] sample: {len(out)} instances across {len(by_cat)} categories "
          f"({n_per_cat} per category, seed={seed})", file=sys.stderr)
    for cat in sorted(by_cat):
        kept = min(n_per_cat, len(by_cat[cat]))
        print(f"  {cat:32s} {kept}/{len(by_cat[cat])}", file=sys.stderr)
    return out


def _process_instance(
    inst: Instance,
    work_dir: str,
    mem_kwargs: dict,
    *,
    pre_memories: list[dict] | None = None,
    max_steps: int = 1,
) -> Hypothesis:
    from memory import Memory
    from memory.pipeline.ingest import materialise_observations

    db_path = os.path.join(work_dir, f"{inst.instance_id}.db")
    idx_path = db_path + ".faiss"
    try:
        with Memory(db_path=db_path, index_path=idx_path, **mem_kwargs) as mem:
            if pre_memories is None:
                # Default end-to-end path: ingest sessions, run our Observer.
                mem.ingest_sessions_parallel([
                    {
                        "session_id": s.session_id,
                        "session_date": s.session_date,
                        "turns": s.turns,
                    }
                    for s in inst.sessions
                ])
            else:
                # memory-only mode: skip Observer/Linker; only materialise the
                # provided memory list as observations. We still need raw turns
                # in the index so the turns / sessions retrieval paths can fire.
                mem.ingest_sessions_parallel([
                    {
                        "session_id": s.session_id,
                        "session_date": s.session_date,
                        "turns": s.turns,
                    }
                    for s in inst.sessions
                ])
                # Synthetic session for the foreign memories: one row, no LLM,
                # all observations are tagged with this session_id.
                mem_session_id = f"{inst.instance_id}-pre"
                mem_session_date = inst.question.question_date or "1970-01-01T00:00"
                materialise_observations(
                    mem.conn, mem.faiss, pre_memories,
                    turn_ids=[],
                    session_id=mem_session_id,
                    session_date=mem_session_date,
                )
            ans = mem.answer(
                inst.question.question,
                query_date=inst.question.question_date,
                max_steps=max_steps,
            )
        return Hypothesis(
            question_id=inst.question.question_id,
            hypothesis=ans,
            category=inst.question.category,
            gold=inst.question.answer if isinstance(inst.question.answer, str) else json.dumps(inst.question.answer),
        )
    except Exception as e:
        return Hypothesis(
            question_id=inst.question.question_id,
            hypothesis=f"ERROR: {type(e).__name__}: {e}",
            category=inst.question.category,
            gold=inst.question.answer if isinstance(inst.question.answer, str) else json.dumps(inst.question.answer),
        )
    finally:
        for p in (db_path, idx_path, db_path + "-wal", db_path + "-shm"):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True,
                    choices=["longmemeval_s", "longmemeval_m", "longmemeval_oracle", "locomo"])
    ap.add_argument("--out", required=True, help="path to write hypotheses jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap number of instances (smoke runs)")
    ap.add_argument("--workers", type=int, default=4, help="parallel instances")
    ap.add_argument("--no-observer", action="store_true",
                    help="disable Observer extraction (Mempal raw-only mode)")
    ap.add_argument("--no-linker", action="store_true",
                    help="disable supersede linker (ablation)")
    ap.add_argument("--no-summary-doc", action="store_true",
                    help="disable per-session aggregate doc (ablation)")
    ap.add_argument("--markdown-digest", action="store_true",
                    help="build a conversation-level markdown digest during ingest and feed it to the reader")
    ap.add_argument("--no-cross-encoder", action="store_true",
                    help="disable cross-encoder reranker")
    ap.add_argument("--no-llm-rerank", action="store_true",
                    help="disable LLM rerank (pick-1 of top-K sessions)")
    ap.add_argument("--workdir", default=None, help="scratch dir for per-instance dbs")
    ap.add_argument("--resume", action="store_true",
                    help="skip question_ids already in --out (only re-run errors)")
    ap.add_argument("--sample-per-category", type=int, default=None,
                    help="random stratified sample N instances per question category")
    ap.add_argument("--seed", type=int, default=42, help="rng seed for sampling")
    ap.add_argument("--qids-file", default=None,
                    help="JSON file with a list of question_ids to run (regression set)")
    ap.add_argument("--qids-key", default="all_failed",
                    help="key inside the qids-file dict (e.g. 'all_failed' or 'hard_in_all_versions')")
    ap.add_argument("--memory-only", default=None,
                    help="path to Memori-format augmented memories JSON (or 'auto' to "
                         "download the Memori LoCoMo set). Skips our Observer and "
                         "ingests these memories as observations; lets us compare "
                         "retrieval+reader on the same memory set Memori reports against. "
                         "LoCoMo only.")
    ap.add_argument("--max-steps", type=int, default=1,
                    help="reader budget. 1 = one-shot reader (default). >1 enables ReAct "
                         "loop with raw_turn_search / raw_session_read / memory_search tools.")
    args = ap.parse_args(argv)

    instances = _load_instances(args.bench, args.limit)
    if args.qids_file:
        with open(args.qids_file) as f:
            data = json.load(f)
        qids = data[args.qids_key] if isinstance(data, dict) else data
        wanted = set(qids)
        instances = [i for i in instances if i.question.question_id in wanted]
        print(f"[runner] qids-file: kept {len(instances)} matching '{args.qids_key}'", file=sys.stderr)
    if args.sample_per_category is not None:
        instances = _stratified_sample(instances, args.sample_per_category, args.seed)
    print(f"[runner] {args.bench}: {len(instances)} instances loaded", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: read existing successful hypotheses; drop errors so they re-run.
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
        print(f"[runner] resume: {len(done_ids)} kept, {len(instances)} to run", file=sys.stderr)

    work_dir = args.workdir or tempfile.mkdtemp(prefix=f"unibase-{args.bench}-")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    pre_memory_lookup: dict[str, list[dict]] | None = None
    if args.memory_only is not None:
        if args.bench != "locomo":
            raise SystemExit("--memory-only is only supported for --bench locomo")
        from bench.locomo import memori_memories
        path = None if args.memory_only == "auto" else args.memory_only
        pre_memory_lookup = memori_memories.load(path=path)
        print(
            f"[runner] memory-only: loaded {sum(len(v) for v in pre_memory_lookup.values())} "
            f"memories across {len(pre_memory_lookup)} conversations",
            file=sys.stderr,
        )
        # Memory-only implies skipping our Observer + Linker. The summary doc stays
        # on so the sessions retrieval path still has something to rank.
        args.no_observer = True
        args.no_linker = True

    mem_kwargs = dict(
        enable_observer=not args.no_observer,
        enable_linker=not args.no_linker,
        enable_summary_doc=not args.no_summary_doc,
        enable_markdown_digest=args.markdown_digest,
        enable_cross_encoder=not args.no_cross_encoder,
        enable_llm_rerank=not args.no_llm_rerank,
    )

    t0 = time.time()
    results: list[Hypothesis] = []
    mode = "w" if not args.resume else "w"
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open(mode) as fout:
        if args.resume and kept_lines:
            for line in kept_lines:
                fout.write(line + "\n")
            fout.flush()
        futures = {}
        for inst in instances:
            pre_mem = None
            if pre_memory_lookup is not None:
                from bench.locomo.memori_memories import conv_id_from_instance
                pre_mem = pre_memory_lookup.get(conv_id_from_instance(inst.instance_id), [])
            futures[pool.submit(
                _process_instance,
                inst,
                work_dir,
                mem_kwargs,
                pre_memories=pre_mem,
                max_steps=args.max_steps,
            )] = inst
        for fut in tqdm(as_completed(futures), total=len(futures), desc=args.bench):
            h = fut.result()
            results.append(h)
            fout.write(json.dumps(asdict(h)) + "\n")
            fout.flush()

    print(
        f"[runner] {args.bench}: {len(results)} new + {len(done_ids)} resumed "
        f"in {time.time()-t0:.0f}s -> {out_path}",
        file=sys.stderr,
    )
    if not args.workdir:
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
