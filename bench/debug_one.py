"""Debug a single LongMemEval question end-to-end.

Shows everything: ground-truth answer sessions, what got extracted,
what retrieval returned (per path), what the reader saw, what it answered, why.

Usage:
    uv run python -m bench.debug_one <question_id>
    uv run python -m bench.debug_one <question_id> --no-observer  # try raw mode
"""

from __future__ import annotations

import argparse
import os
import tempfile

from bench.longmemeval.adapter import load


def find_instance(qid: str):
    insts = load(split="longmemeval_s", limit=500)
    for i in insts:
        if i.question.question_id == qid:
            return i
    raise SystemExit(f"qid not found: {qid}")


def show_gold_sessions(inst):
    """Print the gold-answer sessions in full so we can compare to what got retrieved."""
    answer_sids = set(inst.question.extra.get("answer_session_ids") or [])
    print(f"\n{'=' * 70}")
    print(f"  GOLD ANSWER SESSIONS ({len(answer_sids)} total)")
    print(f"{'=' * 70}\n")
    for s in inst.sessions:
        if s.session_id in answer_sids:
            print(f"--- {s.session_id} ({s.session_date}) ---")
            for j, t in enumerate(s.turns):
                content = t["content"][:500].replace("\n", " ")
                print(f"  [{j}] {t['role']:9s}: {content}")
            print()


def show_observations(mem, qid):
    rows = mem.conn.execute(
        "SELECT id, text, type, importance, valid_at, invalid_at, subject, session_id "
        "FROM observations ORDER BY id"
    ).fetchall()
    print(f"\n{'=' * 70}")
    print(f"  OBSERVATIONS EXTRACTED ({len(rows)} total)")
    print(f"{'=' * 70}\n")
    for r in rows:
        marker = " [SUPERSEDED]" if r["invalid_at"] else ""
        print(f"  #{r['id']:3d} [{r['type']:18s}] imp={r['importance']} subj={r['subject'] or '?':12s}"
              f" valid={r['valid_at'] or '?':16s} sess={r['session_id'][:12]}{marker}")
        print(f"        {r['text'][:200]}")


def show_retrieval(result, gold_sids):
    print(f"\n{'=' * 70}")
    print("  RETRIEVAL")
    print(f"{'=' * 70}\n")

    print(f"--- top sessions ({len(result.sessions_top)}) ---")
    for c in result.sessions_top:
        hit = " ✅GOLD" if c.session_id in gold_sids else ""
        snippet = c.text[:200].replace("\n", " ")
        print(f"  score={c.score:.3f}  rrf={c.rrf:.3f}  vec={c.vec:.3f}  bm25={c.bm25:.2f}  "
              f"sess={c.session_id[:18]}{hit}")
        print(f"        {snippet}")
    print()

    print(f"--- top observations ({len(result.observations_top)}) ---")
    for c in result.observations_top:
        hit = " ✅GOLD-SESS" if c.session_id in gold_sids else ""
        print(f"  score={c.score:.3f}  type={c.obs_type:18s}  imp={c.importance}  "
              f"valid={c.valid_at or '?':12s}  sess={c.session_id[:12]}{hit}")
        print(f"        {c.text[:200]}")
    print()

    print(f"--- top turns ({len(result.turns_top)}) ---")
    for c in result.turns_top:
        hit = " ✅GOLD-SESS" if c.session_id in gold_sids else ""
        snippet = c.text[:200].replace("\n", " ")
        print(f"  score={c.score:.3f}  sess={c.session_id[:18]}{hit}")
        print(f"        {snippet}")
    print()

    if result.chosen_session_id:
        hit = " ✅GOLD" if result.chosen_session_id in gold_sids else " ❌NOT GOLD"
        print(f"  LLM picked session: {result.chosen_session_id}{hit}")


def show_reader_input(mem, question, query_date, result):
    from memory.recall.reader import pack_context
    full = pack_context(mem.conn, question, query_date, result).text
    print(f"\n{'=' * 70}")
    print("  READER INPUT (what gpt-4o saw)")
    print(f"{'=' * 70}\n")
    print(full[:6000])
    if len(full) > 6000:
        print(f"\n... [truncated {len(full)-6000} chars]")


def show_answer(answer, gold):
    print(f"\n{'=' * 70}")
    print("  ANSWER")
    print(f"{'=' * 70}\n")
    print(f"  GOLD : {gold}")
    print(f"  HYP  : {answer}")
    same = str(answer).strip().lower() == str(gold).strip().lower().strip('"')
    print(f"  match: {'✅' if same else '❌'} (string equality only — judge would be more lenient)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qid", help="LongMemEval question_id")
    ap.add_argument("--no-observer", action="store_true")
    ap.add_argument("--no-linker", action="store_true")
    ap.add_argument("--no-llm-rerank", action="store_true")
    ap.add_argument("--no-cross-encoder", action="store_true")
    ap.add_argument("--keep-db", action="store_true",
                    help="don't delete the DB after run; print path so you can inspect")
    ap.add_argument("--reuse-db", default=None,
                    help="skip ingest, use this existing DB path (e.g. /tmp/debug-XXX/x.db)")
    args = ap.parse_args()

    inst = find_instance(args.qid)
    print(f"\nQuestion ID:   {inst.question.question_id}")
    print(f"Category:      {inst.question.category}")
    print(f"Question:      {inst.question.question}")
    print(f"Asked at:      {inst.question.question_date}")
    print(f"Gold answer:   {inst.question.answer}")
    print(f"Sessions:      {len(inst.sessions)} total")
    print(f"Gold sessions: {inst.question.extra.get('answer_session_ids')}")

    show_gold_sessions(inst)

    if args.reuse_db:
        db_path = args.reuse_db
        work = os.path.dirname(db_path)
        print(f"\n  Reusing DB: {db_path}", flush=True)
    else:
        work = tempfile.mkdtemp(prefix=f"debug-{args.qid}-")
        db_path = os.path.join(work, "x.db")

    from memory import Memory
    mem = Memory(
        db_path=db_path,
        index_path=db_path + ".faiss",
        enable_observer=not args.no_observer,
        enable_linker=not args.no_linker,
        enable_llm_rerank=not args.no_llm_rerank,
        enable_cross_encoder=not args.no_cross_encoder,
    )

    if not args.reuse_db:
        print(f"\n  Ingesting {len(inst.sessions)} sessions ...", flush=True)
        mem.ingest_sessions_parallel([
            {"session_id": s.session_id, "session_date": s.session_date, "turns": s.turns}
            for s in inst.sessions
        ])
        print("  done.")

    if not args.no_observer:
        show_observations(mem, args.qid)

    gold_sids = set(inst.question.extra.get("answer_session_ids") or [])
    result = mem.search(inst.question.question, query_date=inst.question.question_date)
    show_retrieval(result, gold_sids)
    show_reader_input(mem, inst.question.question, inst.question.question_date, result)

    answer = mem.answer(inst.question.question, query_date=inst.question.question_date)
    show_answer(answer, inst.question.answer)

    if args.keep_db:
        # Need to checkpoint the WAL so reuse-db sees the data.
        mem.conn.execute("PRAGMA wal_checkpoint(FULL)")
        mem.close()
        print(f"\nDB kept at: {db_path}")
    else:
        mem.close()
        if not args.reuse_db:
            import shutil
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
