"""Serial latency and context size on stores a runner built with --workdir."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import tiktoken
from membase_core import CoreMemoryEngine

from bench.common.llm import chat_text
from bench.dmr import adapter as dmr_adapter
from bench.dmr import runner as dmr_runner
from bench.locomo import adapter as locomo_adapter
from bench.locomo import ours_core
from bench.longmemeval import adapter as lme_adapter
from bench.longmemeval import runner as lme_runner

_ENC = tiktoken.get_encoding("o200k_base")


def _tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000


def _spread(items: list, n: int) -> list:
    return items[:: max(1, len(items) // n)][:n] if items else []


def _built(work_dir: str, key: str) -> str | None:
    db = os.path.join(work_dir, f"{key}.db")
    return db if os.path.exists(db + ".ingested") else None


def _engine_question(engine, q, owner, prompt, date_line) -> tuple[float, float, int, int]:
    t0 = time.perf_counter()
    engine.search(q.question, q.question_date, owner=owner)
    t1 = time.perf_counter()
    detail = engine.answer_detail(
        q.question, q.question_date, owner=owner, answer_prompt=prompt, reader_date_line=date_line
    )
    t2 = time.perf_counter()
    # answer_detail searches again: its time is the end-to-end total, and answer is the rest.
    search, total = _ms(t0, t1), _ms(t1, t2)
    return search, max(0.0, total - search), _tokens(detail.context.text), _tokens(detail.reasoning or detail.answer)


def _locomo(work_dir: str, n: int):
    for inst in _spread([i for i in locomo_adapter.load() if _built(work_dir, ours_core._safe_key(ours_core._group_key(i)))], n):
        db = _built(work_dir, ours_core._safe_key(ours_core._group_key(inst)))
        with CoreMemoryEngine(db_path=db, index_path=db + ".faiss") as engine:
            yield _engine_question(engine, inst.question, inst.question.extra.get("eval_owner") or None,
                                   ours_core.ANSWER_PROMPT, False)


def _longmemeval(work_dir: str, n: int):
    os.environ.setdefault("MEMBASE_ANSWER_MAX_TOKENS", "16384")
    for inst in _spread([i for i in lme_adapter.load() if _built(work_dir, i.instance_id.rsplit("-q", 1)[0])], n):
        db = _built(work_dir, inst.instance_id.rsplit("-q", 1)[0])
        with CoreMemoryEngine(db_path=db, index_path=db + ".faiss") as engine:
            yield _engine_question(engine, inst.question, lme_adapter.EVAL_OWNER, lme_runner.ANSWER_PROMPT, True)


def _dmr(work_dir: str, n: int):
    for inst in _spread([i for i in dmr_adapter.load() if _built(work_dir, i.instance_id.rsplit("-q", 1)[0])], n):
        db = _built(work_dir, inst.instance_id.rsplit("-q", 1)[0])
        q = inst.question
        with CoreMemoryEngine(db_path=db, index_path=db + ".faiss") as engine:
            t0 = time.perf_counter()
            result = engine.search(q.question, q.question_date, owner=dmr_adapter.EVAL_OWNER)
            t1 = time.perf_counter()
            prompt = dmr_runner.PROMPT.format(context=dmr_runner._context(result), question=q.question)
            answer = chat_text(dmr_runner.ANSWER_MODEL, dmr_runner.SYSTEM, prompt, max_tokens=512)
            t2 = time.perf_counter()
        yield _ms(t0, t1), _ms(t1, t2), _tokens(prompt), _tokens(answer)


BENCHES = {"locomo": (_locomo, 40), "longmemeval": (_longmemeval, 30), "dmr": (_dmr, 30)}


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bench", choices=sorted(BENCHES))
    ap.add_argument("--workdir", required=True, help="the --workdir the runner built its stores in")
    ap.add_argument("--limit", type=int, default=None, help="questions, spread over the built stores")
    ap.add_argument("--out", default=None, help="append the raw timings to this JSON file")
    args = ap.parse_args(argv)

    fn, default_n = BENCHES[args.bench]
    rows = list(fn(args.workdir, args.limit or default_n))
    if not rows:
        print(f"no built stores under {args.workdir}; run the {args.bench} runner with --workdir first",
              file=sys.stderr)
        return 1
    search, answer, ctx, out = (list(col) for col in zip(*rows))
    total = [s + a for s, a in zip(search, answer)]
    print(f"{args.bench}: {len(rows)} questions, one at a time")
    for name, vals, unit in (("search", search, "s"), ("answer", answer, "s"), ("total", total, "s")):
        print(f"  {name:<22} p50 {_pct(vals, .5) / 1000:6.2f} {unit}   p95 {_pct(vals, .95) / 1000:6.2f} {unit}")
    print(f"  {'context tokens (mean)':<22} {statistics.mean(ctx):,.0f}")
    print(f"  {'answer tokens (mean)':<22} {statistics.mean(out):,.0f}")
    if args.out:
        path = Path(args.out)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[args.bench] = {"search": search, "answer": answer, "total": total, "ctx": ctx, "outtok": out}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
