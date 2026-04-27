"""Orchestrate the LoCoMo head-to-head comparison.

Runs three configurations end-to-end on the same LoCoMo split, judges all of
them with gpt-5-mini, and prints / writes a side-by-side per-category table.

Configurations
--------------
  full       Our full pipeline: ingest sessions -> Observer -> Linker -> Reader.
             This is what we report as our headline number.

  no_obs     Same retrieval and reader, but skip our Observer entirely. Tests
             how much the derived L2 layer is doing for us.

  memori     --memory-only: skip our Observer and ingest the Memori
             advanced_augmented_memories.json directly as observations. Same
             retrieval, same reader; only the source of L2 facts changes. This
             lets us compare directly against the Memori benchmark setup.

The judge model defaults to gpt-5-mini for parity with memU/Memori.

Usage
-----
  uv run python -m bench.locomo.compare --out runs/locomo_compare/
  uv run python -m bench.locomo.compare --out runs/locomo_compare/ --limit 20
  uv run python -m bench.locomo.compare --out runs/locomo_compare/ --skip memori
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIGS = ["full", "no_obs", "memori"]


def _run_runner(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "bench.common.runner", *args]
    print(f"[compare] $ {' '.join(cmd)}", file=sys.stderr)
    subprocess.check_call(cmd)


def _run_eval(hyp_path: Path, judged_path: Path, judge_model: str) -> None:
    cmd = [
        sys.executable, "-m", "bench.locomo.eval",
        str(hyp_path), "--out", str(judged_path),
        "--judge-model", judge_model,
    ]
    print(f"[compare] $ {' '.join(cmd)}", file=sys.stderr)
    subprocess.check_call(cmd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="output directory; one .jsonl + one .judged.jsonl per config")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap instances per config (smoke runs)")
    ap.add_argument("--sample-per-category", type=int, default=None,
                    help="stratified sample N instances per question category")
    ap.add_argument("--seed", type=int, default=42, help="rng seed for sampling")
    ap.add_argument("--qids-file", default=None,
                    help="JSON list of question_ids to run (regression set)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--judge-model", default="gpt-5-mini")
    ap.add_argument("--memori-data", default="auto",
                    help="path to advanced_augmented_memories.json, or 'auto' to download")
    ap.add_argument("--skip", action="append", default=[], choices=CONFIGS,
                    help="skip one or more configurations (repeat the flag to skip several)")
    ap.add_argument("--report-out", default=None,
                    help="optional path to also write a markdown report")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = ["--bench", "locomo", "--workers", str(args.workers)]
    if args.limit is not None:
        common += ["--limit", str(args.limit)]
    if args.sample_per_category is not None:
        common += ["--sample-per-category", str(args.sample_per_category),
                   "--seed", str(args.seed)]
    if args.qids_file is not None:
        common += ["--qids-file", args.qids_file]

    plans = {
        "full":   common[:],
        "no_obs": common + ["--no-observer", "--no-linker"],
        "memori": common + ["--memory-only", args.memori_data],
    }

    judged_paths: dict[str, Path] = {}
    for name, extra in plans.items():
        if name in args.skip:
            print(f"[compare] skipping {name}", file=sys.stderr)
            continue
        hyp_path = out_dir / f"{name}.jsonl"
        judged_path = out_dir / f"{name}.judged.jsonl"
        _run_runner([*extra, "--out", str(hyp_path)])
        _run_eval(hyp_path, judged_path, args.judge_model)
        judged_paths[name] = judged_path

    if not judged_paths:
        print("[compare] nothing to compare", file=sys.stderr)
        return 1

    report = _build_report(judged_paths, judge_model=args.judge_model)
    print("\n" + report)
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(report)
        print(f"\n[compare] wrote {args.report_out}", file=sys.stderr)
    return 0


def _build_report(judged_paths: dict[str, Path], *, judge_model: str) -> str:
    import collections
    import json

    # config -> category -> list[row]
    rows_by_config: dict[str, dict[str, list[dict]]] = {}
    for name, path in judged_paths.items():
        by_cat: dict[str, list[dict]] = collections.defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_cat[r.get("category", "unknown")].append(r)
        rows_by_config[name] = by_cat

    cats = sorted({c for d in rows_by_config.values() for c in d.keys()})
    configs = list(rows_by_config.keys())

    lines: list[str] = []
    lines.append(f"# LoCoMo comparison\n")
    lines.append(f"Judge: `{judge_model}`. Metrics are accuracy (judge) and word-overlap F1.\n")

    # Accuracy table
    lines.append("\n## Accuracy (LLM judge)\n")
    header = "| category " + "".join(f"| {c} " for c in configs) + "| n |"
    sep = "|---" * (len(configs) + 2) + "|"
    lines.append(header)
    lines.append(sep)
    for cat in cats:
        ns = [len(rows_by_config[c].get(cat, [])) for c in configs]
        n = max(ns) if ns else 0
        cells = []
        for c in configs:
            v = rows_by_config[c].get(cat, [])
            if v:
                acc = sum(r.get("correct", False) for r in v) / len(v) * 100
                cells.append(f"{acc:.1f}%")
            else:
                cells.append("—")
        lines.append(f"| {cat} | " + " | ".join(cells) + f" | {n} |")
    # overall
    cells = []
    overall_ns = []
    for c in configs:
        all_rows = [r for v in rows_by_config[c].values() for r in v]
        overall_ns.append(len(all_rows))
        if all_rows:
            acc = sum(r.get("correct", False) for r in all_rows) / len(all_rows) * 100
            cells.append(f"**{acc:.1f}%**")
        else:
            cells.append("—")
    lines.append(f"| **overall** | " + " | ".join(cells) + f" | {max(overall_ns) if overall_ns else 0} |")

    # F1 table
    lines.append("\n## F1 (word overlap)\n")
    lines.append(header)
    lines.append(sep)
    for cat in cats:
        ns = [len(rows_by_config[c].get(cat, [])) for c in configs]
        n = max(ns) if ns else 0
        cells = []
        for c in configs:
            v = rows_by_config[c].get(cat, [])
            if v:
                f1 = sum(r.get("f1", 0.0) for r in v) / len(v) * 100
                cells.append(f"{f1:.1f}")
            else:
                cells.append("—")
        lines.append(f"| {cat} | " + " | ".join(cells) + f" | {n} |")
    cells = []
    for c in configs:
        all_rows = [r for v in rows_by_config[c].values() for r in v]
        if all_rows:
            f1 = sum(r.get("f1", 0.0) for r in all_rows) / len(all_rows) * 100
            cells.append(f"**{f1:.1f}**")
        else:
            cells.append("—")
    lines.append(f"| **overall** | " + " | ".join(cells) + f" | {max(overall_ns) if overall_ns else 0} |")

    lines.append("\n## What each column means\n")
    lines.append("- **full** — our default end-to-end pipeline.")
    lines.append("- **no_obs** — skip our Observer and Linker (raw sessions only).")
    lines.append("- **memori** — skip our Observer and load Memori's `advanced_augmented_memories.json` "
                 "as observations. Direct comparison to the Memori benchmark setup.")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
