"""Aggregate per-runner judged.jsonl files into a single comparison table.

Usage:
  python -m bench.locomo.report runs/membase.judged.jsonl runs/mem0.judged.jsonl ...
  membase bench locomo report runs/*.judged.jsonl

Output columns (matching the LoCoMo paper convention):

  Method   Single-hop (%)  Multi-hop (%)  Open-domain (%)  Temporal (%)  Overall (%)

`adversarial` rows (gold = null trick questions) are reported separately
since they invert the scoring axis.

Method label is derived from the filename stem by default; override with
`--label name=path` syntax.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass
from pathlib import Path


# Map our internal category ids to the published-format display labels.
DISPLAY = {
    "single_hop": "Single-hop",
    "multi_hop":  "Multi-hop",
    "open_domain": "Open-domain",
    "temporal":   "Temporal",
}
COLUMN_ORDER = ["Single-hop", "Multi-hop", "Open-domain", "Temporal"]


@dataclass
class Score:
    n: int = 0
    correct: int = 0
    f1_sum: float = 0.0

    @property
    def acc(self) -> float:
        return 100.0 * self.correct / self.n if self.n else 0.0

    @property
    def f1(self) -> float:
        return 100.0 * self.f1_sum / self.n if self.n else 0.0


def _load_judged(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _aggregate(rows: list[dict]) -> tuple[dict[str, Score], Score, Score]:
    """Return (per-category Scores, overall on the 4 main categories,
    adversarial Score)."""
    by_cat: dict[str, Score] = collections.defaultdict(Score)
    for r in rows:
        cat = r.get("category", "")
        s = by_cat[cat]
        s.n += 1
        if r.get("correct"):
            s.correct += 1
        s.f1_sum += float(r.get("f1") or 0.0)

    main = Score()
    for cat in DISPLAY:
        if cat in by_cat:
            s = by_cat[cat]
            main.n += s.n
            main.correct += s.correct
            main.f1_sum += s.f1_sum
    adv = by_cat.get("adversarial") or Score()
    return by_cat, main, adv


def _label_from_path(p: Path) -> str:
    name = p.stem
    for suf in (".judged", ".sample5", ".sample"):
        if name.endswith(suf):
            name = name[:-len(suf)]
    return name


def _parse_label_args(label_args: list[str]) -> dict[str, str]:
    """Parse `--label name=path` entries into {path: name}."""
    out: dict[str, str] = {}
    for la in label_args:
        if "=" not in la:
            raise SystemExit(f"--label expects name=path, got: {la}")
        name, _, path = la.partition("=")
        out[path] = name
    return out


def _format_row(label: str, by_cat: dict[str, Score], overall: Score, *, metric: str = "acc") -> str:
    cells: list[str] = [f"{label:<22s}"]
    for col in COLUMN_ORDER:
        cat_key = next(k for k, v in DISPLAY.items() if v == col)
        s = by_cat.get(cat_key) or Score()
        cells.append(f"{getattr(s, metric):>14.2f}")
    cells.append(f"{getattr(overall, metric):>11.2f}")
    return "  ".join(cells)


def _format_header() -> str:
    cells = [f"{'Method':<22s}"]
    for col in COLUMN_ORDER:
        cells.append(f"{col + ' (%)':>14s}")
    cells.append(f"{'Overall (%)':>11s}")
    return "  ".join(cells)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="judged.jsonl files, one per runner")
    ap.add_argument("--label", action="append", default=[],
                    help="override label: --label Membase=runs/membase.judged.jsonl")
    ap.add_argument("--metric", choices=["acc", "f1"], default="acc",
                    help="primary metric to display (default: acc, matches the LoCoMo paper)")
    ap.add_argument("--show-adversarial", action="store_true",
                    help="add a separate row for adversarial (gold=null) questions")
    args = ap.parse_args(argv)

    overrides = _parse_label_args(args.label)

    # Collect (label, path, by_cat, main, adv) per file.
    rows: list[tuple[str, Path, dict[str, Score], Score, Score]] = []
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"warning: {path} does not exist, skipping", file=sys.stderr)
            continue
        label = overrides.get(p) or _label_from_path(path)
        by_cat, main, adv = _aggregate(_load_judged(path))
        rows.append((label, path, by_cat, main, adv))

    if not rows:
        return 1

    # Header
    print()
    print(_format_header())
    print("-" * len(_format_header()))
    for label, _path, by_cat, main, _adv in rows:
        print(_format_row(label, by_cat, main, metric=args.metric))

    if args.show_adversarial:
        any_adv = any(adv.n for *_, adv in rows)
        if any_adv:
            print()
            print(f"adversarial (gold=null trick questions, lower is the correct outcome)")
            for label, _path, _by_cat, _main, adv in rows:
                print(f"  {label:<20s}  n={adv.n:3d}  acc={adv.acc:6.2f}  f1={adv.f1:6.2f}")

    # Footer: total instance counts per runner (sanity).
    print()
    print("instance counts (Single+Multi+Open+Temporal, excluding adversarial):")
    for label, _path, _by_cat, main, _adv in rows:
        print(f"  {label:<20s}  n={main.n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
