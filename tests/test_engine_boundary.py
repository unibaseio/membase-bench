"""The harness may use only membase-core's public API: `membase_core` top-level names and
`membase_core.sources`. Anything deeper is engine-internal and may change between releases."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"membase_core", "membase_core.sources"}
FORBIDDEN_ROOTS = {"memory", "membase_algo"}


def _imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_only_public_engine_api_is_imported():
    bad = []
    for sub in ("bench", "baselines", "tests"):
        for path in (ROOT / sub).rglob("*.py"):
            for mod in _imports(path):
                root = mod.split(".", 1)[0]
                if root in FORBIDDEN_ROOTS or (root == "membase_core" and mod not in ALLOWED):
                    bad.append(f"{path.relative_to(ROOT)}: {mod}")
    assert not bad, "engine internals imported:\n" + "\n".join(bad)
