"""Dependency direction guardrail.

Architecture rule (by intent):

- Core library code (pricing engines, numerics, models, instruments, market) must **not**
  import from ``option_pricing.diagnostics``.
- The diagnostics package is allowed to depend on core (and is expected to).

This test makes that rule enforceable so docs don't drift from reality.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_PREFIX = "option_pricing.diagnostics"

# Only enforce this for the "core" packages.
CORE_SUBPACKAGES = [
    "pricers",
    "models",
    "numerics",
    "market",
    "instruments",
]

CORE_TOPLEVEL_FILES = [
    "__init__.py",
    "types.py",
]


def _iter_core_pyfiles(repo_root: Path) -> list[Path]:
    pkg_root = repo_root / "src" / "option_pricing"

    files: list[Path] = []
    for rel in CORE_TOPLEVEL_FILES:
        p = pkg_root / rel
        if p.exists():
            files.append(p)

    for sub in CORE_SUBPACKAGES:
        subdir = pkg_root / sub
        if not subdir.exists():
            continue
        files.extend([p for p in subdir.rglob("*.py") if p.is_file()])

    return sorted(set(files))


def _forbidden_imports_in_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == FORBIDDEN_PREFIX or name.startswith(FORBIDDEN_PREFIX + "."):
                    hits.append(f"{path}:{node.lineno}: import {name}")

        elif isinstance(node, ast.ImportFrom):
            # Absolute imports
            if node.module:
                mod = node.module
                if mod == FORBIDDEN_PREFIX or mod.startswith(FORBIDDEN_PREFIX + "."):
                    hits.append(f"{path}:{node.lineno}: from {mod} import ...")

                # Relative imports inside option_pricing, e.g. `from ..diagnostics... import ...`
                if node.level >= 1 and mod.split(".")[0] == "diagnostics":
                    hits.append(
                        f"{path}:{node.lineno}: relative import from diagnostics ({node.level} levels)"
                    )

            # Rare case: `from .. import diagnostics`
            if node.level >= 1 and node.module is None:
                for alias in node.names:
                    if alias.name == "diagnostics":
                        hits.append(
                            f"{path}:{node.lineno}: relative import diagnostics ({node.level} levels)"
                        )

    return hits


def test_core_does_not_import_diagnostics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for path in _iter_core_pyfiles(repo_root):
        offenders.extend(_forbidden_imports_in_file(path))

    assert not offenders, (
        "Core packages must not depend on option_pricing.diagnostics.\n"
        "Found forbidden imports:\n- " + "\n- ".join(offenders)
    )
