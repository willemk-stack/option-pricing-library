from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXAMPLE_SCRIPTS = [
    ROOT / "examples" / "quickstart.py",
    ROOT / "examples" / "curves_first.py",
    ROOT / "examples" / "implied_vol.py",
]

DOC_SNIPPETS = [
    ROOT / "docs" / "api" / "index.md",
    ROOT / "docs" / "user_guides" / "market_api.md",
]


def _run_markdown_python_blocks(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    assert blocks, f"No python blocks found in {path}"

    ns: dict[str, object] = {"__name__": "__main__"}
    for block in blocks:
        code = block.strip()
        if not code:
            continue
        exec(compile(code, str(path), "exec"), ns, ns)


def test_run_examples_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OP_FAST_EXAMPLES", "1")
    for script in EXAMPLE_SCRIPTS:
        runpy.run_path(str(script), run_name="__main__")


def test_run_doc_snippets_smoke() -> None:
    for doc in DOC_SNIPPETS:
        _run_markdown_python_blocks(doc)
