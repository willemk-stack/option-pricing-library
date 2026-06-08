from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROVIDER_PAGE_REL = "provider_market_evidence.md"
OPTIONAL_BRANCH_PAGES = [
    ROOT / "docs" / "user_guides" / "marketdata_cli.md",
    ROOT / "docs" / "user_guides" / "model_ready_heston_workflow.md",
]


@pytest.mark.parametrize("path", OPTIONAL_BRANCH_PAGES)
def test_optional_provider_branch_pages_link_to_real_market_provider_evidence(
    path: Path,
) -> None:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} is not present on this branch")

    text = path.read_text(encoding="utf-8")
    assert PROVIDER_PAGE_REL in text, path
