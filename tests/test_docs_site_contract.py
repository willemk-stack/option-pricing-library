from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import docs_site_contract as contract


def test_public_docs_urls_normalize_base_and_root() -> None:
    urls = contract.public_docs_urls(
        "https://example.com/option-pricing-library",
        ["/", "/performance/"],
    )

    assert urls == [
        "https://example.com/option-pricing-library/",
        "https://example.com/option-pricing-library/performance/",
    ]


def test_verify_public_site_requires_non_empty_base_url() -> None:
    with pytest.raises(ValueError, match="non-empty public docs base URL"):
        contract.normalize_public_base_url("   ")
