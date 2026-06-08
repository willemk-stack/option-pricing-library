from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDER_PAGE = ROOT / "docs" / "user_guides" / "provider_market_evidence.md"

REQUIRED_PAGE_REFERENCES = {
    "provider_evidence_summary_card",
    "provider_public_summary.json",
    "provider_rejection_reason_counts.csv",
    "provider_model_ready_summary.json",
    "provider_heston_fit_summary.csv",
}

REQUIRED_CROSS_LINK_PAGES = [
    ROOT / "docs" / "index.md",
    ROOT / "docs" / "user_guides" / "decision_guide.md",
    ROOT / "docs" / "user_guides" / "market_snapshot_validation.md",
    ROOT / "docs" / "validation_matrix.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "user_guides" / "heston_model_comparison.md",
]

FORBIDDEN_PUBLIC_KEYS = {
    "contracts",
    "latest_quote",
    "api_key",
    "secret",
    "authorization",
    "token",
    "password",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_provider_market_evidence_page_is_in_proof_path_nav() -> None:
    mkdocs = _read(ROOT / "mkdocs.yml")
    assert (
        "Real-market provider evidence: user_guides/provider_market_evidence.md"
        in mkdocs
    )
    assert (
        "Market snapshot validation: user_guides/market_snapshot_validation.md"
        in mkdocs
    )
    assert mkdocs.index(
        "Real-market provider evidence: user_guides/provider_market_evidence.md"
    ) > mkdocs.index("Proof path:")


def test_provider_market_evidence_page_references_required_artifacts() -> None:
    text = _read(PROVIDER_PAGE)
    for expected in REQUIRED_PAGE_REFERENCES:
        assert expected in text


def test_provider_market_evidence_cross_links_are_present() -> None:
    for path in REQUIRED_CROSS_LINK_PAGES:
        assert path.exists(), path
        assert "provider_market_evidence.md" in _read(path), path


def test_public_provider_json_does_not_expose_raw_payload_keys() -> None:
    data_dir = ROOT / "docs" / "assets" / "generated" / "provider_evidence" / "data"
    if not data_dir.exists():
        return
    for path in data_dir.glob("*.json"):
        encoded = json.dumps(json.loads(path.read_text(encoding="utf-8"))).lower()
        for key in FORBIDDEN_PUBLIC_KEYS:
            assert key not in encoded, f"{key} leaked in {path}"


def test_provider_evidence_manifest_has_required_public_contract() -> None:
    manifest_path = (
        ROOT
        / "docs"
        / "assets"
        / "generated"
        / "provider_evidence"
        / "data"
        / "provider_evidence_manifest.json"
    )
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "source_run_id",
        "generated_at",
        "rebuild_command",
        "caveats",
        "artifacts",
    ):
        assert key in payload
    artifact_names = {artifact["filename"] for artifact in payload["artifacts"]}
    expected = {
        "provider_evidence_summary_card.light.svg",
        "provider_evidence_summary_card.dark.svg",
        "provider_pipeline_flow.light.svg",
        "provider_pipeline_flow.dark.svg",
        "provider_quote_cleaning_waterfall.light.png",
        "provider_quote_cleaning_waterfall.dark.png",
        "provider_expiry_strike_coverage.light.png",
        "provider_expiry_strike_coverage.dark.png",
        "provider_heston_fit_summary.light.png",
        "provider_heston_fit_summary.dark.png",
        "provider_public_summary.json",
        "provider_rejection_reason_counts.csv",
        "provider_model_ready_summary.json",
        "provider_heston_fit_summary.csv",
        "provider_heston_parameter_summary.csv",
        "provider_warnings.json",
    }
    assert expected <= artifact_names
