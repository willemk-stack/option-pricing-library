from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "user_guides" / "model_ready_heston_workflow.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_model_ready_heston_workflow_guide_documents_public_ladder() -> None:
    text = _read(GUIDE)

    assert "pricing-ready versus calibration-ready" in text.lower()
    assert "heston_quotes.parquet" in text
    assert "candidate artifact" in text
    assert "load_model_validation_bundle" in text
    assert "prepare_heston_market_fit" in text
    assert "fit_heston_market" in text
    assert "fit_heston_from_bundle" in text
    assert "option_pricing.marketdata" in text
    assert "option_pricing.workflows" in text


def test_model_ready_heston_workflow_guide_documents_status_contracts() -> None:
    text = _read(GUIDE)

    for status in ("ready", "empty", "blocked", "ok", "failed"):
        assert f"`{status}`" in text
    for field in (
        "selected_quotes",
        "rejected_quotes",
        "stats",
        "preflight",
        "summary",
        "warnings",
        "errors",
    ):
        assert field in text


def test_model_ready_heston_workflow_is_discoverable_from_docs_nav() -> None:
    nav = _read(ROOT / "mkdocs.yml")
    guide_index = _read(ROOT / "docs" / "user_guides" / "index.md")

    assert "user_guides/model_ready_heston_workflow.md" in nav
    assert "model_ready_heston_workflow.md" in guide_index


def test_existing_docs_cross_link_to_model_ready_heston_workflow() -> None:
    expected_cross_links = (
        ROOT / "docs" / "user_guides" / "market_snapshot_validation.md",
        ROOT / "docs" / "user_guides" / "market_api.md",
        ROOT / "docs" / "user_guides" / "heston.md",
        ROOT / "docs" / "user_guides" / "heston_diagnostics.md",
        ROOT / "docs" / "api" / "heston.md",
        ROOT / "docs" / "validation_matrix.md",
    )

    for path in expected_cross_links:
        assert "model_ready_heston_workflow.md" in _read(path), path


def test_docs_do_not_teach_manual_bundle_reconstruction_for_calibration() -> None:
    market_snapshot = _read(
        ROOT / "docs" / "user_guides" / "market_snapshot_validation.md"
    )
    market_api = _read(ROOT / "docs" / "user_guides" / "market_api.md")

    assert "calibrate_heston_multistart" not in market_snapshot
    assert "calibrate_heston_multistart" not in market_api
    assert (
        "`heston_quotes.parquet` is the Heston-compatible quote artifact"
        in market_snapshot
    )
    assert "prepare_heston_market_fit(...)" in market_snapshot
