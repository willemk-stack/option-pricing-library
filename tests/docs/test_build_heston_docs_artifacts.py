from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.build_heston_docs_artifacts as build_heston_docs_artifacts


def _profile_quotes(profile_name: str):
    profile = build_heston_docs_artifacts.PROFILE_DEFAULTS[profile_name]
    return build_heston_docs_artifacts.build_market_like_heston_quote_set(
        expiries=np.asarray(profile.expiries, dtype=np.float64),
        log_moneyness=np.asarray(profile.log_moneyness, dtype=np.float64),
    )


def _bucket_names(log_moneyness: np.ndarray) -> set[str]:
    buckets: set[str] = set()
    for value in np.asarray(log_moneyness, dtype=np.float64):
        if value < -0.03:
            buckets.add("downside_wing")
        elif value > 0.03:
            buckets.add("upside_wing")
        else:
            buckets.add("atm")
    return buckets


def test_release_held_out_mask_is_stratified_and_deterministic() -> None:
    profile = build_heston_docs_artifacts.PROFILE_DEFAULTS["release"]
    quotes = _profile_quotes("release")

    mask_a, indices_a, labels_a = (
        build_heston_docs_artifacts._build_deterministic_held_out_mask(quotes)
    )
    mask_b, indices_b, labels_b = (
        build_heston_docs_artifacts._build_deterministic_held_out_mask(quotes)
    )

    assert np.array_equal(mask_a, mask_b)
    assert indices_a == indices_b
    assert labels_a == labels_b
    assert 7 <= len(indices_a) <= 9
    assert len(indices_a) == int(np.count_nonzero(mask_a))
    assert len(indices_a) < quotes.n_quotes

    selected_expiries = {float(quotes.expiry[index]) for index in indices_a}
    assert float(profile.expiries[0]) in selected_expiries
    assert float(profile.expiries[len(profile.expiries) // 2]) in selected_expiries
    assert float(profile.expiries[-1]) in selected_expiries

    selected_log_m = np.asarray(
        [float(quotes.log_moneyness[index]) for index in indices_a],
        dtype=np.float64,
    )
    assert _bucket_names(selected_log_m) == {
        "atm",
        "downside_wing",
        "upside_wing",
    }


def test_smoke_held_out_mask_degrades_gracefully() -> None:
    profile = build_heston_docs_artifacts.PROFILE_DEFAULTS["smoke"]
    quotes = _profile_quotes("smoke")

    mask, indices, labels = (
        build_heston_docs_artifacts._build_deterministic_held_out_mask(quotes)
    )

    assert len(indices) >= 3
    assert len(indices) == len(labels)
    assert len(indices) == int(np.count_nonzero(mask))
    assert len(indices) < quotes.n_quotes
    assert {float(quotes.expiry[index]) for index in indices} == {
        float(expiry) for expiry in profile.expiries
    }

    selected_log_m = np.asarray(
        [float(quotes.log_moneyness[index]) for index in indices],
        dtype=np.float64,
    )
    assert _bucket_names(selected_log_m) == {
        "atm",
        "downside_wing",
        "upside_wing",
    }


def test_build_heston_docs_artifacts_smoke_profile_writes_required_bundle(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "generated" / "heston"

    exit_code = build_heston_docs_artifacts.main(
        [
            "--profile",
            "smoke",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    manifest_path = output_dir / "data" / "heston_artifact_manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "smoke"
    assert payload["fixture"]["data_source"] == "synthetic_not_market_data"
    assert payload["fixture"]["generated_from_heston"] is False
    assert len(payload["held_out_policy"]["indices"]) >= 3
    assert len(payload["held_out_policy"]["labels"]) == len(
        payload["held_out_policy"]["indices"]
    )
    assert "stratified" in payload["held_out_policy"]["description"].lower()
    assert "three-point split" not in payload["held_out_policy"]["description"].lower()
    assert "not real market data" in payload["caveat"]

    held_out_svg = (
        output_dir / "heston_train_vs_heldout_comparison.light.svg"
    ).read_text(encoding="utf-8")
    assert "Train vs stratified held-out diagnostic" in held_out_svg
    assert (
        "Stratified deterministic holdout across expiry and moneyness; not a market-date out-of-sample test."
        in held_out_svg
    )
    assert "Direct local-vol PDE" not in held_out_svg
    assert "n=" in held_out_svg

    artifacts_by_filename = {
        artifact["filename"]: artifact for artifact in payload["artifacts"]
    }
    tradeoff_summary = artifacts_by_filename["heston_comparison_tradeoff_summary.csv"]
    assert tradeoff_summary["artifact_type"] == "data"
    assert tradeoff_summary["source"] == "heston_comparison_tradeoff_summary.csv"
    assert tradeoff_summary["fixture_is_synthetic"] is True
    assert tradeoff_summary["caveats"]
    assert (
        "docs/user_guides/heston_model_comparison.md" in tradeoff_summary["docs_pages"]
    )

    mc_convergence = artifacts_by_filename["heston_mc_convergence_summary.csv"]
    assert mc_convergence["artifact_type"] == "data"
    assert mc_convergence["source"] == "heston_mc_convergence_summary.csv"
    assert mc_convergence["fixture_is_synthetic"] is True
    assert mc_convergence["caveats"]
    assert "docs/performance.md" in mc_convergence["docs_pages"]

    multistart_panel = artifacts_by_filename["heston_multistart_stability_panel.png"]
    assert multistart_panel["artifact_type"] == "figure"
    assert multistart_panel["source"] == "heston_multistart_stability_panel"
    assert multistart_panel["fixture_is_synthetic"] is True
    assert multistart_panel["caption"]
    assert "docs/validation_matrix.md" in multistart_panel["docs_pages"]

    freshness_path = (
        output_dir
        / "data"
        / build_heston_docs_artifacts.HESTON_ARTIFACT_FRESHNESS_MANIFEST
    )
    assert freshness_path.exists()

    freshness_payload = json.loads(freshness_path.read_text(encoding="utf-8"))
    assert freshness_payload["profile"] == "smoke"
    assert freshness_payload["formats"] == ["svg", "png"]
    assert (
        freshness_payload["rebuild_command"]
        == build_heston_docs_artifacts.HESTON_ARTIFACT_REBUILD_COMMAND
    )
    assert set(freshness_payload["source_inputs"]) == set(
        build_heston_docs_artifacts.HESTON_ARTIFACT_SOURCE_INPUTS
    )
    assert any(
        item["path"] == "data/heston_artifact_manifest.json"
        for item in freshness_payload["generated_files"]
    )

    expected_files = {
        "heston_comparison_summary_card.svg",
        "heston_comparison_summary_card.light.svg",
        "heston_comparison_summary_card.dark.svg",
        "heston_comparison_summary_card.png",
        "heston_comparison_summary_card.light.png",
        "heston_comparison_summary_card.dark.png",
        "heston_model_comparison_smile_overlay.svg",
        "heston_model_comparison_smile_overlay.light.svg",
        "heston_model_comparison_smile_overlay.dark.svg",
        "heston_model_comparison_smile_overlay.png",
        "heston_model_comparison_smile_overlay.light.png",
        "heston_model_comparison_smile_overlay.dark.png",
        "heston_iv_residual_heatmap.svg",
        "heston_iv_residual_heatmap.light.svg",
        "heston_iv_residual_heatmap.dark.svg",
        "heston_iv_residual_heatmap.png",
        "heston_iv_residual_heatmap.light.png",
        "heston_iv_residual_heatmap.dark.png",
        "heston_model_comparison_error_buckets.svg",
        "heston_model_comparison_error_buckets.light.svg",
        "heston_model_comparison_error_buckets.dark.svg",
        "heston_model_comparison_error_buckets.png",
        "heston_model_comparison_error_buckets.light.png",
        "heston_model_comparison_error_buckets.dark.png",
        "heston_multistart_stability_panel.svg",
        "heston_multistart_stability_panel.light.svg",
        "heston_multistart_stability_panel.dark.svg",
        "heston_multistart_stability_panel.png",
        "heston_multistart_stability_panel.light.png",
        "heston_multistart_stability_panel.dark.png",
        "heston_mc_vs_fourier_convergence.svg",
        "heston_mc_vs_fourier_convergence.light.svg",
        "heston_mc_vs_fourier_convergence.dark.svg",
        "heston_mc_vs_fourier_convergence.png",
        "heston_mc_vs_fourier_convergence.light.png",
        "heston_mc_vs_fourier_convergence.dark.png",
        "heston_train_vs_heldout_comparison.svg",
        "heston_train_vs_heldout_comparison.light.svg",
        "heston_train_vs_heldout_comparison.dark.svg",
        "heston_train_vs_heldout_comparison.png",
        "heston_train_vs_heldout_comparison.light.png",
        "heston_train_vs_heldout_comparison.dark.png",
        "heston_workflow_architecture.svg",
        "heston_workflow_architecture.light.svg",
        "heston_workflow_architecture.dark.svg",
        "heston_comparison_fit_errors.csv",
        "heston_comparison_error_summary.csv",
        "heston_comparison_heldout.csv",
        "heston_comparison_direct_local_vol_pde.csv",
        "heston_comparison_direct_pde_matched_error_summary.csv",
        "heston_comparison_tradeoff_summary.csv",
        "heston_mc_convergence_summary.csv",
        "heston_artifact_manifest.json",
        "heston_artifact_freshness.json",
    }

    artifact_names = {artifact["filename"] for artifact in payload["artifacts"]}
    assert expected_files <= artifact_names

    expected_paths = [
        output_dir / "heston_comparison_summary_card.svg",
        output_dir / "heston_model_comparison_smile_overlay.png",
        output_dir / "heston_workflow_architecture.svg",
        output_dir / "data" / "heston_comparison_fit_errors.csv",
        output_dir / "data" / "heston_comparison_direct_pde_matched_error_summary.csv",
        output_dir / "data" / "heston_mc_convergence_summary.csv",
    ]
    for path in expected_paths:
        assert path.exists(), path

    stdout = capsys.readouterr().out.strip()
    assert stdout.endswith("data/heston_artifact_manifest.json")


def test_themed_architecture_svgs_use_bounded_text_blocks(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated" / "heston"

    paths = build_heston_docs_artifacts._write_themed_architecture_svgs(output_dir)

    assert output_dir / "heston_workflow_architecture.light.svg" in paths
    assert output_dir / "heston_workflow_architecture.dark.svg" in paths
    assert output_dir / "heston_workflow_architecture.svg" in paths

    for filename in (
        "heston_workflow_architecture.light.svg",
        "heston_workflow_architecture.dark.svg",
        "heston_workflow_architecture.svg",
    ):
        content = (output_dir / filename).read_text(encoding="utf-8")
        assert '<title id="title">Heston workflow architecture</title>' in content
        assert '<desc id="desc">' in content
        assert 'id="architecture-card-heston-body"' in content
        assert 'id="architecture-footer"' in content
        assert "<tspan" in content
        assert "Directional evidence, not universal speed claims" not in content
        assert "Bias, CI, and runtime/error tradeoff" not in content
        assert "Synthetic quote target" in content
        assert "Model-choice comparison" in content
        assert (
            "Capstone 3 compares Heston against the eSSVI/local-vol baseline" in content
        )


def test_heston_docs_pages_reference_theme_aware_asset_pairs() -> None:
    root = Path(build_heston_docs_artifacts.ROOT)
    expected_references = {
        Path("README.template.md"): [
            "heston_comparison_summary_card.light.png#gh-light-mode-only",
            "heston_comparison_summary_card.dark.png#gh-dark-mode-only",
        ],
        Path("README.md"): [
            "heston_comparison_summary_card.light.png#gh-light-mode-only",
            "heston_comparison_summary_card.dark.png#gh-dark-mode-only",
        ],
        Path("docs/index.md"): [
            "heston_comparison_summary_card.light.png",
            "heston_comparison_summary_card.dark.png",
        ],
        Path("docs/performance.md"): [
            "heston_mc_vs_fourier_convergence.light.png",
            "heston_mc_vs_fourier_convergence.dark.png",
        ],
        Path("docs/user_guides/heston_model_comparison.md"): [
            "heston_comparison_summary_card.light.png",
            "heston_comparison_summary_card.dark.png",
            "heston_model_comparison_smile_overlay.light.png",
            "heston_model_comparison_smile_overlay.dark.png",
            "heston_multistart_stability_panel.light.png",
            "heston_multistart_stability_panel.dark.png",
            "heston_mc_vs_fourier_convergence.light.png",
            "heston_mc_vs_fourier_convergence.dark.png",
        ],
    }

    for relative_path, references in expected_references.items():
        content = (root / relative_path).read_text(encoding="utf-8")
        for reference in references:
            assert reference in content, f"Missing {reference} in {relative_path}"
