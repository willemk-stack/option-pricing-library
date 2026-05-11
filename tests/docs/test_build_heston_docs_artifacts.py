from __future__ import annotations

import json
from pathlib import Path

import scripts.build_heston_docs_artifacts as build_heston_docs_artifacts


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
