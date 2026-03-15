from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from option_pricing.demos import (
    build_plot_specs,
    build_visual_bundle,
    export_dataset_vts,
    render_plot_presets,
)
from option_pricing.demos.integration import run_surface_to_localvol_pde_integration


def _required_columns(df: pd.DataFrame, columns: set[str]) -> None:
    assert columns.issubset(df.columns), f"Missing columns: {columns - set(df.columns)}"


@pytest.fixture(scope="module")
def visual_bundle_ci(tmp_path_factory: pytest.TempPathFactory):
    bundle_dir = tmp_path_factory.mktemp("visual_bundle_ci") / "bundle"
    result = build_visual_bundle(
        profile="ci",
        seed=7,
        bundle_dir=bundle_dir,
    )
    return result


def test_visual_bundle_ci_schema(visual_bundle_ci) -> None:
    result = visual_bundle_ci
    manifest = result.manifest

    assert manifest.version == 1
    assert manifest.profile == "ci"
    assert manifest.workflow_profile == "quick"
    assert manifest.manifest_path().exists()

    dataset_names = {dataset.name for dataset in manifest.datasets}
    assert "surface/svi_repaired_grid" in dataset_names
    assert "surface/essvi_smoothed_grid" in dataset_names
    assert "localvol/gatheral_grid" in dataset_names
    assert "repricing/pde_roundtrip_grid" in dataset_names

    smoothed = manifest.resolve_dataset("essvi_smoothed_grid")
    assert smoothed.rectangular_grid
    assert smoothed.axis_columns["i"] == "y"
    assert smoothed.axis_columns["j"] == "T"
    assert smoothed.source_object_type == "ESSVISmoothedSurface"

    svi_df = pd.read_csv(manifest.bundle_root / "surface" / "svi_repaired_grid.csv")
    _required_columns(svi_df, {"T", "F", "y", "moneyness", "K", "iv", "w"})

    smoothed_df = pd.read_csv(
        manifest.bundle_root / "surface" / "essvi_smoothed_grid.csv"
    )
    _required_columns(
        smoothed_df,
        {"T", "F", "y", "moneyness", "K", "iv", "w", "w_T", "w_y", "w_yy"},
    )

    localvol_df = pd.read_csv(manifest.bundle_root / "localvol" / "gatheral_grid.csv")
    _required_columns(
        localvol_df,
        {"T", "F", "y", "K", "sigma_loc", "local_var", "denom", "invalid", "reasons"},
    )

    repricing_df = pd.read_csv(
        manifest.bundle_root / "repricing" / "pde_roundtrip_grid.csv"
    )
    _required_columns(
        repricing_df,
        {
            "T",
            "K",
            "F",
            "moneyness",
            "target_price",
            "pde_price",
            "price_error",
            "target_iv",
            "pde_iv",
            "iv_error_bp",
        },
    )


def test_visual_plot_presets_smoke(
    visual_bundle_ci,
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    result = visual_bundle_ci

    written = render_plot_presets(
        result.manifest,
        presets=["static", "dupire", "poster", "docs", "numerics"],
        out_root=tmp_path / "assets",
    )

    assert written
    assert len(written) == 13
    assert all(path.exists() for path in written)


def test_full_profile_dupire_compare_regression() -> None:
    artifacts = run_surface_to_localvol_pde_integration(
        profile="full",
        seed=7,
        overrides={
            "RUN_*": {
                "RUN_GJ_PAPER_SANITY_CHECK": False,
                "RUN_EXPLICIT_SVI_REPAIR_DEMO": False,
                "RUN_DIGITAL_PDE_BASELINE": False,
                "RUN_LOCALVOL_REPRICING": False,
                "RUN_LOCALVOL_CONVERGENCE_SWEEP": False,
                "RUN_DUPIRE_VS_GATHERAL_COMPARE": True,
                "RUN_ESSVI_TIME_SMOOTHNESS_COMPARE": True,
            }
        },
    )

    report = artifacts.reports["lv_compare"]
    assert report is not None
    assert np.asarray(report.dupire.reason).dtype == np.uint32
    assert report.summary["n_total"] > 0


def test_plot_presets_keep_surface_story_split(visual_bundle_ci) -> None:
    result = visual_bundle_ci
    manifest = result.manifest

    poster_datasets = {
        dataset
        for spec in build_plot_specs(preset="poster")
        for dataset in spec.datasets
    }
    static_datasets = {
        dataset
        for spec in build_plot_specs(preset="static")
        for dataset in spec.datasets
    }

    assert "surface/essvi_smoothed_grid" in poster_datasets
    assert "surface/svi_repaired_grid" not in poster_datasets
    assert "surface/svi_repaired_grid" in static_datasets

    assert (
        manifest.resolve_dataset("essvi_smoothed_grid").source_object_type
        == "ESSVISmoothedSurface"
    )
    assert (
        manifest.resolve_dataset("svi_repaired_grid").source_object_type == "VolSurface"
    )


def test_vts_export_uses_manifest_dataset_metadata(
    visual_bundle_ci,
    tmp_path: Path,
) -> None:
    result = visual_bundle_ci

    essvi_path = export_dataset_vts(
        result.manifest,
        dataset_name="essvi_smoothed_grid",
        out_dir=tmp_path / "vts",
    )
    localvol_path = export_dataset_vts(
        result.manifest,
        dataset_name="localvol/gatheral_grid",
        out_dir=tmp_path / "vts",
    )

    essvi_text = essvi_path.read_text(encoding="utf-8")
    localvol_text = localvol_path.read_text(encoding="utf-8")

    assert 'Name="iv"' in essvi_text
    assert 'Name="w"' in essvi_text
    assert 'Name="sigma_loc"' in localvol_text
    assert essvi_path.exists()
    assert localvol_path.exists()
