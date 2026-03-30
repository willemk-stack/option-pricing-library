from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
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
from option_pricing.viz.publishing import (
    PUBLISHING_THEMES,
    SvgTextStyle,
    estimate_svg_text_width,
    layout_svg_contained_raster,
    render_svg_contained_raster,
    wrap_svg_text,
)
from scripts.build_benchmark_artifacts import ROOT as REPO_ROOT
from scripts.build_benchmark_artifacts import (
    build_benchmark_overview_asset,
    build_macro_pipeline_summary_asset,
)

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _required_columns(df: pd.DataFrame, columns: set[str]) -> None:
    assert columns.issubset(df.columns), f"Missing columns: {columns - set(df.columns)}"


def _svg_text_lines(node: ET.Element) -> list[str]:
    tspans = node.findall("svg:tspan", SVG_NS)
    if tspans:
        return [
            "".join(tspan.itertext()).strip()
            for tspan in tspans
            if "".join(tspan.itertext()).strip()
        ]

    text = "".join(node.itertext()).strip()
    return [text] if text else []


def _assert_svg_text_nodes_fit(root: ET.Element) -> None:
    text_nodes = root.findall(".//svg:text[@data-max-width]", SVG_NS)
    assert text_nodes, "expected SVG text nodes with explicit width budgets"

    for node in text_nodes:
        node_id = node.attrib.get("id", "<missing-id>")
        max_width = float(node.attrib["data-max-width"])
        max_height = float(node.attrib["data-max-height"])
        line_height = float(node.attrib["data-line-height"])
        font_size = float(node.attrib["font-size"])
        font_weight = node.attrib.get("font-weight", "400")
        letter_spacing = float(node.attrib.get("letter-spacing", "0"))
        lines = _svg_text_lines(node)

        assert lines, f"{node_id} did not render any text lines"
        assert (
            len(lines) * line_height <= max_height + 1e-6
        ), f"{node_id} exceeded vertical budget"
        for line in lines:
            assert (
                estimate_svg_text_width(
                    line,
                    font_size=font_size,
                    font_weight=font_weight,
                    letter_spacing=letter_spacing,
                )
                <= max_width + 1e-6
            ), f"{node_id} overflowed width budget"


def _assert_svg_raster_nodes_contained(
    root: ET.Element,
    *,
    expected_count: int | None = None,
) -> None:
    image_nodes = root.findall(".//svg:image[@data-fit='contain']", SVG_NS)
    assert image_nodes, "expected SVG raster nodes with explicit contain metadata"
    if expected_count is not None:
        assert len(image_nodes) == expected_count

    for node in image_nodes:
        node_id = node.attrib.get("id", "<missing-id>")
        slot_x = float(node.attrib["data-slot-x"])
        slot_y = float(node.attrib["data-slot-y"])
        slot_width = float(node.attrib["data-slot-width"])
        slot_height = float(node.attrib["data-slot-height"])
        image_x = float(node.attrib["data-image-x"])
        image_y = float(node.attrib["data-image-y"])
        image_width = float(node.attrib["data-image-width"])
        image_height = float(node.attrib["data-image-height"])
        source_width = float(node.attrib["data-source-width"])
        source_height = float(node.attrib["data-source-height"])

        assert node.attrib.get("preserveAspectRatio") == "none"
        assert image_x >= slot_x - 1e-6, f"{node_id} exceeded slot left edge"
        assert image_y >= slot_y - 1e-6, f"{node_id} exceeded slot top edge"
        assert (
            image_x + image_width <= slot_x + slot_width + 1e-6
        ), f"{node_id} exceeded slot right edge"
        assert (
            image_y + image_height <= slot_y + slot_height + 1e-6
        ), f"{node_id} exceeded slot bottom edge"
        assert (
            abs((image_width / image_height) - (source_width / source_height)) <= 1e-5
        )

    for node in root.findall(".//svg:image", SVG_NS):
        assert "slice" not in node.attrib.get("preserveAspectRatio", "")


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
    presets = ["static", "dupire", "poster", "docs", "numerics"]
    expected_count = sum(len(build_plot_specs(preset=preset)) for preset in presets) * (
        len(PUBLISHING_THEMES) + 1
    )

    written = render_plot_presets(
        result.manifest,
        presets=presets,
        out_root=tmp_path / "assets",
    )

    assert written
    assert len(written) == expected_count
    assert all(path.exists() for path in written)


def test_wrap_svg_text_splits_long_copy_within_width_budget() -> None:
    style = SvgTextStyle(font_size=18, line_height=22)
    layout = wrap_svg_text(
        "README, docs visuals, and proof pages are regenerated and checked in CI.",
        max_width=240,
        max_height=88,
        style=style,
    )

    assert layout.fits
    assert len(layout.lines) >= 2
    assert all(
        estimate_svg_text_width(
            line,
            font_size=style.font_size,
            font_weight=style.font_weight,
            letter_spacing=style.letter_spacing,
        )
        <= 240 + 1e-6
        for line in layout.lines
    )
    assert layout.to_svg(x=96, y=128).count("<tspan") == len(layout.lines)


def test_wrap_svg_text_reports_vertical_overflow() -> None:
    style = SvgTextStyle(font_size=18, line_height=22)
    layout = wrap_svg_text(
        "README, docs visuals, and proof pages are regenerated and checked in CI.",
        max_width=240,
        max_height=22,
        style=style,
    )

    assert layout.fits_width
    assert not layout.fits_height
    assert not layout.fits


def test_layout_svg_contained_raster_preserves_aspect_ratio() -> None:
    layout = layout_svg_contained_raster(
        source_width=1890,
        source_height=756,
        slot_x=74,
        slot_y=264,
        slot_width=452,
        slot_height=290,
    )

    assert layout.image_width == pytest.approx(452.0)
    assert layout.image_height == pytest.approx(180.8)
    assert layout.image_x == pytest.approx(74.0)
    assert layout.image_y == pytest.approx(318.6)


def test_render_svg_contained_raster_emits_containment_metadata() -> None:
    asset_path = (
        REPO_ROOT
        / "docs"
        / "assets"
        / "generated"
        / "benchmarks"
        / "iv_scaling.light.png"
    )
    block = render_svg_contained_raster(
        block_id="test-contained-raster",
        image_path=asset_path,
        slot_x=0,
        slot_y=0,
        slot_width=400,
        slot_height=240,
        frame_radius=20,
        frame_fill="#FFFFFF",
        source_label="docs/assets/generated/benchmarks/iv_scaling.light.png",
    )

    assert 'data-fit="contain"' in block.svg
    assert (
        'data-source-path="docs/assets/generated/benchmarks/iv_scaling.light.png"'
        in block.svg
    )
    assert 'preserveAspectRatio="none"' in block.svg
    assert block.layout.image_width == pytest.approx(400.0)
    assert block.layout.image_height == pytest.approx(160.0)


def test_showcase_readme_proof_card_uses_wrapped_svg_text(
    visual_bundle_ci,
    tmp_path: Path,
) -> None:
    result = visual_bundle_ci
    out_root = tmp_path / "assets"

    written = render_plot_presets(
        result.manifest,
        presets=["showcase"],
        out_root=out_root,
    )

    assert any(path.name == "readme_proof_card.light.svg" for path in written)
    for theme in PUBLISHING_THEMES:
        path = out_root / "showcase" / f"readme_proof_card.{theme}.svg"
        root = ET.fromstring(path.read_text(encoding="utf-8"))

        assert (
            root.find(
                ".//svg:clipPath[@id='readmeCardClip-benchmarks-and-delivery']",
                SVG_NS,
            )
            is not None
        )
        body = root.find(
            ".//svg:text[@id='readme-card-benchmarks-and-delivery-body']",
            SVG_NS,
        )
        assert body is not None
        assert len(body.findall("svg:tspan", SVG_NS)) >= 3
        _assert_svg_text_nodes_fit(root)


def test_showcase_architecture_system_map_uses_wrapped_svg_text(
    visual_bundle_ci,
    tmp_path: Path,
) -> None:
    result = visual_bundle_ci
    out_root = tmp_path / "assets"

    render_plot_presets(
        result.manifest,
        presets=["showcase"],
        out_root=out_root,
    )

    for theme in PUBLISHING_THEMES:
        path = out_root / "showcase" / f"architecture_system_map.{theme}.svg"
        root = ET.fromstring(path.read_text(encoding="utf-8"))

        title = root.find(
            ".//svg:text[@id='architecture-system-map-title']",
            SVG_NS,
        )
        support = root.find(
            ".//svg:text[@id='architecture-system-map-support-numerics']",
            SVG_NS,
        )

        assert title is not None
        assert len(title.findall("svg:tspan", SVG_NS)) >= 2
        assert support is not None
        _assert_svg_text_nodes_fit(root)


def test_showcase_reviewer_proof_panel_uses_contained_raster_slots(
    visual_bundle_ci,
    tmp_path: Path,
) -> None:
    result = visual_bundle_ci
    out_root = tmp_path / "assets"

    render_plot_presets(
        result.manifest,
        presets=["showcase"],
        out_root=out_root,
    )

    for theme in PUBLISHING_THEMES:
        path = out_root / "showcase" / f"reviewer_proof_panel.{theme}.svg"
        raw_svg = path.read_text(encoding="utf-8")
        root = ET.fromstring(raw_svg)

        assert "xMidYMid slice" not in raw_svg
        _assert_svg_raster_nodes_contained(root, expected_count=3)


def test_benchmark_overview_uses_wrapped_svg_text() -> None:
    source_plot_dir = REPO_ROOT / "docs" / "assets" / "generated" / "benchmarks"
    plot_dir = REPO_ROOT / "out" / "pytest_benchmark_overview_layout" / "benchmarks"
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        for stem in (
            "iv_scaling",
            "pde_runtime_error_tradeoff",
            "macro_pipeline_summary",
        ):
            for theme in PUBLISHING_THEMES:
                shutil.copyfile(
                    source_plot_dir / f"{stem}.{theme}.png",
                    plot_dir / f"{stem}.{theme}.png",
                )

        build_benchmark_overview_asset(
            artifacts_dir=REPO_ROOT / "benchmarks" / "artifacts",
            plot_dir=plot_dir,
        )

        for theme in PUBLISHING_THEMES:
            path = plot_dir / f"benchmark_overview.{theme}.svg"
            raw_svg = path.read_text(encoding="utf-8")
            root = ET.fromstring(raw_svg)

            assert (
                root.find(
                    ".//svg:clipPath[@id='benchmarkOverviewCallout-local-vol-extraction']",
                    SVG_NS,
                )
                is not None
            )
            metric = root.find(
                ".//svg:text[@id='benchmark-overview-local-vol-extraction-metric']",
                SVG_NS,
            )
            assert metric is not None
            assert len(metric.findall("svg:tspan", SVG_NS)) >= 2
            assert "xMidYMid slice" not in raw_svg
            _assert_svg_text_nodes_fit(root)
            _assert_svg_raster_nodes_contained(root, expected_count=3)
    finally:
        if plot_dir.exists():
            shutil.rmtree(plot_dir)


def test_macro_pipeline_summary_asset_renders_from_snapshot() -> None:
    plot_dir = REPO_ROOT / "out" / "pytest_macro_pipeline_summary" / "benchmarks"
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        build_macro_pipeline_summary_asset(
            artifacts_dir=REPO_ROOT / "benchmarks" / "artifacts",
            plot_dir=plot_dir,
        )

        for theme in PUBLISHING_THEMES:
            path = plot_dir / f"macro_pipeline_summary.{theme}.png"
            assert path.exists()
            assert path.stat().st_size > 0
    finally:
        if plot_dir.exists():
            shutil.rmtree(plot_dir)


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
