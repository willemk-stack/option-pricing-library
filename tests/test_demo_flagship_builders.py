from __future__ import annotations

import pytest

from option_pricing.demos import build_localvol_pde_demo_artifacts
from option_pricing.demos.integration import run_capstone2


def test_demo_builders_reuse_shared_scenario(
    quick_demo_scenario,
    quick_demo_surface_artifacts,
    quick_demo_bridge_artifacts,
) -> None:
    scenario = quick_demo_scenario
    surface = quick_demo_surface_artifacts
    bridge = quick_demo_bridge_artifacts

    assert surface.scenario is scenario
    assert bridge.scenario is scenario
    assert surface.tables["quotes_df"].equals(scenario.quotes_df)
    assert bridge.tables["quotes_df"].equals(scenario.quotes_df)


def test_localvol_pde_builder_uses_smoothed_essvi_surface(
    quick_demo_scenario,
    quick_demo_bridge_artifacts,
    quick_demo_workflow_overrides: dict[str, object],
) -> None:
    scenario = quick_demo_scenario
    bridge = quick_demo_bridge_artifacts
    artifacts = build_localvol_pde_demo_artifacts(
        profile="quick",
        seed=7,
        overrides=quick_demo_workflow_overrides,
        scenario=scenario,
        bridge_artifacts=bridge,
    )

    assert type(artifacts.bridge.smoothed_surface).__name__ == "ESSVISmoothedSurface"
    assert artifacts.localvol.implied is artifacts.bridge.smoothed_surface
    assert artifacts.meta["input_surface_type"] == "ESSVISmoothedSurface"
    assert "localvol_summary" in artifacts.tables


def test_run_capstone2_preserves_surface_meta_compatibility(
    quick_demo_workflow_overrides: dict[str, object],
) -> None:
    with pytest.deprecated_call(match="compatibility alias"):
        artifacts = run_capstone2(
            profile="quick",
            seed=7,
            overrides=quick_demo_workflow_overrides,
        )

    surface_meta = artifacts.meta["surface_demo"]

    assert artifacts.meta["focus_T"] == surface_meta["focus_T"]
    assert (
        artifacts.meta["interpolation_probe_T"] == surface_meta["interpolation_probe_T"]
    )
    assert (
        artifacts.meta["svi_repair_mode_used"] == surface_meta["svi_repair_mode_used"]
    )
    assert (
        artifacts.meta["svi_repair_fallback_used"]
        == surface_meta["svi_repair_fallback_used"]
    )
