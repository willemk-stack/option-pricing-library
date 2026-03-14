from __future__ import annotations

from option_pricing.demos import (
    build_essvi_bridge_artifacts,
    build_localvol_pde_demo_artifacts,
    build_shared_demo_scenario,
    build_surface_demo_artifacts,
)


def test_demo_builders_reuse_shared_scenario() -> None:
    overrides = {
        "RUN_*": {
            "RUN_EXPLICIT_SVI_REPAIR_DEMO": False,
            "RUN_GJ_PAPER_SANITY_CHECK": False,
            "RUN_DIGITAL_PDE_BASELINE": False,
            "RUN_LOCALVOL_REPRICING": False,
            "RUN_LOCALVOL_CONVERGENCE_SWEEP": False,
            "RUN_DUPIRE_VS_GATHERAL_COMPARE": False,
        }
    }
    scenario = build_shared_demo_scenario(profile="quick", seed=7, overrides=overrides)
    surface = build_surface_demo_artifacts(
        profile="quick",
        seed=7,
        overrides=overrides,
        scenario=scenario,
    )
    bridge = build_essvi_bridge_artifacts(
        profile="quick",
        seed=7,
        overrides=overrides,
        scenario=scenario,
        surface_artifacts=surface,
    )

    assert surface.scenario is scenario
    assert bridge.scenario is scenario
    assert surface.tables["quotes_df"].equals(scenario.quotes_df)
    assert bridge.tables["quotes_df"].equals(scenario.quotes_df)


def test_localvol_pde_builder_uses_smoothed_essvi_surface() -> None:
    overrides = {
        "RUN_*": {
            "RUN_EXPLICIT_SVI_REPAIR_DEMO": False,
            "RUN_GJ_PAPER_SANITY_CHECK": False,
            "RUN_DIGITAL_PDE_BASELINE": False,
            "RUN_LOCALVOL_REPRICING": False,
            "RUN_LOCALVOL_CONVERGENCE_SWEEP": False,
            "RUN_DUPIRE_VS_GATHERAL_COMPARE": False,
        }
    }
    scenario = build_shared_demo_scenario(profile="quick", seed=7, overrides=overrides)
    bridge = build_essvi_bridge_artifacts(
        profile="quick",
        seed=7,
        overrides=overrides,
        scenario=scenario,
    )
    artifacts = build_localvol_pde_demo_artifacts(
        profile="quick",
        seed=7,
        overrides=overrides,
        scenario=scenario,
        bridge_artifacts=bridge,
    )

    assert type(artifacts.bridge.smoothed_surface).__name__ == "ESSVISmoothedSurface"
    assert artifacts.localvol.implied is artifacts.bridge.smoothed_surface
    assert artifacts.meta["input_surface_type"] == "ESSVISmoothedSurface"
    assert "localvol_summary" in artifacts.tables
