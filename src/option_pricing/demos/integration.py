from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .essvi_bridge import ESSVIBridgeArtifacts, build_essvi_bridge_artifacts
from .localvol_pde_flagship import (
    LocalVolPDEDemoArtifacts,
    build_localvol_pde_demo_artifacts,
)
from .scenario import SharedDemoScenario, build_shared_demo_scenario
from .surface_flagship import SurfaceDemoArtifacts, build_surface_demo_artifacts


@dataclass(frozen=True, slots=True)
class SurfaceToLocalVolPDEIntegrationArtifacts:
    profile: str
    cfg: dict[str, Any]
    flags: dict[str, bool]
    scenario: SharedDemoScenario
    surface_demo: SurfaceDemoArtifacts
    essvi_bridge: ESSVIBridgeArtifacts
    localvol_pde_demo: LocalVolPDEDemoArtifacts
    synthetic: dict[str, Any]
    surfaces: dict[str, Any]
    localvol: Any
    tables: dict[str, pd.DataFrame]
    reports: dict[str, Any]
    repricing: Any | None
    convergence: Any | None
    digital_baseline: Any | None
    gj51: Any | None
    meta: dict[str, Any]


def build_demo_run_flags(
    *,
    profile: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, bool]:
    flags = {
        "RUN_EXPLICIT_SVI_REPAIR_DEMO": True,
        "RUN_GJ_PAPER_SANITY_CHECK": True,
        "RUN_ESSVI_TIME_SMOOTHNESS_COMPARE": True,
        "RUN_DIGITAL_PDE_BASELINE": True,
        "RUN_LOCALVOL_REPRICING": True,
        "RUN_LOCALVOL_CONVERGENCE_SWEEP": True,
        "RUN_DUPIRE_VS_GATHERAL_COMPARE": profile == "full",
    }
    if overrides and isinstance(overrides.get("RUN_*"), dict):
        for key, value in overrides["RUN_*"].items():
            if key in flags:
                flags[key] = bool(value)
    return flags


def run_surface_to_localvol_pde_integration(
    *,
    profile: str = "quick",
    seed: int = 7,
    overrides: dict[str, Any] | None = None,
) -> SurfaceToLocalVolPDEIntegrationArtifacts:
    scenario = build_shared_demo_scenario(
        profile=profile,
        seed=seed,
        overrides=overrides,
    )
    surface_demo = build_surface_demo_artifacts(
        profile=scenario.profile,
        seed=scenario.seed,
        overrides=overrides,
        scenario=scenario,
    )
    essvi_bridge = build_essvi_bridge_artifacts(
        profile=scenario.profile,
        seed=scenario.seed,
        overrides=overrides,
        scenario=scenario,
        surface_artifacts=surface_demo,
    )
    localvol_demo = build_localvol_pde_demo_artifacts(
        profile=scenario.profile,
        seed=scenario.seed,
        overrides=overrides,
        scenario=scenario,
        bridge_artifacts=essvi_bridge,
    )

    flags = build_demo_run_flags(profile=scenario.profile, overrides=overrides)
    tables: dict[str, pd.DataFrame] = {
        "quotes_df": scenario.quotes_df,
        "quote_summary": scenario.quote_summary,
        "price_quotes_df": scenario.price_quotes_df,
        "synth_tuning_log": pd.DataFrame(scenario.latent.tuning_log),
        "latent_truth_noarb": pd.DataFrame(
            [
                {
                    "latent_truth_ok": bool(scenario.noarb_true.ok),
                    "message": str(scenario.noarb_true.message),
                    "calendar_ok": (
                        bool(scenario.noarb_true.calendar_total_variance.ok)
                        if scenario.noarb_true.calendar_total_variance.performed
                        else None
                    ),
                    "calendar_max_violation": (
                        float(scenario.noarb_true.calendar_total_variance.max_violation)
                        if scenario.noarb_true.calendar_total_variance.performed
                        else None
                    ),
                }
            ]
        ),
        "synth_cfg_used": pd.DataFrame([scenario.latent.cfg_used]),
    }
    tables.update(surface_demo.tables)
    tables.update(essvi_bridge.tables)
    tables.update(localvol_demo.tables)

    reports: dict[str, Any] = {
        "noarb_true": scenario.noarb_true,
        **surface_demo.reports,
        **essvi_bridge.reports,
        **localvol_demo.reports,
    }
    surfaces = {
        **surface_demo.surfaces,
        "essvi_nodal": essvi_bridge.nodal_surface,
        "essvi_smoothed": essvi_bridge.smoothed_surface,
    }
    meta = {
        "profile": scenario.profile,
        "surface_demo": dict(surface_demo.meta),
        "essvi_bridge": dict(essvi_bridge.meta),
        "localvol_pde_demo": dict(localvol_demo.meta),
    }

    return SurfaceToLocalVolPDEIntegrationArtifacts(
        profile=scenario.profile,
        cfg=scenario.cfg,
        flags=flags,
        scenario=scenario,
        surface_demo=surface_demo,
        essvi_bridge=essvi_bridge,
        localvol_pde_demo=localvol_demo,
        synthetic=scenario.synthetic_bundle,
        surfaces=surfaces,
        localvol=localvol_demo.localvol,
        tables=tables,
        reports=reports,
        repricing=localvol_demo.repricing,
        convergence=localvol_demo.convergence,
        digital_baseline=localvol_demo.digital_baseline,
        gj51=surface_demo.reports.get("gj51"),
        meta=meta,
    )


Capstone2Artifacts = SurfaceToLocalVolPDEIntegrationArtifacts


def run_capstone2(
    profile: str = "quick",
    seed: int = 7,
    overrides: dict | None = None,
) -> SurfaceToLocalVolPDEIntegrationArtifacts:
    warnings.warn(
        "run_capstone2 is a compatibility alias for "
        "run_surface_to_localvol_pde_integration.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return run_surface_to_localvol_pde_integration(
        profile=profile,
        seed=seed,
        overrides=overrides,
    )


__all__ = [
    "Capstone2Artifacts",
    "SurfaceToLocalVolPDEIntegrationArtifacts",
    "build_demo_run_flags",
    "run_capstone2",
    "run_surface_to_localvol_pde_integration",
]
