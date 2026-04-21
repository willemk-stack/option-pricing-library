from __future__ import annotations

import numpy as np

from option_pricing.diagnostics.heston.integration import (
    integration_config_sweep,
    integration_diagnostics,
)
from option_pricing.diagnostics.heston.models import HestonIntegrationDiagnosticsBundle
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig


def _params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def _stressed_params() -> HestonParams:
    return HestonParams(kappa=0.4, vbar=0.04, eta=1.2, rho=-0.95, v=0.02)


def _gauss_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=120.0, n_panels=12, nodes_per_panel=12)


def test_integration_diagnostics_gauss_builds_readable_panel_warning_and_reason_tables() -> (
    None
):
    artifact = integration_diagnostics(
        x=np.array([-0.15, 0.0, 0.15], dtype=np.float64),
        tau=0.75,
        params=_params(),
        j=1,
        backend="gauss_legendre",
        quad_cfg=_gauss_cfg(),
        strike=np.array([115.0, 100.0, 87.0], dtype=np.float64),
    )

    assert isinstance(artifact, HestonIntegrationDiagnosticsBundle)
    assert set(artifact.tables) == {
        "panels",
        "warning_summary",
        "worst_panels",
        "reason_counts",
    }
    assert artifact.meta["backend"] == "gauss_legendre"
    assert artifact.meta["config_resolution"] == "explicit_quad_cfg"
    assert artifact.meta["probability_index"] == 1
    assert artifact.meta["point_count"] == 3
    assert artifact.meta["panel_detail_available"] is True
    assert artifact.meta["u_max"] == _gauss_cfg().u_max
    assert artifact.meta["n_panels"] == _gauss_cfg().n_panels
    assert artifact.meta["nodes_per_panel"] == _gauss_cfg().nodes_per_panel

    panels = artifact.tables["panels"]
    assert {
        "point_index",
        "panel_index",
        "panel_contribution",
        "reason_labels",
        "severity",
        "detail_available",
    } <= set(panels.columns)
    assert bool(panels["detail_available"].all())

    worst_panels = artifact.tables["worst_panels"]
    assert not worst_panels.empty
    assert worst_panels.columns[0] == "rank"


def test_integration_diagnostics_quad_marks_panel_detail_unavailable() -> None:
    artifact = integration_diagnostics(
        x=np.array([-0.25, 0.0, 0.25], dtype=np.float64),
        tau=0.5,
        params=_params(),
        j=0,
        backend="quad",
        strike=np.array([120.0, 100.0, 85.0], dtype=np.float64),
    )

    panels = artifact.tables["panels"]
    assert not bool(panels["detail_available"].any())
    assert panels["notes"].str.contains("unavailable", case=False).all()
    assert artifact.meta["config_resolution"] is None
    assert artifact.meta["u_max"] is None
    assert artifact.meta["n_panels"] is None

    reason_counts = artifact.tables["reason_counts"]
    assert reason_counts.iloc[0]["reason_name"] == "panel_detail_unavailable"
    assert reason_counts.iloc[0]["severity"] == "info"


def test_integration_diagnostics_decodes_warning_and_reason_counts_for_stressed_case() -> (
    None
):
    artifact = integration_diagnostics(
        x=0.0,
        tau=0.1,
        params=_stressed_params(),
        j=0,
        backend="gauss_legendre",
        quad_cfg=QuadratureConfig(u_max=10.0, n_panels=8, nodes_per_panel=8),
        strike=100.0,
    )

    warning_summary = artifact.tables["warning_summary"]
    assert "large_tail_fraction" in set(warning_summary["warning_name"])

    reason_counts = artifact.tables["reason_counts"]
    assert {"underresolved_tail", "tail_too_large"} & set(reason_counts["reason_name"])
    assert set(reason_counts["severity"]).issubset(
        {"ok", "info", "warning", "severe", "critical"}
    )


def test_integration_config_sweep_returns_summary_table_and_artifacts_by_label() -> (
    None
):
    sweep, artifacts = integration_config_sweep(
        x=np.array([-0.1, 0.0, 0.1], dtype=np.float64),
        tau=0.5,
        params=_params(),
        j=0,
        strike=np.array([110.0, 100.0, 90.0], dtype=np.float64),
        cases=[
            {
                "label": "gauss",
                "backend": "gauss_legendre",
                "quad_cfg": _gauss_cfg(),
            },
            {"label": "quad", "backend": "quad"},
        ],
    )

    assert {
        "config_label",
        "backend",
        "config_resolution",
        "resolved_u_max",
        "resolved_n_panels",
        "resolved_nodes_per_panel",
        "resolved_panel_spacing",
        "resolved_cluster_strength",
        "max_severity",
    } <= set(sweep.columns)
    assert set(artifacts) == {"gauss", "quad"}
    assert sweep.shape[0] == 2
    assert isinstance(artifacts["gauss"], HestonIntegrationDiagnosticsBundle)
    assert artifacts["gauss"].tables["worst_panels"].columns[0] == "rank"
    assert artifacts["gauss"].meta["config_resolution"] == "explicit_quad_cfg"
    assert artifacts["quad"].meta["config_resolution"] is None
