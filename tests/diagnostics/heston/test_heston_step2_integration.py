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


def _tail_warning_params() -> HestonParams:
    return HestonParams(kappa=0.4, vbar=0.04, eta=1.2, rho=-0.95, v=0.02)


def _gauss_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=120.0, n_panels=12, nodes_per_panel=12)


def test_integration_diagnostics_gauss_exposes_readable_panel_artifacts() -> None:
    x = np.array([-0.15, 0.0, 0.15], dtype=np.float64)
    strike = np.array([115.0, 100.0, 87.0], dtype=np.float64)

    artifact = integration_diagnostics(
        x=x,
        tau=0.75,
        params=_params(),
        j=1,
        backend="gauss_legendre",
        quad_cfg=_gauss_cfg(),
        strike=strike,
    )

    assert isinstance(artifact, HestonIntegrationDiagnosticsBundle)
    assert set(artifact.tables) == {
        "panels",
        "warning_summary",
        "worst_panels",
        "reason_counts",
    }
    assert {"severity", "warning_labels"} <= set(artifact.probability.table.columns)

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

    reason_counts = artifact.tables["reason_counts"]
    assert reason_counts.iloc[0]["reason_name"] == "panel_detail_unavailable"
    assert reason_counts.iloc[0]["severity"] == "info"


def test_integration_diagnostics_decodes_warnings_and_builds_config_sweep() -> None:
    stressed_cfg = QuadratureConfig(u_max=10.0, n_panels=8, nodes_per_panel=8)

    artifact = integration_diagnostics(
        x=0.0,
        tau=0.1,
        params=_tail_warning_params(),
        j=0,
        backend="gauss_legendre",
        quad_cfg=stressed_cfg,
        strike=100.0,
    )

    warning_summary = artifact.tables["warning_summary"]
    assert "large_tail_fraction" in set(warning_summary["warning_name"])

    reason_counts = artifact.tables["reason_counts"]
    assert {"underresolved_tail", "tail_too_large"} & set(reason_counts["reason_name"])
    assert set(reason_counts["severity"]).issubset(
        {"ok", "info", "warning", "severe", "critical"}
    )

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

    assert {"config_label", "backend", "max_severity"} <= set(sweep.columns)
    assert set(artifacts) == {"gauss", "quad"}
    assert sweep.shape[0] == 2
