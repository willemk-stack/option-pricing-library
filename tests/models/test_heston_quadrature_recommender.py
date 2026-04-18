from __future__ import annotations

from option_pricing.models.heston import (
    HestonParams,
    recommend_heston_quadrature_config,
)
from option_pricing.numerics.quadrature import PanelSpacing, QuadratureConfig


def _ordinary_params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def _near_deterministic_params() -> HestonParams:
    return HestonParams(kappa=100.0, vbar=0.04, eta=1e-6, rho=0.0, v=0.04)


def test_recommend_heston_quadrature_config_returns_valid_cfg() -> None:
    cfg = recommend_heston_quadrature_config(
        x=-0.2,
        tau=1.0,
        params=_ordinary_params(),
        quality="balanced",
    )

    assert isinstance(cfg, QuadratureConfig)
    assert cfg.validate() is None
    assert cfg.u_max > 0.0
    assert cfg.n_panels >= 1
    assert cfg.nodes_per_panel >= 1
    assert cfg.panel_spacing in (PanelSpacing.UNIFORM, PanelSpacing.CLUSTERED)


def test_recommend_heston_quadrature_config_preserves_balanced_baseline_in_ordinary_regime() -> (
    None
):
    cfg = recommend_heston_quadrature_config(
        x=-0.2,
        tau=1.0,
        params=_ordinary_params(),
        quality="balanced",
    )

    assert cfg == QuadratureConfig(
        u_max=150.0,
        n_panels=24,
        nodes_per_panel=16,
        panel_spacing=PanelSpacing.UNIFORM,
        cluster_strength=2.0,
    )


def test_recommend_heston_quadrature_config_increases_u_max_for_short_maturity() -> (
    None
):
    ordinary = _ordinary_params()
    balanced = recommend_heston_quadrature_config(
        x=-0.2,
        tau=1.0,
        params=ordinary,
        quality="balanced",
    )
    short_tau = recommend_heston_quadrature_config(
        x=-0.2,
        tau=0.04,
        params=ordinary,
        quality="balanced",
    )

    assert short_tau.u_max > balanced.u_max
    assert short_tau.n_panels >= balanced.n_panels
    assert short_tau.nodes_per_panel >= balanced.nodes_per_panel


def test_recommend_heston_quadrature_config_increases_panel_count_for_large_abs_x() -> (
    None
):
    ordinary = _ordinary_params()
    balanced = recommend_heston_quadrature_config(
        x=-0.2,
        tau=1.0,
        params=ordinary,
        quality="balanced",
    )
    large_abs_x = recommend_heston_quadrature_config(
        x=1.2,
        tau=1.0,
        params=ordinary,
        quality="balanced",
    )

    assert large_abs_x.n_panels > balanced.n_panels
    assert large_abs_x.u_max >= balanced.u_max
    assert large_abs_x.nodes_per_panel >= balanced.nodes_per_panel


def test_recommend_heston_quadrature_config_clusters_panels_in_near_deterministic_regime() -> (
    None
):
    cfg = recommend_heston_quadrature_config(
        x=-0.2,
        tau=1.0,
        params=_near_deterministic_params(),
        quality="balanced",
    )

    assert cfg.panel_spacing == PanelSpacing.CLUSTERED
    assert cfg.cluster_strength > 0.0
    assert cfg.n_panels >= 40
    assert cfg.nodes_per_panel >= 24


def test_recommend_heston_quadrature_config_robust_and_diagnostics_are_monotone_vs_balanced() -> (
    None
):
    params = _ordinary_params()

    balanced = recommend_heston_quadrature_config(
        x=-0.2,
        tau=1.0,
        params=params,
        quality="balanced",
    )
    robust = recommend_heston_quadrature_config(
        x=-0.2,
        tau=1.0,
        params=params,
        quality="robust",
    )
    diagnostics = recommend_heston_quadrature_config(
        x=-0.2,
        tau=1.0,
        params=params,
        quality="diagnostics",
    )

    assert robust.u_max >= balanced.u_max
    assert robust.n_panels >= balanced.n_panels
    assert robust.nodes_per_panel >= balanced.nodes_per_panel

    assert diagnostics.u_max >= robust.u_max
    assert diagnostics.n_panels >= robust.n_panels
    assert diagnostics.nodes_per_panel >= robust.nodes_per_panel
