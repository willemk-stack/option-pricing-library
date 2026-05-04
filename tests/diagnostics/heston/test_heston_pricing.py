from __future__ import annotations

import numpy as np
import pytest

import option_pricing.diagnostics.heston.pricing as heston_pricing
from option_pricing.diagnostics.heston import run_heston_pricing_diagnostics
from option_pricing.diagnostics.heston.contracts import (
    HESTON_REQUIRED_REPORT_TABLES,
    HESTON_SLICE_TABLE_COLUMNS,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import (
    QuadratureConfig,
    build_gauss_legendre_rule,
)
from option_pricing.types import MarketData, OptionType


def _params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def _market() -> MarketData:
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)


def _gauss_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=120.0, n_panels=12, nodes_per_panel=12)


def _rule():
    return build_gauss_legendre_rule(_gauss_cfg())


def _minimal_parameter_perturbations(params: HestonParams) -> list[dict[str, object]]:
    return [
        {
            "label": "eta_up",
            "parameter": "eta",
            "direction": "up",
            "bump": 0.01,
            "params": HestonParams(
                kappa=params.kappa,
                vbar=params.vbar,
                eta=params.eta + 0.01,
                rho=params.rho,
                v=params.v,
            ),
        },
        {
            "label": "rho_down",
            "parameter": "rho",
            "direction": "down",
            "bump": -0.01,
            "params": HestonParams(
                kappa=params.kappa,
                vbar=params.vbar,
                eta=params.eta,
                rho=params.rho - 0.01,
                v=params.v,
            ),
        },
    ]


def test_run_heston_pricing_diagnostics_returns_required_tables_and_slice_contract() -> (
    None
):
    params = _params()
    report = run_heston_pricing_diagnostics(
        strike=np.array([85.0, 95.0, 100.0, 105.0, 115.0], dtype=np.float64),
        tau=0.75,
        market=_market(),
        params=params,
        kind=OptionType.CALL,
        backend="gauss_legendre",
        quad_cfg=_gauss_cfg(),
        comparison_backend="quad",
        config_sweep_cases=[
            {
                "label": "primary",
                "backend": "gauss_legendre",
                "quad_cfg": _gauss_cfg(),
            },
            {"label": "comparison", "backend": "quad"},
        ],
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )

    assert set(report.tables) == set(HESTON_REQUIRED_REPORT_TABLES)
    assert list(
        report.tables["slice"].columns[: len(HESTON_SLICE_TABLE_COLUMNS)]
    ) == list(HESTON_SLICE_TABLE_COLUMNS)
    assert {"meta", "tables", "arrays"} == set(report.to_dict())

    assert {
        "primary_backend_config",
        "comparison_backend_config",
        "parameter_perturbation_backend_config",
        "acceptance_policy",
        "provisional_policy",
    } <= set(report.meta)
    assert {"slice", "parameter_perturbation_table"} <= set(report.arrays)
    assert {"smoothness_signal", "discontinuity_signal", "suspicious_flag"} <= set(
        report.arrays["slice"]
    )


def test_perturbation_instability_mask_requires_relative_and_absolute_thresholds() -> (
    None
):
    policy = heston_pricing._DEFAULT_ACCEPTANCE_POLICY

    mask = heston_pricing._perturbation_instability_mask(
        perturbation_abs_price_change=np.array(
            [2.5e-4, 7.5e-4, 7.5e-4],
            dtype=np.float64,
        ),
        perturbation_rel_price_change=np.array(
            [0.25, 0.05, 0.25],
            dtype=np.float64,
        ),
        policy=policy,
    )

    np.testing.assert_array_equal(mask, np.array([False, False, True], dtype=np.bool_))


def test_demo11_style_otm_flags_split_parameter_sensitivity_from_suspicious() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.80, rho=-0.75, v=0.05)
    strike = np.linspace(75.0, 125.0, 17, dtype=np.float64)

    report = run_heston_pricing_diagnostics(
        strike=strike,
        tau=0.15,
        market=market,
        params=params,
        use_recommended_cfg=True,
    )

    slice_table = report.tables["slice"]
    flagged = slice_table.loc[
        slice_table["parameter_sensitivity_flag"].fillna(False).astype(bool)
    ]

    assert set(flagged["strike"].to_numpy(dtype=np.float64)) == {115.625, 118.75}
    assert not slice_table["suspicious_flag"].fillna(False).astype(bool).any()


def test_recommended_cfg_perturbation_runs_reuse_resolved_primary_config() -> None:
    params = _params()
    strike = np.array([85.0, 95.0, 100.0, 105.0, 115.0], dtype=np.float64)
    tau = 0.75
    forward = _market().to_context().fwd(tau)
    x = np.log(forward / strike)
    balanced_cfg = heston_pricing.recommend_heston_quadrature_config(
        x=float(np.max(np.abs(x))),
        tau=tau,
        params=params,
        quality="balanced",
    )

    report_recommended = run_heston_pricing_diagnostics(
        strike=strike,
        tau=tau,
        market=_market(),
        params=params,
        backend="gauss_legendre",
        comparison_backend="quad",
        use_recommended_cfg=True,
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )
    report_explicit = run_heston_pricing_diagnostics(
        strike=strike,
        tau=tau,
        market=_market(),
        params=params,
        backend="gauss_legendre",
        quad_cfg=balanced_cfg,
        comparison_backend="quad",
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )

    assert (
        report_recommended.meta["parameter_perturbation_backend_config"][
            "config_resolution"
        ]
        == "recommended_balanced"
    )

    perturbation_table = report_recommended.arrays["parameter_perturbation_table"]
    assert {
        "backend",
        "config_resolution",
        "resolved_u_max",
        "resolved_n_panels",
        "resolved_nodes_per_panel",
        "resolved_panel_spacing",
        "resolved_cluster_strength",
    } <= set(perturbation_table)
    assert set(perturbation_table["config_resolution"]) == {"recommended_balanced"}
    assert all(
        value == pytest.approx(balanced_cfg.u_max)
        for value in perturbation_table["resolved_u_max"]
    )
    assert set(perturbation_table["resolved_n_panels"]) == {balanced_cfg.n_panels}
    assert set(perturbation_table["resolved_nodes_per_panel"]) == {
        balanced_cfg.nodes_per_panel
    }

    for label, recommended_prices in report_recommended.arrays[
        "parameter_perturbation_prices"
    ].items():
        np.testing.assert_allclose(
            np.asarray(recommended_prices, dtype=np.float64),
            np.asarray(
                report_explicit.arrays["parameter_perturbation_prices"][label],
                dtype=np.float64,
            ),
            atol=1.0e-12,
            rtol=0.0,
        )


def test_run_heston_pricing_diagnostics_uses_recommended_configs_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    balanced_cfg = QuadratureConfig(u_max=180.0, n_panels=32, nodes_per_panel=20)
    robust_cfg = QuadratureConfig(u_max=260.0, n_panels=44, nodes_per_panel=24)
    calls: list[str] = []

    def _fake_recommendation(
        *,
        x: float,
        tau: float,
        params: HestonParams,
        quality: str,
    ) -> QuadratureConfig:
        assert x >= 0.0
        assert tau == pytest.approx(0.5)
        assert params == _params()
        calls.append(quality)
        if quality == "balanced":
            return balanced_cfg
        if quality == "robust":
            return robust_cfg
        raise AssertionError(f"Unexpected quality: {quality}")

    monkeypatch.setattr(
        heston_pricing,
        "recommend_heston_quadrature_config",
        _fake_recommendation,
    )

    params = _params()
    report = run_heston_pricing_diagnostics(
        strike=np.array([90.0, 100.0, 110.0], dtype=np.float64),
        tau=0.5,
        market=_market(),
        params=params,
        backend="gauss_legendre",
        comparison_backend="gauss_legendre",
        use_recommended_cfg=True,
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )

    assert calls == ["balanced", "robust", "balanced", "robust"]

    primary_cfg = report.meta["primary_backend_config"]
    comparison_cfg = report.meta["comparison_backend_config"]
    assert primary_cfg["config_resolution"] == "recommended_balanced"
    assert primary_cfg["u_max"] == pytest.approx(balanced_cfg.u_max)
    assert primary_cfg["n_panels"] == balanced_cfg.n_panels
    assert comparison_cfg["config_resolution"] == "recommended_robust"
    assert comparison_cfg["u_max"] == pytest.approx(robust_cfg.u_max)
    assert comparison_cfg["n_panels"] == robust_cfg.n_panels


def test_run_heston_pricing_diagnostics_respects_explicit_quad_cfg_over_recommendation_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_recommendation(**_: object) -> QuadratureConfig:
        raise AssertionError("Explicit quad_cfg should bypass the recommender.")

    monkeypatch.setattr(
        heston_pricing,
        "recommend_heston_quadrature_config",
        _fail_recommendation,
    )

    params = _params()
    explicit_cfg = QuadratureConfig(u_max=190.0, n_panels=20, nodes_per_panel=12)
    report = run_heston_pricing_diagnostics(
        strike=np.array([90.0, 100.0, 110.0], dtype=np.float64),
        tau=0.5,
        market=_market(),
        params=params,
        backend="gauss_legendre",
        quad_cfg=explicit_cfg,
        comparison_backend="quad",
        use_recommended_cfg=True,
        config_sweep_cases=[
            {
                "label": "primary",
                "backend": "gauss_legendre",
                "quad_cfg": explicit_cfg,
            },
            {"label": "comparison", "backend": "quad"},
        ],
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )

    primary_cfg = report.meta["primary_backend_config"]
    assert primary_cfg["config_resolution"] == "explicit_quad_cfg"
    assert primary_cfg["u_max"] == pytest.approx(explicit_cfg.u_max)
    assert primary_cfg["n_panels"] == explicit_cfg.n_panels
    assert primary_cfg["nodes_per_panel"] == explicit_cfg.nodes_per_panel


def test_run_heston_pricing_diagnostics_respects_explicit_rule_over_recommendation_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_recommendation(**_: object) -> QuadratureConfig:
        raise AssertionError("Explicit rule should bypass the recommender.")

    monkeypatch.setattr(
        heston_pricing,
        "recommend_heston_quadrature_config",
        _fail_recommendation,
    )

    params = _params()
    report = run_heston_pricing_diagnostics(
        strike=np.array([90.0, 100.0, 110.0], dtype=np.float64),
        tau=0.5,
        market=_market(),
        params=params,
        backend="gauss_legendre",
        rule=_rule(),
        comparison_backend="quad",
        use_recommended_cfg=True,
        config_sweep_cases=[
            {"label": "primary", "backend": "gauss_legendre", "rule": _rule()},
            {"label": "comparison", "backend": "quad"},
        ],
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )

    primary_cfg = report.meta["primary_backend_config"]
    assert primary_cfg["config_resolution"] == "explicit_rule"
    assert primary_cfg["u_max"] is None
    assert primary_cfg["n_panels"] is None
    assert primary_cfg["nodes_per_panel"] is None


@pytest.mark.parametrize(
    ("backend_kwargs",),
    [
        ({"quad_cfg": _gauss_cfg()},),
        ({"rule": _rule()},),
    ],
)
def test_run_heston_pricing_diagnostics_quad_backend_rejects_quad_cfg_and_rule(
    backend_kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError, match="quad backend does not accept quad_cfg or rule"
    ):
        run_heston_pricing_diagnostics(
            strike=np.array([90.0, 100.0], dtype=np.float64),
            tau=0.5,
            market=_market(),
            params=_params(),
            backend="quad",
            parameter_perturbations=_minimal_parameter_perturbations(_params()),
            **backend_kwargs,
        )
