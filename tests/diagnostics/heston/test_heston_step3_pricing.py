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


def test_run_heston_pricing_diagnostics_preserves_report_shape_and_slice_contract() -> (
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

    slice_arrays = report.arrays["slice"]
    assert {
        "smoothness_signal",
        "discontinuity_signal",
        "config_price_span",
        "perturbation_max_relative_price_change",
        "suspicious_flag",
    } <= set(slice_arrays)
    assert "parameter_perturbation_table" in report.arrays


def test_run_heston_pricing_diagnostics_builds_readable_backend_compare_and_summary() -> (
    None
):
    params = _params()
    report = run_heston_pricing_diagnostics(
        strike=np.array([90.0, 100.0, 110.0, 120.0], dtype=np.float64),
        tau=0.5,
        market=_market(),
        params=params,
        kind=OptionType.PUT,
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

    backend_compare = report.tables["backend_compare"]
    assert {
        "backend_a",
        "backend_b",
        "price_a",
        "price_b",
        "price_diff",
        "abs_price_diff",
    } <= set(backend_compare.columns)

    summary = report.tables["summary"]
    suspicious_row = summary.loc[summary["metric"] == "suspicious_strike_count"].iloc[0]
    assert "approval required" in str(suspicious_row["notes"]).lower()

    worst_strikes = report.tables["worst_strikes"]
    assert {"rank", "suspicious_flag", "suspicious_reasons"} <= set(
        worst_strikes.columns
    )

    config_sweep = report.tables["config_sweep"]
    assert {
        "config_label",
        "config_resolution",
        "resolved_u_max",
        "resolved_n_panels",
        "resolved_nodes_per_panel",
        "resolved_panel_spacing",
        "resolved_cluster_strength",
        "max_abs_price_diff_vs_baseline",
        "max_abs_probability_diff_p0",
        "max_abs_probability_diff_p1",
    } <= set(config_sweep.columns)


def test_run_heston_pricing_diagnostics_quad_primary_handles_missing_panel_detail() -> (
    None
):
    params = _params()
    robust_cfg = QuadratureConfig(u_max=150.0, n_panels=16, nodes_per_panel=12)
    report = run_heston_pricing_diagnostics(
        strike=np.array([90.0, 100.0, 110.0], dtype=np.float64),
        tau=0.5,
        market=_market(),
        params=params,
        kind=OptionType.CALL,
        backend="quad",
        comparison_backend="gauss_legendre",
        comparison_quad_cfg=robust_cfg,
        config_sweep_cases=[
            {"label": "primary", "backend": "quad"},
            {
                "label": "comparison",
                "backend": "gauss_legendre",
                "quad_cfg": robust_cfg,
            },
        ],
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )

    worst_panels_p0 = report.tables["worst_panels_p0"]
    assert not bool(worst_panels_p0["detail_available"].any())
    assert worst_panels_p0["notes"].str.contains("unavailable", case=False).all()

    slice_table = report.tables["slice"]
    assert list(slice_table.columns[: len(HESTON_SLICE_TABLE_COLUMNS)]) == list(
        HESTON_SLICE_TABLE_COLUMNS
    )
    assert set(report.tables["backend_compare"]["backend_a"]) == {"quad"}


def test_run_heston_pricing_diagnostics_flag_off_uses_default_hard_coded_gauss_for_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_recommendation(**_: object) -> QuadratureConfig:
        raise AssertionError("Flag-off path should not call the recommender.")

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
        comparison_backend="gauss_legendre",
        use_recommended_cfg=False,
        config_sweep_cases=[
            {"label": "primary", "backend": "gauss_legendre"},
            {"label": "comparison", "backend": "gauss_legendre"},
        ],
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )

    primary_cfg = report.meta["primary_backend_config"]
    comparison_cfg = report.meta["comparison_backend_config"]

    for cfg in (primary_cfg, comparison_cfg):
        assert cfg["backend"] == "gauss_legendre"
        assert cfg["config_resolution"] == "default_hard_coded"
        assert cfg["u_max"] == pytest.approx(150.0)
        assert cfg["n_panels"] == 24
        assert cfg["nodes_per_panel"] == 16
        assert cfg["panel_spacing"] == "uniform"
        assert cfg["cluster_strength"] == pytest.approx(2.0)

    config_sweep = report.tables["config_sweep"]
    assert set(config_sweep["config_resolution"]) == {"default_hard_coded"}
    assert set(config_sweep["resolved_n_panels"]) == {24}


def test_run_heston_pricing_diagnostics_flag_on_uses_recommended_balanced_and_robust_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    balanced_cfg = QuadratureConfig(
        u_max=180.0,
        n_panels=32,
        nodes_per_panel=20,
    )
    robust_cfg = QuadratureConfig(
        u_max=260.0,
        n_panels=44,
        nodes_per_panel=24,
    )
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
    assert primary_cfg["nodes_per_panel"] == balanced_cfg.nodes_per_panel
    assert comparison_cfg["config_resolution"] == "recommended_robust"
    assert comparison_cfg["u_max"] == pytest.approx(robust_cfg.u_max)
    assert comparison_cfg["n_panels"] == robust_cfg.n_panels
    assert comparison_cfg["nodes_per_panel"] == robust_cfg.nodes_per_panel

    config_sweep = report.tables["config_sweep"]
    assert {
        "primary",
        "comparison",
        "gauss_robust",
        "gauss_balanced",
    } == set(config_sweep["config_label"])

    rows = {
        str(row["config_label"]): row for row in config_sweep.to_dict(orient="records")
    }
    assert rows["primary"]["config_resolution"] == "recommended_balanced"
    assert rows["comparison"]["config_resolution"] == "recommended_robust"
    assert rows["gauss_balanced"]["config_resolution"] == "recommended_balanced"
    assert rows["gauss_robust"]["config_resolution"] == "recommended_robust"
    assert rows["primary"]["resolved_n_panels"] == balanced_cfg.n_panels
    assert rows["gauss_robust"]["resolved_n_panels"] == robust_cfg.n_panels


def test_run_heston_pricing_diagnostics_explicit_quad_cfg_wins_over_recommendation_flag(
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


def test_run_heston_pricing_diagnostics_explicit_rule_wins_over_recommendation_flag(
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
            {
                "label": "primary",
                "backend": "gauss_legendre",
                "rule": _rule(),
            },
            {"label": "comparison", "backend": "quad"},
        ],
        parameter_perturbations=_minimal_parameter_perturbations(params),
    )

    primary_cfg = report.meta["primary_backend_config"]
    assert primary_cfg["config_resolution"] == "explicit_rule"
    assert primary_cfg["u_max"] is None
    assert primary_cfg["n_panels"] is None
    assert primary_cfg["nodes_per_panel"] is None
    assert primary_cfg["panel_spacing"] is None
    assert primary_cfg["cluster_strength"] is None

    primary_row = report.tables["config_sweep"].set_index("config_label").loc["primary"]
    assert primary_row["config_resolution"] == "explicit_rule"
    assert primary_row["resolved_u_max"] is None
    assert primary_row["resolved_n_panels"] is None
    assert primary_row["resolved_nodes_per_panel"] is None
    assert primary_row["resolved_panel_spacing"] is None


@pytest.mark.parametrize(
    ("backend_kwargs",),
    [
        ({"quad_cfg": _gauss_cfg()},),
        ({"rule": _rule()},),
    ],
)
def test_run_heston_pricing_diagnostics_quad_backend_still_rejects_quad_cfg_and_rule(
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
