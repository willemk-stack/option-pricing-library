from __future__ import annotations

import numpy as np

from option_pricing.diagnostics.heston import run_heston_pricing_diagnostics
from option_pricing.diagnostics.heston.contracts import (
    HESTON_REQUIRED_REPORT_TABLES,
    HESTON_SLICE_TABLE_COLUMNS,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.types import MarketData, OptionType


def _params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def _market() -> MarketData:
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)


def _gauss_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=120.0, n_panels=12, nodes_per_panel=12)


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
