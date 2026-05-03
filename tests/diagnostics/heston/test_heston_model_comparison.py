from __future__ import annotations

import numpy as np

from option_pricing.diagnostics.heston import (
    HestonModelComparisonDiagnostics,
    build_synthetic_heston_quote_set,
    run_heston_vs_local_vol_comparison,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.vol.ssvi import ESSVICalibrationConfig


def _true_params() -> HestonParams:
    return HestonParams(kappa=1.5, vbar=0.04, eta=0.45, rho=-0.50, v=0.042)


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=45.0, n_panels=5, nodes_per_panel=5)


def test_heston_vs_local_vol_comparison_uses_essvi_proxy() -> None:
    quotes = build_synthetic_heston_quote_set(
        market=None,
        true_params=_true_params(),
        expiries=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        log_moneyness=np.array([-0.08, 0.0, 0.08], dtype=np.float64),
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        random_seed=123,
        noise_vol_bps=0.0,
    )

    report = run_heston_vs_local_vol_comparison(
        quotes=quotes,
        heston_fit=_true_params(),
        heston_quad_cfg=_quad_cfg(),
        essvi_cfg=ESSVICalibrationConfig(max_nfev=300),
    )

    assert isinstance(report, HestonModelComparisonDiagnostics)
    assert set(report.tables) == {
        "fit_errors",
        "error_summary",
        "tradeoff_summary",
        "held_out_comparison",
    }

    fit_errors = report.tables["fit_errors"]
    assert set(fit_errors["model"]) == {"Heston", "ESSVI local-vol proxy"}
    assert set(fit_errors["moneyness_bucket"]) == {
        "atm",
        "downside_wing",
        "upside_wing",
    }

    summary = report.tables["error_summary"]
    assert set(summary["model"]) == {"Heston", "ESSVI local-vol proxy"}
    assert {"atm", "downside_wing", "upside_wing"} <= set(summary["bucket"])
    metric_columns = [
        "price_rmse",
        "price_mae",
        "price_max_abs",
        "iv_rmse_bps",
        "iv_mae_bps",
        "iv_max_abs_bps",
    ]
    for column in metric_columns:
        values = summary[column].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(values))

    notes = " ".join(str(note) for note in report.meta["notes"])
    assert "REVIEW:" in notes
    assert "direct Dupire/PDE repricing is not run" in notes


def test_heston_vs_local_vol_held_out_comparison_is_partitioned() -> None:
    quotes = build_synthetic_heston_quote_set(
        market=None,
        true_params=_true_params(),
        expiries=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        log_moneyness=np.array([-0.08, 0.0, 0.08], dtype=np.float64),
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        random_seed=321,
        noise_vol_bps=0.0,
    )
    held_out = np.zeros(quotes.n_quotes, dtype=np.bool_)
    held_out[2::3] = True

    report = run_heston_vs_local_vol_comparison(
        quotes=quotes,
        heston_fit=_true_params(),
        held_out_mask=held_out,
        heston_quad_cfg=_quad_cfg(),
        essvi_cfg=ESSVICalibrationConfig(max_nfev=300),
    )

    held_out_comparison = report.tables["held_out_comparison"]
    assert set(held_out_comparison["model"]) == {"Heston", "ESSVI local-vol proxy"}
    assert set(held_out_comparison["sample"]) == {"train", "held_out"}
    assert (
        int(
            held_out_comparison.loc[
                held_out_comparison["model"] == "Heston", "n_quotes"
            ].sum()
        )
        == quotes.n_quotes
    )
