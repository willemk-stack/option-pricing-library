from __future__ import annotations

import numpy as np

from option_pricing.diagnostics.heston import (
    HestonCalibrationFitDiagnostics,
    build_synthetic_heston_quote_set,
    run_heston_calibration_fit_diagnostics,
)
from option_pricing.models.heston.calibration.heston_types import (
    HestonCalibrationRun,
    HestonMultistartResult,
)
from option_pricing.models.heston.params import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig


def _true_params() -> HestonParams:
    return HestonParams(kappa=1.7, vbar=0.04, eta=0.55, rho=-0.55, v=0.045)


def _seed_params() -> HestonParams:
    return HestonParams(kappa=1.2, vbar=0.035, eta=0.40, rho=-0.35, v=0.04)


def _quad_cfg() -> QuadratureConfig:
    return QuadratureConfig(u_max=45.0, n_panels=5, nodes_per_panel=5)


def _quotes():
    return build_synthetic_heston_quote_set(
        market=None,
        true_params=_true_params(),
        expiries=np.array([0.5, 1.0], dtype=np.float64),
        log_moneyness=np.array([-0.08, 0.0, 0.08], dtype=np.float64),
        backend="gauss_legendre",
        quad_cfg=_quad_cfg(),
        random_seed=123,
        noise_vol_bps=0.0,
    )


def _multistart_result() -> HestonMultistartResult:
    seed = _seed_params()
    best = HestonCalibrationRun(
        seed_index=1,
        seed_params=seed,
        fitted_params=_true_params(),
        success=True,
        cost=0.0,
        optimality=1.0e-10,
        nfev=4,
        njev=3,
        status=1,
        message="synthetic success",
        raw_x=np.zeros(5, dtype=np.float64),
    )
    failed = HestonCalibrationRun(
        seed_index=0,
        seed_params=HestonParams(kappa=4.0, vbar=0.08, eta=1.2, rho=0.2, v=0.08),
        fitted_params=None,
        success=False,
        cost=np.inf,
        optimality=None,
        nfev=None,
        njev=None,
        status=None,
        message="NoConvergenceError: synthetic failure",
        raw_x=None,
    )
    return HestonMultistartResult(
        best_params=_true_params(),
        best_run=best,
        runs=(best, failed),
        objective_type="vega_scaled_price",
        parameter_transform="bounded",
        backend="gauss_legendre",
        quote_count=6,
        success_count=1,
        failure_count=1,
    )


def test_heston_calibration_fit_diagnostics_tables_and_exports() -> None:
    quotes = _quotes()
    held_out = np.zeros(quotes.n_quotes, dtype=np.bool_)
    held_out[2::3] = True

    report = run_heston_calibration_fit_diagnostics(
        quotes=quotes,
        fit=_multistart_result(),
        true_params=_true_params(),
        held_out_mask=held_out,
        quad_cfg=_quad_cfg(),
        objective_slice_grid_size=2,
    )

    assert isinstance(report, HestonCalibrationFitDiagnostics)
    assert set(report.tables) == {
        "residuals",
        "smile_fit",
        "iv_residual_grid",
        "parameter_recovery",
        "multistart_runs",
        "held_out_errors",
        "objective_slices",
    }

    residuals = report.tables["residuals"]
    assert len(residuals) == quotes.n_quotes
    assert np.all(np.isfinite(residuals["model_price"].to_numpy(dtype=np.float64)))
    assert np.all(np.isfinite(residuals["model_iv"].to_numpy(dtype=np.float64)))

    held_out_errors = report.tables["held_out_errors"]
    assert set(held_out_errors["sample"]) == {"train", "held_out"}
    assert int(held_out_errors["n_quotes"].sum()) == quotes.n_quotes

    parameter_recovery = report.tables["parameter_recovery"]
    assert {"parameter", "true", "seed", "fitted", "fit_minus_true"} <= set(
        parameter_recovery.columns
    )

    multistart = report.tables["multistart_runs"]
    assert set(multistart["success"]) == {True, False}
    assert bool(multistart.loc[multistart["best_run"], "success"].iloc[0])

    objective_slices = report.tables["objective_slices"]
    assert {"kappa_vs_vbar", "eta_vs_rho", "v_vs_vbar"} <= set(
        objective_slices["slice_name"]
    )
    assert np.all(np.isfinite(objective_slices["cost"].to_numpy(dtype=np.float64)))


def test_heston_calibration_fit_without_truth_or_heldout_is_explicit() -> None:
    quotes = _quotes()
    report = run_heston_calibration_fit_diagnostics(
        quotes=quotes,
        fit=_true_params(),
        quad_cfg=_quad_cfg(),
        objective_type="vega_scaled_price",
        objective_slice_grid_size=2,
    )

    recovery = report.tables["parameter_recovery"]
    assert {"parameter", "fitted"} <= set(recovery.columns)
    assert "true" not in recovery.columns
    assert report.tables["held_out_errors"].empty
    assert not bool(report.meta["held_out_mask_provided"])
