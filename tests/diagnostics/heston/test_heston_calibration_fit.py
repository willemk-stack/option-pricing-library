from __future__ import annotations

import json

import numpy as np
import pandas as pd

from option_pricing.diagnostics.heston import (
    HestonCalibrationFitDiagnostics,
    build_synthetic_heston_quote_set,
    heston_calibration_fit_summary,
    run_heston_calibration_fit_diagnostics,
    summarize_heston_calibration_fit,
)
from option_pricing.diagnostics.heston.contracts import (
    HESTON_CALIBRATION_FIT_IV_GRID_COLUMNS,
    HESTON_CALIBRATION_FIT_RESIDUAL_COLUMNS,
    HESTON_CALIBRATION_FIT_SMILE_COLUMNS,
)
from option_pricing.models.heston.calibration.bounds import HestonCalibrationBounds
from option_pricing.models.heston.calibration.heston_types import (
    HestonCalibrationRun,
    HestonMultistartResult,
)
from option_pricing.models.heston.fourier import HestonIntegralWarning
from option_pricing.models.heston.params import HESTON_PARAM_NAMES, HestonParams
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


def _boundary_multistart_result() -> HestonMultistartResult:
    best_params = HestonParams(kappa=19.99, vbar=0.04, eta=0.55, rho=-0.55, v=0.045)
    other_params = HestonParams(
        kappa=19.995,
        vbar=0.041,
        eta=0.57,
        rho=-0.57,
        v=0.044,
    )
    seed = _seed_params()
    best = HestonCalibrationRun(
        seed_index=0,
        seed_params=seed,
        fitted_params=best_params,
        success=True,
        cost=0.1,
        optimality=1.0e-8,
        nfev=6,
        njev=5,
        status=1,
        message="upper boundary solution",
        raw_x=np.zeros(5, dtype=np.float64),
    )
    other = HestonCalibrationRun(
        seed_index=1,
        seed_params=HestonParams(kappa=4.0, vbar=0.08, eta=1.2, rho=0.2, v=0.08),
        fitted_params=other_params,
        success=True,
        cost=0.2,
        optimality=2.0e-8,
        nfev=7,
        njev=6,
        status=1,
        message="also near upper boundary",
        raw_x=np.ones(5, dtype=np.float64),
    )
    return HestonMultistartResult(
        best_params=best_params,
        best_run=best,
        runs=(best, other),
        objective_type="vega_scaled_price",
        parameter_transform="bounded",
        backend="gauss_legendre",
        quote_count=6,
        success_count=2,
        failure_count=0,
    )


def _feller_ratio_for_params(params: HestonParams) -> float:
    if float(params.eta) == 0.0:
        return np.inf
    return float(2.0 * params.kappa * params.vbar / (params.eta * params.eta))


def _synthetic_parameter_boundaries(params: HestonParams) -> pd.DataFrame:
    bounds = HestonCalibrationBounds()
    values = params.as_array()
    lower = bounds.lower_array()
    upper = bounds.upper_array()
    fraction = (values - lower) / (upper - lower)
    tolerance = 1.0e-3
    rows = []
    for idx, parameter in enumerate(HESTON_PARAM_NAMES):
        hit_lower = bool(fraction[idx] <= tolerance)
        hit_upper = bool((1.0 - fraction[idx]) <= tolerance)
        rows.append(
            {
                "parameter": parameter,
                "value": float(values[idx]),
                "lower_bound": float(lower[idx]),
                "upper_bound": float(upper[idx]),
                "normalized_position": float(fraction[idx]),
                "distance_to_lower": float(values[idx] - lower[idx]),
                "distance_to_upper": float(upper[idx] - values[idx]),
                "normalized_distance_to_lower": float(fraction[idx]),
                "normalized_distance_to_upper": float(1.0 - fraction[idx]),
                "hit_lower": hit_lower,
                "hit_upper": hit_upper,
                "hit_boundary": bool(hit_lower or hit_upper),
                "tolerance": tolerance,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_constraint_diagnostics(params: HestonParams) -> pd.DataFrame:
    ratio = _feller_ratio_for_params(params)
    return pd.DataFrame(
        [
            {
                "constraint": "feller",
                "value": ratio,
                "margin": float(
                    2.0 * params.kappa * params.vbar - params.eta * params.eta
                ),
                "satisfied": bool(ratio >= 1.0),
                "policy": "reported_not_hard_enforced",
                "notes": "synthetic summary-only fixture",
            }
        ]
    )


def _synthetic_held_out_errors(*, include: bool = True) -> pd.DataFrame:
    columns = [
        "sample",
        "n_quotes",
        "price_rmse",
        "price_mae",
        "price_max_abs",
        "iv_rmse_bps",
        "iv_mae_bps",
        "iv_max_abs_bps",
    ]
    if not include:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [
            {
                "sample": "train",
                "n_quotes": 48,
                "price_rmse": 0.02,
                "price_mae": 0.015,
                "price_max_abs": 0.05,
                "iv_rmse_bps": 4.0,
                "iv_mae_bps": 3.0,
                "iv_max_abs_bps": 8.0,
            },
            {
                "sample": "held_out",
                "n_quotes": 12,
                "price_rmse": 0.025,
                "price_mae": 0.018,
                "price_max_abs": 0.06,
                "iv_rmse_bps": 5.0,
                "iv_mae_bps": 4.0,
                "iv_max_abs_bps": 9.0,
            },
        ],
        columns=columns,
    )


def _synthetic_multistart_runs(
    *,
    success_count: int = 3,
    failure_count: int = 0,
) -> pd.DataFrame:
    rows = [{"success": True} for _ in range(success_count)]
    rows.extend({"success": False} for _ in range(failure_count))
    return pd.DataFrame(rows, columns=["success"])


def _synthetic_multistart_dispersion(
    params: HestonParams,
    *,
    success_count: int = 3,
) -> pd.DataFrame:
    rows = []
    for parameter, value in zip(
        HESTON_PARAM_NAMES,
        params.as_array(),
        strict=True,
    ):
        rows.append(
            {
                "parameter": parameter,
                "success_count": success_count,
                "mean": float(value),
                "std": 1.0e-5,
                "min": float(value) - 1.0e-4,
                "max": float(value) + 1.0e-4,
                "best_value": float(value),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_summary_report(
    params: HestonParams,
    *,
    quote_count: int = 60,
    expiry_count: int = 4,
    held_out: bool = True,
    success_count: int = 3,
    failure_count: int = 0,
) -> HestonCalibrationFitDiagnostics:
    ratio = _feller_ratio_for_params(params)
    return HestonCalibrationFitDiagnostics(
        meta={
            "diagnostic": "heston_calibration_fit",
            "objective_type": "vega_scaled_price",
            "backend": "gauss_legendre",
            "quote_count": quote_count,
            "expiry_count": expiry_count,
            "boundary_tolerance": 1.0e-3,
            "feller_ratio": ratio,
            "feller_satisfied": bool(ratio >= 1.0),
        },
        tables={
            "residuals": pd.DataFrame(columns=HESTON_CALIBRATION_FIT_RESIDUAL_COLUMNS),
            "smile_fit": pd.DataFrame(columns=HESTON_CALIBRATION_FIT_SMILE_COLUMNS),
            "iv_residual_grid": pd.DataFrame(
                columns=HESTON_CALIBRATION_FIT_IV_GRID_COLUMNS
            ),
            "parameter_recovery": pd.DataFrame(
                {
                    "parameter": HESTON_PARAM_NAMES,
                    "fitted": params.as_array(),
                }
            ),
            "constraint_diagnostics": _synthetic_constraint_diagnostics(params),
            "parameter_boundaries": _synthetic_parameter_boundaries(params),
            "multistart_parameter_dispersion": _synthetic_multistart_dispersion(
                params,
                success_count=success_count,
            ),
            "residual_buckets": pd.DataFrame(
                [
                    {
                        "bucket_type": "expiry",
                        "bucket": "1.0",
                        "n_quotes": quote_count,
                        "price_rmse": 0.03,
                        "iv_rmse_bps": 6.0,
                    }
                ]
            ),
            "quote_policy": pd.DataFrame(),
            "quote_policy_summary": pd.DataFrame(),
            "multistart_runs": _synthetic_multistart_runs(
                success_count=success_count,
                failure_count=failure_count,
            ),
            "held_out_errors": _synthetic_held_out_errors(include=held_out),
            "objective_slices": pd.DataFrame(),
            "kappa_profile": pd.DataFrame(),
        },
        arrays={"fitted_params": params.as_array()},
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
        "constraint_diagnostics",
        "parameter_boundaries",
        "multistart_parameter_dispersion",
        "residual_buckets",
        "quote_policy",
        "quote_policy_summary",
        "multistart_runs",
        "held_out_errors",
        "objective_slices",
        "kappa_profile",
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

    feller = report.tables["constraint_diagnostics"].set_index("constraint")
    assert "feller" in feller.index
    assert feller.loc["feller", "policy"] == "reported_not_hard_enforced"
    assert "feller_margin" in report.meta

    boundaries = report.tables["parameter_boundaries"]
    assert {"parameter", "hit_upper", "normalized_distance_to_upper"} <= set(
        boundaries.columns
    )

    buckets = report.tables["residual_buckets"]
    assert {"expiry", "log_moneyness", "option_right", "vega", "spread"} <= set(
        buckets["bucket_type"]
    )

    kappa_profile = report.tables["kappa_profile"]
    assert {"kappa", "cost", "profile_mode", "price_rmse"} <= set(kappa_profile.columns)


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


def test_heston_calibration_fit_reports_quote_warning_policy() -> None:
    quotes = _quotes()
    quote_diagnostics = {
        "quote_index": np.array([0, 1, 2, 3], dtype=np.int64),
        "warning_flags": np.array(
            [
                int(HestonIntegralWarning.NONFINITE_TOTAL),
                int(HestonIntegralWarning.PROBABILITY_OUT_OF_RANGE),
                int(HestonIntegralWarning.LARGE_TAIL_FRACTION),
                0,
            ],
            dtype=np.uint32,
        ),
        "persistent_backend_disagreement": np.array(
            [False, False, False, True],
            dtype=np.bool_,
        ),
    }

    report = run_heston_calibration_fit_diagnostics(
        quotes=quotes,
        fit=_true_params(),
        quad_cfg=_quad_cfg(),
        objective_type="vega_scaled_price",
        objective_slice_grid_size=2,
        quote_diagnostics=quote_diagnostics,
        fit_used_filtered_quotes=True,
    )

    policy = report.tables["quote_policy"].set_index("quote_index")
    assert policy.loc[0, "calibration_action"] == "block"
    assert policy.loc[1, "calibration_action"] == "quarantine"
    assert policy.loc[2, "calibration_action"] == "review"
    assert policy.loc[3, "calibration_action"] == "quarantine"
    assert bool(policy.loc[0, "fit_used_quote"]) is False
    assert bool(policy.loc[2, "fit_used_quote"]) is True

    assert report.meta["blocked_quote_count"] == 1
    assert report.meta["quarantined_quote_count"] == 2
    assert report.meta["review_quote_count"] == 1
    assert report.meta["quote_filtering_policy"] == "filtered_blocked_and_quarantined"


def test_heston_boundary_diagnostics_flag_upper_kappa_without_mutating_fit() -> None:
    quotes = _quotes()
    fit = _boundary_multistart_result()
    before = fit.best_params.as_array().copy()

    report = run_heston_calibration_fit_diagnostics(
        quotes=quotes,
        fit=fit,
        quad_cfg=_quad_cfg(),
        objective_slice_grid_size=2,
        boundary_tolerance=1.0e-3,
        include_kappa_profile=False,
    )

    np.testing.assert_allclose(fit.best_params.as_array(), before)
    assert report.meta["best_solution_boundary_bound"] is True
    assert report.meta["kappa_hit_upper_bound"] is True

    boundaries = report.tables["parameter_boundaries"].set_index("parameter")
    assert bool(boundaries.loc["kappa", "hit_upper"]) is True

    dispersion = report.tables["multistart_parameter_dispersion"].set_index("parameter")
    assert int(dispersion.loc["kappa", "upper_hit_count"]) == 2
    assert float(dispersion.loc["kappa", "boundary_hit_frac"]) == 1.0


def test_heston_calibration_fit_summary_clean_synthetic_fit() -> None:
    report = _synthetic_summary_report(
        HestonParams(kappa=1.5, vbar=0.04, eta=0.25, rho=-0.45, v=0.04)
    )

    summary = heston_calibration_fit_summary(report)

    assert summary["warning_labels"] == []
    assert summary["verdict"] == "usable"
    assert summary["quote_count"] == 60
    assert summary["expiry_count"] == 4
    assert summary["held_out_available"] is True
    assert summary["max_bucket_price_rmse"] == 0.03
    assert summary["max_bucket_iv_rmse_bps"] == 6.0
    assert summary["multistart_success_count"] == 3
    assert summary["multistart_failure_count"] == 0
    assert summarize_heston_calibration_fit(report) == summary


def test_heston_calibration_fit_summary_flags_kappa_near_upper_bound() -> None:
    report = _synthetic_summary_report(
        HestonParams(kappa=19.99, vbar=0.04, eta=0.50, rho=-0.45, v=0.04)
    )

    summary = heston_calibration_fit_summary(report)

    assert summary["kappa_near_upper_bound"] is True
    assert "PARAMETER_NEAR_BOUND" in summary["warning_labels"]
    assert "KAPPA_NEAR_UPPER_BOUND" in summary["warning_labels"]
    assert summary["verdict"] == "usable_with_caution"


def test_heston_calibration_fit_summary_flags_weak_feller() -> None:
    report = _synthetic_summary_report(
        HestonParams(kappa=0.5, vbar=0.01, eta=0.80, rho=-0.30, v=0.02)
    )

    summary = heston_calibration_fit_summary(report)

    assert summary["feller_satisfied"] is False
    assert "FELLER_WEAK_OR_VIOLATED" in summary["warning_labels"]


def test_heston_calibration_fit_summary_flags_high_eta() -> None:
    report = _synthetic_summary_report(
        HestonParams(kappa=15.0, vbar=0.60, eta=3.50, rho=-0.40, v=0.40)
    )

    summary = heston_calibration_fit_summary(report)

    assert summary["eta_high"] is True
    assert "ETA_HIGH" in summary["warning_labels"]
    assert "strong smile" in summary["public_note"]


def test_heston_calibration_fit_summary_missing_heldout_is_explicit() -> None:
    report = _synthetic_summary_report(
        HestonParams(kappa=1.5, vbar=0.04, eta=0.25, rho=-0.45, v=0.04),
        held_out=False,
    )

    summary = heston_calibration_fit_summary(report)

    assert summary["held_out_available"] is False
    assert summary["train_price_rmse"] is None
    assert summary["holdout_price_rmse"] is None
    assert summary["train_iv_rmse_bps"] is None
    assert summary["holdout_iv_rmse_bps"] is None


def test_heston_calibration_fit_summary_is_serialization_friendly() -> None:
    report = _synthetic_summary_report(
        HestonParams(kappa=1.5, vbar=0.04, eta=0.25, rho=-0.45, v=0.04)
    )

    summary = heston_calibration_fit_summary(report)
    frame = heston_calibration_fit_summary(report, as_frame=True)

    json.dumps(summary, allow_nan=False)
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[0] == 1
    assert frame.loc[0, "warning_labels"] == []
    assert frame.loc[0, "verdict"] == "usable"


def test_heston_calibration_fit_summary_calls_no_optimizer_or_pricer(
    monkeypatch,
) -> None:
    import option_pricing.diagnostics.heston.calibration_fit as calibration_fit_module

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("summary helper must not run pricing or optimization")

    monkeypatch.setattr(
        calibration_fit_module,
        "heston_price_from_ctx",
        fail_if_called,
    )
    monkeypatch.setattr(calibration_fit_module, "least_squares", fail_if_called)
    report = _synthetic_summary_report(
        HestonParams(kappa=1.5, vbar=0.04, eta=0.25, rho=-0.45, v=0.04)
    )

    summary = calibration_fit_module.heston_calibration_fit_summary(report)

    assert summary["verdict"] == "usable"
