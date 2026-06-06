"""Production-facing Heston calibration fit diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from ..._scipy_compat import least_squares
from ...models.heston.calibration.bounds import HestonCalibrationBounds
from ...models.heston.calibration.heston_types import (
    HestonMultistartResult,
    HestonObjectiveType,
    HestonParameterTransform,
    HestonQuoteSet,
)
from ...models.heston.calibration.objective import HestonObjective
from ...models.heston.calibration.regulate import feller_ratio, feller_satisfied
from ...models.heston.fourier import HestonBackend
from ...models.heston.params import HESTON_PARAM_NAMES, HestonParams
from ...numerics.quadrature import QuadratureConfig
from ...pricers.heston import heston_price_from_ctx
from ...types import OptionType
from ...typing import FloatArray
from ...vol.implied_vol_slice import implied_vol_black76_slice
from ._provenance import backend_config_meta
from .models import HestonCalibrationFitDiagnostics
from .quote_policy import heston_calibration_quote_policy_tables

_OBJECTIVE_SLICE_PAIRS: tuple[tuple[str, str], ...] = (
    ("kappa", "vbar"),
    ("eta", "rho"),
    ("v", "vbar"),
)
_DEFAULT_KAPPA_PROFILE_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)


def _params_to_dict(params: HestonParams, *, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}{name}": float(value)
        for name, value in zip(HESTON_PARAM_NAMES, params.as_array(), strict=True)
    }


def _price_heston_quotes(
    quotes: HestonQuoteSet,
    params: HestonParams,
    *,
    backend: HestonBackend,
    quad_cfg: QuadratureConfig | None,
) -> FloatArray:
    prices = np.empty(quotes.n_quotes, dtype=np.float64)

    for tau in np.unique(quotes.expiry):
        idx_tau = np.flatnonzero(quotes.expiry == tau)
        for is_call_value, kind in (
            (True, OptionType.CALL),
            (False, OptionType.PUT),
        ):
            idx = idx_tau[quotes.is_call[idx_tau] == is_call_value]
            if idx.size == 0:
                continue
            prices[idx] = np.asarray(
                heston_price_from_ctx(
                    kind=kind,
                    strike=quotes.strike[idx],
                    tau=float(tau),
                    ctx=quotes.ctx,
                    params=params,
                    backend=backend,
                    quad_cfg=quad_cfg,
                ),
                dtype=np.float64,
            )

    return prices


def _implied_vols_from_prices(
    quotes: HestonQuoteSet,
    prices: FloatArray,
) -> FloatArray:
    price_arr = np.asarray(prices, dtype=np.float64).reshape(-1)
    if price_arr.shape != (quotes.n_quotes,):
        raise ValueError(
            f"prices must have shape ({quotes.n_quotes},), got {price_arr.shape}."
        )

    implied_vol = np.full(quotes.n_quotes, np.nan, dtype=np.float64)
    initial = (
        np.full(quotes.n_quotes, 0.2, dtype=np.float64)
        if quotes.iv_mid is None
        else np.asarray(quotes.iv_mid, dtype=np.float64)
    )

    for tau in np.unique(quotes.expiry):
        idx = np.flatnonzero(quotes.expiry == tau)
        implied_vol[idx] = np.asarray(
            implied_vol_black76_slice(
                forward=quotes.ctx.fwd(float(tau)),
                strikes=quotes.strike[idx],
                tau=float(tau),
                df=quotes.ctx.df(float(tau)),
                prices=price_arr[idx],
                is_call=quotes.is_call[idx],
                initial_sigma=initial[idx],
            ),
            dtype=np.float64,
        )

    return implied_vol


def _resolved_fit(
    fit: HestonParams | HestonMultistartResult,
) -> tuple[HestonParams, HestonMultistartResult | None]:
    if isinstance(fit, HestonMultistartResult):
        return fit.best_params, fit
    return fit, None


def _coerce_real_scalar(value: object, *, label: str) -> float:
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, Real):
            return float(scalar)
    raise TypeError(f"{label} must be a real scalar, got {type(value).__name__}.")


def _resolve_mask(
    held_out_mask: Sequence[bool] | np.ndarray | None,
    *,
    n_quotes: int,
) -> tuple[np.ndarray, bool]:
    if held_out_mask is None:
        return np.zeros(n_quotes, dtype=np.bool_), False

    mask = np.asarray(held_out_mask, dtype=np.bool_).reshape(-1)
    if mask.shape != (n_quotes,):
        raise ValueError(f"held_out_mask must have shape ({n_quotes},).")
    return mask, True


def _resolve_sqrt_weights(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None,
) -> FloatArray:
    if sqrt_weights is None:
        if quotes.sqrt_weights is None:
            return np.ones(quotes.n_quotes, dtype=np.float64)
        return np.asarray(quotes.sqrt_weights, dtype=np.float64)

    weights = np.asarray(sqrt_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (quotes.n_quotes,):
        raise ValueError(f"sqrt_weights must have shape ({quotes.n_quotes},).")
    if not np.all(np.isfinite(weights)):
        raise ValueError("sqrt_weights must be finite.")
    if np.any(weights < 0.0):
        raise ValueError("sqrt_weights must be nonnegative.")
    return weights


def _market_iv(quotes: HestonQuoteSet) -> FloatArray:
    if quotes.iv_mid is None:
        return np.full(quotes.n_quotes, np.nan, dtype=np.float64)
    return np.asarray(quotes.iv_mid, dtype=np.float64)


def _optional_array(values: FloatArray | None, *, n_quotes: int) -> FloatArray:
    if values is None:
        return np.full(n_quotes, np.nan, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _residual_tables(
    *,
    quotes: HestonQuoteSet,
    model_prices: FloatArray,
    model_iv: FloatArray,
    sqrt_weights: FloatArray,
    held_out_mask: np.ndarray,
    mask_provided: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market_iv = _market_iv(quotes)
    price_residual = np.asarray(model_prices - quotes.mid, dtype=np.float64)
    iv_residual = np.asarray(model_iv - market_iv, dtype=np.float64)
    iv_residual_bps = np.asarray(iv_residual * 1.0e4, dtype=np.float64)
    sample = np.where(held_out_mask, "held_out", "train")
    if not mask_provided:
        sample = np.full(quotes.n_quotes, "fit", dtype=object)

    residuals = pd.DataFrame(
        {
            "quote_index": np.arange(quotes.n_quotes, dtype=np.int64),
            "expiry": np.asarray(quotes.expiry, dtype=np.float64),
            "strike": np.asarray(quotes.strike, dtype=np.float64),
            "log_moneyness": np.asarray(quotes.log_moneyness, dtype=np.float64),
            "is_call": np.asarray(quotes.is_call, dtype=np.bool_),
            "market_price": np.asarray(quotes.mid, dtype=np.float64),
            "model_price": np.asarray(model_prices, dtype=np.float64),
            "price_residual": price_residual,
            "market_iv": market_iv,
            "model_iv": np.asarray(model_iv, dtype=np.float64),
            "iv_residual": iv_residual,
            "iv_residual_bps": iv_residual_bps,
            "bs_vega": _optional_array(quotes.bs_vega, n_quotes=quotes.n_quotes),
            "sqrt_weight": np.asarray(sqrt_weights, dtype=np.float64),
            "is_held_out": np.asarray(held_out_mask, dtype=np.bool_),
            "sample": sample,
        }
    )

    smile_fit = residuals[
        [
            "quote_index",
            "expiry",
            "strike",
            "log_moneyness",
            "is_call",
            "market_iv",
            "model_iv",
            "iv_residual_bps",
            "is_held_out",
        ]
    ].copy()

    iv_residual_grid = residuals[
        ["quote_index", "expiry", "strike", "log_moneyness", "iv_residual_bps"]
    ].copy()
    iv_residual_grid["grid_kind"] = "long_form_no_interpolation"
    iv_residual_grid.attrs["notes"] = [
        "NOTE: IV residual grids are returned long-form with no interpolation; "
        "irregular quote grids therefore remain explicit."
    ]

    return residuals, smile_fit, iv_residual_grid


def _metric_values(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan)
    abs_values = np.abs(finite)
    return (
        float(np.sqrt(np.mean(finite * finite))),
        float(np.mean(abs_values)),
        float(np.max(abs_values)),
    )


def _held_out_errors(
    residuals: pd.DataFrame,
    *,
    mask_provided: bool,
) -> pd.DataFrame:
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
    if not mask_provided:
        table = pd.DataFrame(columns=columns)
        table.attrs["notes"] = [
            "No held_out_mask was supplied; held-out evaluation was not run."
        ]
        return table

    rows: list[dict[str, float | int | str]] = []
    for sample_name in ("train", "held_out"):
        group = residuals.loc[residuals["sample"] == sample_name]
        price_rmse, price_mae, price_max_abs = _metric_values(
            group["price_residual"].to_numpy(dtype=np.float64, copy=False)
        )
        iv_rmse, iv_mae, iv_max_abs = _metric_values(
            group["iv_residual_bps"].to_numpy(dtype=np.float64, copy=False)
        )
        rows.append(
            {
                "sample": sample_name,
                "n_quotes": int(len(group)),
                "price_rmse": price_rmse,
                "price_mae": price_mae,
                "price_max_abs": price_max_abs,
                "iv_rmse_bps": iv_rmse,
                "iv_mae_bps": iv_mae,
                "iv_max_abs_bps": iv_max_abs,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _error_summary_row(
    *,
    bucket_type: str,
    bucket: str,
    group: pd.DataFrame,
    notes: str = "",
) -> dict[str, Any]:
    price_rmse, price_mae, price_max_abs = _metric_values(
        group["price_residual"].to_numpy(dtype=np.float64, copy=False)
    )
    iv_rmse, iv_mae, iv_max_abs = _metric_values(
        group["iv_residual_bps"].to_numpy(dtype=np.float64, copy=False)
    )
    return {
        "bucket_type": bucket_type,
        "bucket": bucket,
        "n_quotes": int(len(group)),
        "price_rmse": price_rmse,
        "price_mae": price_mae,
        "price_max_abs": price_max_abs,
        "iv_rmse_bps": iv_rmse,
        "iv_mae_bps": iv_mae,
        "iv_max_abs_bps": iv_max_abs,
        "notes": notes,
    }


def _quantile_bucket_labels(values: np.ndarray, *, prefix: str) -> pd.Series:
    x = pd.Series(np.asarray(values, dtype=np.float64))
    finite = np.isfinite(x.to_numpy(dtype=np.float64, copy=False))
    labels = pd.Series(np.full(len(x), "unavailable", dtype=object))
    if np.count_nonzero(finite) < 2:
        return labels

    ranks = x[finite].rank(method="first")
    try:
        buckets = pd.qcut(
            ranks,
            q=min(3, int(np.count_nonzero(finite))),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return labels

    bucket_names = {
        0: f"{prefix}_low",
        1: f"{prefix}_mid",
        2: f"{prefix}_high",
    }
    labels.loc[finite] = [
        bucket_names.get(int(value), f"{prefix}_{int(value)}")
        for value in np.asarray(buckets, dtype=np.int64)
    ]
    return labels


def _residual_bucket_table(
    residuals: pd.DataFrame, quotes: HestonQuoteSet
) -> pd.DataFrame:
    columns = [
        "bucket_type",
        "bucket",
        "n_quotes",
        "price_rmse",
        "price_mae",
        "price_max_abs",
        "iv_rmse_bps",
        "iv_mae_bps",
        "iv_max_abs_bps",
        "notes",
    ]
    rows: list[dict[str, Any]] = []

    for _expiry, group in residuals.groupby("expiry", sort=True):
        expiry_value = float(group["expiry"].to_numpy(dtype=np.float64, copy=False)[0])
        rows.append(
            _error_summary_row(
                bucket_type="expiry",
                bucket=f"{expiry_value:.10g}",
                group=group,
            )
        )

    y = residuals["log_moneyness"].to_numpy(dtype=np.float64, copy=False)
    y_edges = np.array([-0.10, -0.03, 0.03, 0.10], dtype=np.float64)
    y_labels = ["y<=-0.10", "-0.10<y<=-0.03", "|y|<=0.03", "0.03<y<=0.10", "y>0.10"]
    y_bucket_index = np.digitize(y, y_edges, right=True)
    y_bucket_values = np.asarray(y_labels, dtype=object)[
        np.clip(y_bucket_index, 0, len(y_labels) - 1)
    ]
    y_bucket_values = np.where(np.isfinite(y), y_bucket_values, "unavailable")
    residuals_mny = residuals.assign(_bucket=y_bucket_values)
    for bucket, group in residuals_mny.groupby("_bucket", sort=False, observed=False):
        if group.empty:
            continue
        rows.append(
            _error_summary_row(
                bucket_type="log_moneyness",
                bucket=str(bucket),
                group=group,
            )
        )

    for is_call, group in residuals.groupby("is_call", sort=True):
        rows.append(
            _error_summary_row(
                bucket_type="option_right",
                bucket="call" if bool(is_call) else "put",
                group=group,
            )
        )

    vega = residuals["bs_vega"].to_numpy(dtype=np.float64, copy=False)
    vega_labels = _quantile_bucket_labels(vega, prefix="vega")
    residuals_vega = residuals.assign(_bucket=vega_labels.to_numpy(dtype=object))
    for bucket, group in residuals_vega.groupby("_bucket", sort=True):
        rows.append(
            _error_summary_row(
                bucket_type="vega",
                bucket=str(bucket),
                group=group,
                notes=(
                    "quotes.bs_vega was unavailable"
                    if str(bucket) == "unavailable"
                    else ""
                ),
            )
        )

    if quotes.bid is not None and quotes.ask is not None:
        spread = np.asarray(quotes.ask - quotes.bid, dtype=np.float64)
        spread_labels = _quantile_bucket_labels(spread, prefix="spread")
    else:
        spread_labels = pd.Series(np.full(quotes.n_quotes, "unavailable", dtype=object))
    residuals_spread = residuals.assign(_bucket=spread_labels.to_numpy(dtype=object))
    for bucket, group in residuals_spread.groupby("_bucket", sort=True):
        rows.append(
            _error_summary_row(
                bucket_type="spread",
                bucket=str(bucket),
                group=group,
                notes=(
                    "quotes.bid/quotes.ask were unavailable"
                    if str(bucket) == "unavailable"
                    else ""
                ),
            )
        )

    return pd.DataFrame(rows, columns=columns)


def _parameter_recovery_table(
    *,
    fitted_params: HestonParams,
    true_params: HestonParams | None,
    seed_params: HestonParams | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fitted = _params_to_dict(fitted_params)
    truth = None if true_params is None else _params_to_dict(true_params)
    seed = None if seed_params is None else _params_to_dict(seed_params)

    for parameter in HESTON_PARAM_NAMES:
        row: dict[str, Any] = {
            "parameter": parameter,
            "fitted": fitted[parameter],
        }
        if seed is not None:
            row["seed"] = seed[parameter]
        if truth is not None:
            true_value = truth[parameter]
            fitted_value = fitted[parameter]
            diff = fitted_value - true_value
            row.update(
                {
                    "true": true_value,
                    "fit_minus_true": diff,
                    "absolute_error": abs(diff),
                    "relative_error": (
                        np.nan
                        if abs(true_value) <= 1.0e-12
                        else abs(diff) / abs(true_value)
                    ),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _constraint_diagnostics_table(params: HestonParams) -> pd.DataFrame:
    feller_margin = 2.0 * float(params.kappa) * float(params.vbar) - float(
        params.eta
    ) * float(params.eta)
    ratio = feller_ratio(params)
    satisfied = feller_satisfied(params)
    return pd.DataFrame(
        [
            {
                "constraint": "feller",
                "value": float(ratio),
                "margin": float(feller_margin),
                "satisfied": bool(satisfied),
                "policy": "reported_not_hard_enforced",
                "notes": (
                    "Feller is visible calibration evidence. It is not hard-blocked "
                    "by default; use HestonRegConfig.feller_penalty_weight for an "
                    "optional soft penalty."
                ),
            }
        ]
    )


def _parameter_boundary_table(
    *,
    params: HestonParams,
    bounds: HestonCalibrationBounds,
    tolerance: float,
) -> pd.DataFrame:
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("boundary_tolerance must be finite and nonnegative.")

    values = params.as_array()
    lower = bounds.lower_array()
    upper = bounds.upper_array()
    width = upper - lower
    fraction = (values - lower) / width
    distance_to_lower = values - lower
    distance_to_upper = upper - values
    lower_hit = fraction <= tol
    upper_hit = (1.0 - fraction) <= tol

    rows = []
    for i, name in enumerate(HESTON_PARAM_NAMES):
        rows.append(
            {
                "parameter": name,
                "value": float(values[i]),
                "lower_bound": float(lower[i]),
                "upper_bound": float(upper[i]),
                "normalized_position": float(fraction[i]),
                "distance_to_lower": float(distance_to_lower[i]),
                "distance_to_upper": float(distance_to_upper[i]),
                "normalized_distance_to_lower": float(fraction[i]),
                "normalized_distance_to_upper": float(1.0 - fraction[i]),
                "hit_lower": bool(lower_hit[i]),
                "hit_upper": bool(upper_hit[i]),
                "hit_boundary": bool(lower_hit[i] or upper_hit[i]),
                "tolerance": tol,
            }
        )
    return pd.DataFrame(rows)


def _multistart_parameter_dispersion_table(
    *,
    multistart: HestonMultistartResult | None,
    bounds: HestonCalibrationBounds,
    tolerance: float,
) -> pd.DataFrame:
    columns = [
        "parameter",
        "success_count",
        "mean",
        "std",
        "min",
        "max",
        "best_value",
        "best_hit_lower",
        "best_hit_upper",
        "lower_hit_count",
        "upper_hit_count",
        "boundary_hit_count",
        "boundary_hit_frac",
    ]
    if multistart is None or not multistart.successful_runs:
        table = pd.DataFrame(columns=columns)
        table.attrs["notes"] = ["No successful multistart runs were supplied."]
        return table

    success_params = np.vstack(
        [
            run.fitted_params.as_array()
            for run in multistart.successful_runs
            if run.fitted_params is not None
        ]
    )
    best = multistart.best_params.as_array()
    lower = bounds.lower_array()
    upper = bounds.upper_array()
    fraction = (success_params - lower[None, :]) / (upper - lower)[None, :]
    best_fraction = (best - lower) / (upper - lower)
    lower_hit = fraction <= float(tolerance)
    upper_hit = (1.0 - fraction) <= float(tolerance)
    best_lower_hit = best_fraction <= float(tolerance)
    best_upper_hit = (1.0 - best_fraction) <= float(tolerance)

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(HESTON_PARAM_NAMES):
        boundary_hit = lower_hit[:, i] | upper_hit[:, i]
        rows.append(
            {
                "parameter": name,
                "success_count": int(success_params.shape[0]),
                "mean": float(np.mean(success_params[:, i])),
                "std": float(np.std(success_params[:, i], ddof=0)),
                "min": float(np.min(success_params[:, i])),
                "max": float(np.max(success_params[:, i])),
                "best_value": float(best[i]),
                "best_hit_lower": bool(best_lower_hit[i]),
                "best_hit_upper": bool(best_upper_hit[i]),
                "lower_hit_count": int(np.sum(lower_hit[:, i])),
                "upper_hit_count": int(np.sum(upper_hit[:, i])),
                "boundary_hit_count": int(np.sum(boundary_hit)),
                "boundary_hit_frac": float(np.mean(boundary_hit)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _multistart_runs_table(
    multistart: HestonMultistartResult | None,
) -> pd.DataFrame:
    columns = [
        "seed_index",
        "success",
        "cost",
        "optimality",
        "nfev",
        "njev",
        "status",
        "message",
        "best_run",
        *[f"seed_{name}" for name in HESTON_PARAM_NAMES],
        *[f"fitted_{name}" for name in HESTON_PARAM_NAMES],
    ]
    if multistart is None:
        table = pd.DataFrame(columns=columns)
        table.attrs["notes"] = ["No HestonMultistartResult was supplied."]
        return table

    rows: list[dict[str, Any]] = []
    for run in multistart.runs:
        fitted_values = (
            {f"fitted_{name}": np.nan for name in HESTON_PARAM_NAMES}
            if run.fitted_params is None
            else _params_to_dict(run.fitted_params, prefix="fitted_")
        )
        rows.append(
            {
                "seed_index": int(run.seed_index),
                "success": bool(run.success),
                "cost": float(run.cost),
                "optimality": run.optimality,
                "nfev": run.nfev,
                "njev": run.njev,
                "status": run.status,
                "message": str(run.message),
                "best_run": bool(run == multistart.best_run),
                **_params_to_dict(run.seed_params, prefix="seed_"),
                **fitted_values,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _candidate_values(
    *,
    base: float,
    lower: float,
    upper: float,
    parameter: str,
    grid_size: int,
    relative_width: float,
) -> np.ndarray:
    offsets = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float64)
    if parameter == "rho":
        width = min(0.25, 0.25 * (upper - lower))
    else:
        width = max(abs(base) * float(relative_width), 0.03 * (upper - lower))
    return np.asarray(np.clip(base + offsets * width, lower, upper), dtype=np.float64)


def _clip_params(values: np.ndarray, bounds: HestonCalibrationBounds) -> HestonParams:
    lower = bounds.lower_array()
    upper = bounds.upper_array()
    clipped = np.asarray(np.clip(values, lower, upper), dtype=np.float64)
    return HestonParams(
        kappa=float(clipped[0]),
        vbar=float(clipped[1]),
        eta=float(clipped[2]),
        rho=float(clipped[3]),
        v=float(clipped[4]),
    )


def _objective_cost(
    *,
    objective: HestonObjective,
    params: HestonParams,
    bounds: HestonCalibrationBounds,
    parameter_transform: HestonParameterTransform,
) -> float:
    if parameter_transform == "bounded":
        raw = params.transform_to_bounded_unconstrained(bounds)
    else:
        raw = params.transform_to_unconstrained()
    residual = objective.residual(raw)
    return float(0.5 * np.sum(residual * residual))


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    raw = np.asarray(x, dtype=np.float64)
    out = np.empty_like(raw, dtype=np.float64)
    positive = raw >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
    exp_x = np.exp(raw[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _params_from_fixed_kappa_raw_free(
    *,
    kappa: float,
    raw_free: np.ndarray,
    bounds: HestonCalibrationBounds,
) -> HestonParams:
    lower = bounds.lower_array()
    upper = bounds.upper_array()
    z = _stable_sigmoid(np.asarray(raw_free, dtype=np.float64).reshape(-1))
    if z.size != 4:
        raise ValueError("raw_free must contain vbar, eta, rho, and v raw values.")
    values = lower.copy()
    values[0] = float(kappa)
    values[1:] = lower[1:] + (upper[1:] - lower[1:]) * z
    return HestonParams(
        kappa=float(values[0]),
        vbar=float(values[1]),
        eta=float(values[2]),
        rho=float(values[3]),
        v=float(values[4]),
    )


def _profile_error_metrics(
    *,
    quotes: HestonQuoteSet,
    params: HestonParams,
    backend: HestonBackend,
    quad_cfg: QuadratureConfig | None,
) -> dict[str, float]:
    model_prices = _price_heston_quotes(
        quotes,
        params,
        backend=backend,
        quad_cfg=quad_cfg,
    )
    model_iv = _implied_vols_from_prices(quotes, model_prices)
    price_residual = np.asarray(model_prices - quotes.mid, dtype=np.float64)
    market_iv = _market_iv(quotes)
    iv_residual_bps = np.asarray((model_iv - market_iv) * 1.0e4, dtype=np.float64)
    price_rmse, price_mae, price_max_abs = _metric_values(price_residual)
    iv_rmse, iv_mae, iv_max_abs = _metric_values(iv_residual_bps)
    return {
        "price_rmse": price_rmse,
        "price_mae": price_mae,
        "price_max_abs": price_max_abs,
        "iv_rmse_bps": iv_rmse,
        "iv_mae_bps": iv_mae,
        "iv_max_abs_bps": iv_max_abs,
    }


def run_heston_kappa_profile_diagnostics(
    *,
    quotes: HestonQuoteSet,
    base_params: HestonParams,
    kappa_grid: Sequence[float] = _DEFAULT_KAPPA_PROFILE_GRID,
    objective_type: HestonObjectiveType = "vega_scaled_price",
    sqrt_weights: FloatArray | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    bounds: HestonCalibrationBounds | None = None,
    parameter_transform: HestonParameterTransform = "bounded",
    refit_remaining: bool = False,
    loss: str = "soft_l1",
    max_nfev: int | None = None,
) -> pd.DataFrame:
    """Profile the Heston objective over fixed kappa values.

    By default this is a lightweight hold-other-parameters diagnostic. Set
    ``refit_remaining=True`` to optimize ``vbar, eta, rho, v`` for each fixed
    kappa with finite-difference least squares. The helper is diagnostic-only
    and does not mutate or replace the supplied calibration result.
    """

    if parameter_transform != "bounded":
        raise ValueError(
            "kappa profile diagnostics currently require bounded transform."
        )

    resolved_bounds = HestonCalibrationBounds() if bounds is None else bounds
    grid = np.asarray(list(kappa_grid), dtype=np.float64).reshape(-1)
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("kappa_grid must contain finite values.")
    kappa_lo, kappa_hi = resolved_bounds.kappa
    if np.any((grid < float(kappa_lo)) | (grid > float(kappa_hi))):
        raise ValueError("kappa_grid values must lie inside calibration bounds.")

    objective = HestonObjective(
        quotes=quotes,
        sqrt_weights=sqrt_weights,
        objective_type=objective_type,
        backend=backend,
        quad_cfg=quad_cfg,
        parameter_transform=parameter_transform,
        bounds=resolved_bounds,
    )
    base_raw = base_params.transform_to_bounded_unconstrained(resolved_bounds)
    base_cost = _objective_cost(
        objective=objective,
        params=base_params,
        bounds=resolved_bounds,
        parameter_transform=parameter_transform,
    )

    rows: list[dict[str, Any]] = []
    for kappa in grid:
        kappa_value = float(kappa)
        status = None
        message = "held other parameters fixed"
        success = True
        nfev = 0

        params = HestonParams(
            kappa=kappa_value,
            vbar=base_params.vbar,
            eta=base_params.eta,
            rho=base_params.rho,
            v=base_params.v,
        )
        raw_free = base_raw[1:].copy()

        if refit_remaining:

            def residual_free(
                raw_free_candidate: np.ndarray,
                *,
                fixed_kappa: float = kappa_value,
            ) -> np.ndarray:
                candidate = _params_from_fixed_kappa_raw_free(
                    kappa=fixed_kappa,
                    raw_free=np.asarray(raw_free_candidate, dtype=np.float64),
                    bounds=resolved_bounds,
                )
                raw_full = candidate.transform_to_bounded_unconstrained(resolved_bounds)
                return objective.residual(raw_full)

            res = least_squares(
                fun=residual_free,
                x0=raw_free,
                jac="2-point",
                loss=loss,
                max_nfev=max_nfev,
            )
            success = bool(res.success and np.all(np.isfinite(res.x)))
            status = (
                int(res.status) if getattr(res, "status", None) is not None else None
            )
            message = str(res.message)
            nfev = int(res.nfev) if getattr(res, "nfev", None) is not None else 0
            if success:
                params = _params_from_fixed_kappa_raw_free(
                    kappa=kappa_value,
                    raw_free=np.asarray(res.x, dtype=np.float64),
                    bounds=resolved_bounds,
                )

        cost = _objective_cost(
            objective=objective,
            params=params,
            bounds=resolved_bounds,
            parameter_transform=parameter_transform,
        )
        metrics = _profile_error_metrics(
            quotes=quotes,
            params=params,
            backend=backend,
            quad_cfg=quad_cfg,
        )
        row = {
            "kappa": kappa_value,
            "profile_mode": (
                "refit_remaining" if refit_remaining else "hold_other_params_fixed"
            ),
            "success": bool(success),
            "cost": float(cost),
            "delta_cost_vs_base": float(cost - base_cost),
            "nfev": int(nfev),
            "status": status,
            "message": message,
            **_params_to_dict(params, prefix="fitted_"),
            **metrics,
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    table.attrs["notes"] = [
        "Kappa profiles diagnose weak identification; a flat curve is not an "
        "instruction to widen bounds.",
        "The default mode holds other parameters fixed. Set refit_remaining=True "
        "for a slower conditional profile over the remaining Heston parameters.",
    ]
    return table


def _objective_slices_table(
    *,
    quotes: HestonQuoteSet,
    fitted_params: HestonParams,
    objective_type: HestonObjectiveType,
    sqrt_weights: FloatArray | None,
    backend: HestonBackend,
    quad_cfg: QuadratureConfig | None,
    bounds: HestonCalibrationBounds,
    parameter_transform: HestonParameterTransform,
    grid_size: int,
    relative_width: float,
    pairs: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    if int(grid_size) < 2:
        raise ValueError("objective_slice_grid_size must be at least 2.")
    if float(relative_width) <= 0.0:
        raise ValueError("objective_slice_relative_width must be positive.")

    objective = HestonObjective(
        quotes=quotes,
        sqrt_weights=sqrt_weights,
        objective_type=objective_type,
        backend=backend,
        quad_cfg=quad_cfg,
        parameter_transform=parameter_transform,
        bounds=bounds if parameter_transform == "bounded" else None,
    )
    fitted_cost = _objective_cost(
        objective=objective,
        params=fitted_params,
        bounds=bounds,
        parameter_transform=parameter_transform,
    )

    param_index = {name: idx for idx, name in enumerate(HESTON_PARAM_NAMES)}
    base = fitted_params.as_array()
    lower = bounds.lower_array()
    upper = bounds.upper_array()
    rows: list[dict[str, Any]] = []

    for parameter_x, parameter_y in pairs:
        if parameter_x not in param_index or parameter_y not in param_index:
            raise ValueError("objective slice parameters must be Heston parameters.")
        ix = param_index[parameter_x]
        iy = param_index[parameter_y]
        x_values = _candidate_values(
            base=float(base[ix]),
            lower=float(lower[ix]),
            upper=float(upper[ix]),
            parameter=parameter_x,
            grid_size=grid_size,
            relative_width=relative_width,
        )
        y_values = _candidate_values(
            base=float(base[iy]),
            lower=float(lower[iy]),
            upper=float(upper[iy]),
            parameter=parameter_y,
            grid_size=grid_size,
            relative_width=relative_width,
        )

        for grid_i, value_x in enumerate(x_values):
            for grid_j, value_y in enumerate(y_values):
                candidate = base.copy()
                candidate[ix] = float(value_x)
                candidate[iy] = float(value_y)
                candidate_params = _clip_params(candidate, bounds)
                cost = _objective_cost(
                    objective=objective,
                    params=candidate_params,
                    bounds=bounds,
                    parameter_transform=parameter_transform,
                )
                rows.append(
                    {
                        "slice_name": f"{parameter_x}_vs_{parameter_y}",
                        "parameter_x": parameter_x,
                        "parameter_y": parameter_y,
                        "grid_i": int(grid_i),
                        "grid_j": int(grid_j),
                        "value_x": float(value_x),
                        "value_y": float(value_y),
                        "fitted_x": float(base[ix]),
                        "fitted_y": float(base[iy]),
                        "cost": cost,
                        "delta_cost": cost - fitted_cost,
                        "note": (
                            "LIMITATION: Objective slices are local constrained-"
                            "coordinate approximations, not global identifiability "
                            "proofs."
                        ),
                    }
                )

    return pd.DataFrame(rows)


def run_heston_calibration_fit_diagnostics(
    *,
    quotes: HestonQuoteSet,
    fit: HestonParams | HestonMultistartResult,
    true_params: HestonParams | None = None,
    seed_params: HestonParams | None = None,
    held_out_mask: Sequence[bool] | np.ndarray | None = None,
    objective_type: HestonObjectiveType | None = None,
    objective_metadata: Mapping[str, Any] | None = None,
    backend: HestonBackend | None = None,
    quad_cfg: QuadratureConfig | None = None,
    sqrt_weights: FloatArray | None = None,
    bounds: HestonCalibrationBounds | None = None,
    parameter_transform: HestonParameterTransform = "bounded",
    boundary_tolerance: float = 1.0e-3,
    include_kappa_profile: bool = True,
    kappa_profile_grid: Sequence[float] = _DEFAULT_KAPPA_PROFILE_GRID,
    kappa_profile_refit_remaining: bool = False,
    kappa_profile_max_nfev: int | None = None,
    objective_slice_grid_size: int = 3,
    objective_slice_relative_width: float = 0.25,
    objective_slice_pairs: Sequence[tuple[str, str]] = _OBJECTIVE_SLICE_PAIRS,
    quote_diagnostics: pd.DataFrame | Mapping[str, Any] | None = None,
    fit_used_filtered_quotes: bool = False,
) -> HestonCalibrationFitDiagnostics:
    """Build final Heston calibration evidence tables for a fitted quote set.

    The function runs no optimizer. It reprices the supplied quotes with the
    fitted Heston parameters, computes implied-vol residuals where possible, and
    packages calibration evidence into deterministic DataFrames.

    NOTE: Held-out rows are evaluation partitions for the supplied fit.
    The function cannot infer whether the fitted parameters were calibrated
    without seeing those rows.
    """
    fitted_params, multistart = _resolved_fit(fit)
    resolved_objective = (
        multistart.objective_type
        if objective_type is None and multistart is not None
        else ("vega_scaled_price" if objective_type is None else objective_type)
    )
    resolved_backend = (
        multistart.backend
        if backend is None and multistart is not None
        else ("gauss_legendre" if backend is None else backend)
    )
    resolved_bounds = HestonCalibrationBounds() if bounds is None else bounds
    resolved_seed = (
        multistart.best_run.seed_params
        if seed_params is None and multistart is not None
        else seed_params
    )
    resolved_sqrt_weights = _resolve_sqrt_weights(quotes, sqrt_weights)
    mask, mask_provided = _resolve_mask(held_out_mask, n_quotes=quotes.n_quotes)

    model_prices = _price_heston_quotes(
        quotes,
        fitted_params,
        backend=resolved_backend,
        quad_cfg=quad_cfg,
    )
    model_iv = _implied_vols_from_prices(quotes, model_prices)
    residuals, smile_fit, iv_residual_grid = _residual_tables(
        quotes=quotes,
        model_prices=model_prices,
        model_iv=model_iv,
        sqrt_weights=resolved_sqrt_weights,
        held_out_mask=mask,
        mask_provided=mask_provided,
    )

    held_out_errors = _held_out_errors(residuals, mask_provided=mask_provided)
    parameter_recovery = _parameter_recovery_table(
        fitted_params=fitted_params,
        true_params=true_params,
        seed_params=resolved_seed,
    )
    constraint_diagnostics = _constraint_diagnostics_table(fitted_params)
    parameter_boundaries = _parameter_boundary_table(
        params=fitted_params,
        bounds=resolved_bounds,
        tolerance=boundary_tolerance,
    )
    multistart_parameter_dispersion = _multistart_parameter_dispersion_table(
        multistart=multistart,
        bounds=resolved_bounds,
        tolerance=boundary_tolerance,
    )
    residual_buckets = _residual_bucket_table(residuals, quotes)
    quote_policy, quote_policy_summary, quote_policy_meta = (
        heston_calibration_quote_policy_tables(
            n_quotes=quotes.n_quotes,
            quote_diagnostics=quote_diagnostics,
            fit_used_filtered_quotes=fit_used_filtered_quotes,
        )
    )
    multistart_runs = _multistart_runs_table(multistart)
    objective_slices = _objective_slices_table(
        quotes=quotes,
        fitted_params=fitted_params,
        objective_type=resolved_objective,
        sqrt_weights=sqrt_weights,
        backend=resolved_backend,
        quad_cfg=quad_cfg,
        bounds=resolved_bounds,
        parameter_transform=parameter_transform,
        grid_size=objective_slice_grid_size,
        relative_width=objective_slice_relative_width,
        pairs=objective_slice_pairs,
    )
    kappa_profile = (
        run_heston_kappa_profile_diagnostics(
            quotes=quotes,
            base_params=fitted_params,
            kappa_grid=kappa_profile_grid,
            objective_type=resolved_objective,
            sqrt_weights=sqrt_weights,
            backend=resolved_backend,
            quad_cfg=quad_cfg,
            bounds=resolved_bounds,
            parameter_transform=parameter_transform,
            refit_remaining=kappa_profile_refit_remaining,
            max_nfev=kappa_profile_max_nfev,
        )
        if include_kappa_profile
        else pd.DataFrame(
            columns=[
                "kappa",
                "profile_mode",
                "success",
                "cost",
                "delta_cost_vs_base",
                "nfev",
                "status",
                "message",
            ]
        )
    )

    backend_meta = backend_config_meta(
        backend=resolved_backend,
        quad_cfg=quad_cfg,
        rule=None,
    )
    notes = [
        "LIMITATION: Heston vanilla calibration can be weakly identifiable; "
        "residual evidence should be read alongside multistart and slice tables.",
        "NOTE: Vega-scaled price residuals are an IV-like calibration proxy, "
        "not direct IV RMSE unless implied vols are inverted and reported.",
        "NOTE: Objective slices are local diagnostics and can be runtime- and "
        "coordinate-convention dependent.",
        "NOTE: IV residual grids are long-form with no interpolation so "
        "irregular quote grids remain explicit.",
    ]
    meta: dict[str, Any] = {
        "diagnostic": "heston_calibration_fit",
        "objective_type": resolved_objective,
        "objective_metadata": dict(objective_metadata or {}),
        "backend": resolved_backend,
        "backend_config": backend_meta,
        "quote_count": int(quotes.n_quotes),
        "expiry_count": int(np.unique(quotes.expiry).size),
        "strike_count": int(np.unique(quotes.strike).size),
        "log_moneyness_count": int(np.unique(quotes.log_moneyness).size),
        "true_params_provided": true_params is not None,
        "held_out_mask_provided": mask_provided,
        "multistart_result_provided": multistart is not None,
        "objective_slice_grid_size": int(objective_slice_grid_size),
        "boundary_tolerance": float(boundary_tolerance),
        "best_solution_boundary_bound": bool(
            parameter_boundaries["hit_boundary"].any()
        ),
        "best_solution_upper_boundary_bound": bool(
            parameter_boundaries["hit_upper"].any()
        ),
        "kappa_hit_upper_bound": bool(
            parameter_boundaries.set_index("parameter").loc["kappa", "hit_upper"]
        ),
        "kappa_profile_included": bool(include_kappa_profile),
        "kappa_profile_refit_remaining": bool(kappa_profile_refit_remaining),
        "feller_ratio": _coerce_real_scalar(
            constraint_diagnostics.at[0, "value"],
            label="feller_ratio",
        ),
        "feller_margin": _coerce_real_scalar(
            constraint_diagnostics.at[0, "margin"],
            label="feller_margin",
        ),
        "feller_satisfied": bool(constraint_diagnostics.at[0, "satisfied"]),
        **quote_policy_meta,
        "notes": notes,
    }

    arrays: dict[str, Any] = {
        "strike": quotes.strike,
        "expiry": quotes.expiry,
        "log_moneyness": quotes.log_moneyness,
        "market_price": quotes.mid,
        "model_price": model_prices,
        "market_iv": _market_iv(quotes),
        "model_iv": model_iv,
        "fitted_params": fitted_params.as_array(),
    }
    if true_params is not None:
        arrays["true_params"] = true_params.as_array()
    if resolved_seed is not None:
        arrays["seed_params"] = resolved_seed.as_array()

    return HestonCalibrationFitDiagnostics(
        meta=meta,
        tables={
            "residuals": residuals,
            "smile_fit": smile_fit,
            "iv_residual_grid": iv_residual_grid,
            "parameter_recovery": parameter_recovery,
            "constraint_diagnostics": constraint_diagnostics,
            "parameter_boundaries": parameter_boundaries,
            "multistart_parameter_dispersion": multistart_parameter_dispersion,
            "residual_buckets": residual_buckets,
            "quote_policy": quote_policy,
            "quote_policy_summary": quote_policy_summary,
            "multistart_runs": multistart_runs,
            "held_out_errors": held_out_errors,
            "objective_slices": objective_slices,
            "kappa_profile": kappa_profile,
        },
        arrays=arrays,
    )


run_heston_calibration_diagnostics = run_heston_calibration_fit_diagnostics


__all__ = [
    "run_heston_kappa_profile_diagnostics",
    "run_heston_calibration_diagnostics",
    "run_heston_calibration_fit_diagnostics",
]
