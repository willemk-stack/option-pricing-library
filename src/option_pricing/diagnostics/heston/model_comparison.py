"""Heston versus local-vol/eSSVI model comparison diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ...models.heston.calibration.heston_types import (
    HestonMultistartResult,
    HestonQuoteSet,
)
from ...models.heston.fourier import HestonBackend
from ...models.heston.params import HestonParams
from ...numerics.quadrature import QuadratureConfig
from ...types import OptionType
from ...typing import FloatArray
from ...vol.ssvi import (
    ESSVICalibrationConfig,
    ESSVIFitResult,
    ESSVINodalSurface,
    calibrate_essvi,
)
from ._provenance import backend_config_meta
from .calibration_fit import (
    _implied_vols_from_prices,
    _market_iv,
    _metric_values,
    _price_heston_quotes,
    _resolve_mask,
    _resolved_fit,
)
from .models import HestonModelComparisonDiagnostics

_HESTON_MODEL_NAME = "Heston"
_LOCAL_VOL_MODEL_NAME = "ESSVI local-vol proxy"


def _moneyness_bucket(log_moneyness: FloatArray) -> np.ndarray:
    x = np.asarray(log_moneyness, dtype=np.float64)
    bucket = np.full(x.shape, "atm", dtype=object)
    bucket[x < -0.03] = "downside_wing"
    bucket[x > 0.03] = "upside_wing"
    return bucket


def _resolve_sqrt_weights(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None,
) -> FloatArray | None:
    if sqrt_weights is not None:
        weights = np.asarray(sqrt_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (quotes.n_quotes,):
            raise ValueError(f"sqrt_weights must have shape ({quotes.n_quotes},).")
        if not np.all(np.isfinite(weights)):
            raise ValueError("sqrt_weights must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("sqrt_weights must be nonnegative.")
        return weights
    if quotes.sqrt_weights is None:
        return None
    return np.asarray(quotes.sqrt_weights, dtype=np.float64)


def _fit_essvi_proxy(
    *,
    quotes: HestonQuoteSet,
    held_out_mask: np.ndarray,
    mask_provided: bool,
    sqrt_weights: FloatArray | None,
    cfg: ESSVICalibrationConfig | None,
) -> ESSVIFitResult:
    fit_mask = ~held_out_mask if mask_provided else np.ones(quotes.n_quotes, dtype=bool)
    if not np.any(fit_mask):
        raise ValueError("At least one training quote is required for eSSVI fit.")
    weights = None if sqrt_weights is None else sqrt_weights[fit_mask]
    return calibrate_essvi(
        y=np.asarray(quotes.log_moneyness[fit_mask], dtype=np.float64),
        T=np.asarray(quotes.expiry[fit_mask], dtype=np.float64),
        price_mkt=np.asarray(quotes.mid[fit_mask], dtype=np.float64),
        market=quotes.ctx,
        sqrt_weights=weights,
        is_call=np.asarray(quotes.is_call[fit_mask], dtype=np.bool_),
        cfg=cfg,
    )


def _price_essvi_quotes(
    quotes: HestonQuoteSet,
    fit: ESSVIFitResult,
) -> FloatArray:
    surface = ESSVINodalSurface(fit.nodes)
    call_price = np.asarray(
        surface.price(
            kind=OptionType.CALL,
            strike=quotes.strike,
            forward=quotes.forward,
            df=quotes.discount,
            T=quotes.expiry,
        ),
        dtype=np.float64,
    )
    put_price = np.asarray(
        call_price - quotes.discount * (quotes.forward - quotes.strike),
        dtype=np.float64,
    )
    return np.asarray(np.where(quotes.is_call, call_price, put_price), dtype=np.float64)


def _fit_error_table(
    *,
    quotes: HestonQuoteSet,
    model_name: str,
    model_prices: FloatArray,
    model_iv: FloatArray,
    held_out_mask: np.ndarray,
    mask_provided: bool,
) -> pd.DataFrame:
    market_iv = _market_iv(quotes)
    log_moneyness = np.asarray(quotes.log_moneyness, dtype=np.float64)
    iv_residual_bps = np.asarray((model_iv - market_iv) * 1.0e4, dtype=np.float64)
    price_residual = np.asarray(model_prices - quotes.mid, dtype=np.float64)
    sample = np.where(held_out_mask, "held_out", "train")
    if not mask_provided:
        sample = np.full(quotes.n_quotes, "fit", dtype=object)

    return pd.DataFrame(
        {
            "model": np.full(quotes.n_quotes, model_name, dtype=object),
            "quote_index": np.arange(quotes.n_quotes, dtype=np.int64),
            "expiry": np.asarray(quotes.expiry, dtype=np.float64),
            "strike": np.asarray(quotes.strike, dtype=np.float64),
            "log_moneyness": log_moneyness,
            "moneyness_bucket": _moneyness_bucket(log_moneyness),
            "is_call": np.asarray(quotes.is_call, dtype=np.bool_),
            "market_iv": market_iv,
            "model_iv": np.asarray(model_iv, dtype=np.float64),
            "iv_residual_bps": iv_residual_bps,
            "market_price": np.asarray(quotes.mid, dtype=np.float64),
            "model_price": np.asarray(model_prices, dtype=np.float64),
            "price_residual": price_residual,
            "is_held_out": np.asarray(held_out_mask, dtype=np.bool_),
            "sample": sample,
        }
    )


def _summary_row(
    *,
    model_name: str,
    bucket: str,
    group: pd.DataFrame,
) -> dict[str, Any]:
    price_rmse, price_mae, price_max_abs = _metric_values(
        group["price_residual"].to_numpy(dtype=np.float64, copy=False)
    )
    iv_rmse, iv_mae, iv_max_abs = _metric_values(
        group["iv_residual_bps"].to_numpy(dtype=np.float64, copy=False)
    )
    return {
        "model": model_name,
        "bucket": bucket,
        "n_quotes": int(len(group)),
        "price_rmse": price_rmse,
        "price_mae": price_mae,
        "price_max_abs": price_max_abs,
        "iv_rmse_bps": iv_rmse,
        "iv_mae_bps": iv_mae,
        "iv_max_abs_bps": iv_max_abs,
    }


def _error_summary(fit_errors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model_group in fit_errors.groupby("model", sort=False):
        rows.append(
            _summary_row(
                model_name=str(model_name),
                bucket="all",
                group=model_group,
            )
        )
        for bucket in ("atm", "downside_wing", "upside_wing"):
            bucket_group = model_group.loc[model_group["moneyness_bucket"] == bucket]
            if bucket_group.empty:
                continue
            rows.append(
                _summary_row(
                    model_name=str(model_name),
                    bucket=bucket,
                    group=bucket_group,
                )
            )
    return pd.DataFrame(rows)


def _held_out_comparison(
    fit_errors: pd.DataFrame,
    *,
    mask_provided: bool,
) -> pd.DataFrame:
    columns = [
        "model",
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
            "No held_out_mask was supplied; held-out comparison was not run."
        ]
        return table

    rows: list[dict[str, Any]] = []
    for (model_name, sample), group in fit_errors.groupby(
        ["model", "sample"], sort=False
    ):
        row = _summary_row(
            model_name=str(model_name),
            bucket=str(sample),
            group=group,
        )
        rows.append(
            {
                "model": row["model"],
                "sample": row["bucket"],
                "n_quotes": row["n_quotes"],
                "price_rmse": row["price_rmse"],
                "price_mae": row["price_mae"],
                "price_max_abs": row["price_max_abs"],
                "iv_rmse_bps": row["iv_rmse_bps"],
                "iv_mae_bps": row["iv_mae_bps"],
                "iv_max_abs_bps": row["iv_max_abs_bps"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _tradeoff_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": _HESTON_MODEL_NAME,
                "fit_quality_note": (
                    "Five-parameter stochastic-volatility fit can be too rigid "
                    "for some vanilla smiles."
                ),
                "interpretability_note": (
                    "Parameters map to mean reversion, long-run variance, "
                    "vol-of-vol, correlation, and initial variance."
                ),
                "smoothness_extrapolation_note": (
                    "Parametric surface is smooth but extrapolation is model-driven."
                ),
                "dynamic_plausibility_note": (
                    "Variance dynamics are explicit and can support path-dependent "
                    "or forward-looking analysis."
                ),
            },
            {
                "model": _LOCAL_VOL_MODEL_NAME,
                "fit_quality_note": (
                    "eSSVI/local-vol-facing proxy can fit vanilla implied-vol "
                    "targets flexibly at the calibrated nodes."
                ),
                "interpretability_note": (
                    "Surface parameters describe smile geometry rather than a "
                    "stochastic variance process."
                ),
                "smoothness_extrapolation_note": (
                    "Nodal eSSVI gives a smooth implied surface, but local-vol "
                    "extrapolation and Dupire stability need separate review."
                ),
                "dynamic_plausibility_note": (
                    "Local volatility matches vanilla marginals under its surface "
                    "assumptions, but it does not model independent variance risk."
                ),
            },
        ]
    )


def run_heston_vs_local_vol_comparison(
    *,
    quotes: HestonQuoteSet,
    heston_fit: HestonParams | HestonMultistartResult,
    local_vol_fit: ESSVIFitResult | None = None,
    held_out_mask: Sequence[bool] | np.ndarray | None = None,
    heston_backend: HestonBackend | None = None,
    heston_quad_cfg: QuadratureConfig | None = None,
    essvi_cfg: ESSVICalibrationConfig | None = None,
    sqrt_weights: FloatArray | None = None,
) -> HestonModelComparisonDiagnostics:
    """Compare Heston against the repo-native eSSVI/local-vol-facing path.

    The local-vol side uses eSSVI calibration and nodal implied-price repricing.
    It does not run a Dupire PDE repricer.

    REVIEW: This is a vanilla-surface comparison proxy. Direct local-vol PDE
    repricing can be materially more expensive and should be audited separately
    when the capstone target requires it.
    """
    heston_params, heston_multistart = _resolved_fit(heston_fit)
    resolved_backend = (
        heston_multistart.backend
        if heston_backend is None and heston_multistart is not None
        else ("gauss_legendre" if heston_backend is None else heston_backend)
    )
    mask, mask_provided = _resolve_mask(held_out_mask, n_quotes=quotes.n_quotes)
    resolved_weights = _resolve_sqrt_weights(quotes, sqrt_weights)
    resolved_local_fit = (
        _fit_essvi_proxy(
            quotes=quotes,
            held_out_mask=mask,
            mask_provided=mask_provided,
            sqrt_weights=resolved_weights,
            cfg=essvi_cfg,
        )
        if local_vol_fit is None
        else local_vol_fit
    )

    heston_prices = _price_heston_quotes(
        quotes,
        heston_params,
        backend=resolved_backend,
        quad_cfg=heston_quad_cfg,
    )
    heston_iv = _implied_vols_from_prices(quotes, heston_prices)
    local_prices = _price_essvi_quotes(quotes, resolved_local_fit)
    local_iv = _implied_vols_from_prices(quotes, local_prices)

    fit_errors = pd.concat(
        [
            _fit_error_table(
                quotes=quotes,
                model_name=_HESTON_MODEL_NAME,
                model_prices=heston_prices,
                model_iv=heston_iv,
                held_out_mask=mask,
                mask_provided=mask_provided,
            ),
            _fit_error_table(
                quotes=quotes,
                model_name=_LOCAL_VOL_MODEL_NAME,
                model_prices=local_prices,
                model_iv=local_iv,
                held_out_mask=mask,
                mask_provided=mask_provided,
            ),
        ],
        ignore_index=True,
    )
    fit_errors.attrs["notes"] = [
        "REVIEW: Moneyness buckets use abs(log_moneyness) <= 0.03 for ATM, "
        "log_moneyness < -0.03 for downside wing/skew, and > 0.03 for upside wing."
    ]

    error_summary = _error_summary(fit_errors)
    tradeoff_summary = _tradeoff_summary()
    held_out_comparison = _held_out_comparison(
        fit_errors,
        mask_provided=mask_provided,
    )

    notes = [
        "REVIEW: Local-vol comparison uses the repo-native eSSVI nodal implied "
        "surface as a local-vol-facing proxy; direct Dupire/PDE repricing is not run.",
        "REVIEW: Moneyness bucket thresholds are simple capstone diagnostics, "
        "not universal smile-region definitions.",
        "REVIEW: Comparison conclusions depend on the target quotes, fit "
        "partition, and chosen local-vol proxy.",
    ]
    meta: dict[str, Any] = {
        "diagnostic": "heston_vs_local_vol_comparison",
        "models": [_HESTON_MODEL_NAME, _LOCAL_VOL_MODEL_NAME],
        "quote_count": int(quotes.n_quotes),
        "expiry_count": int(np.unique(quotes.expiry).size),
        "log_moneyness_count": int(np.unique(quotes.log_moneyness).size),
        "held_out_mask_provided": mask_provided,
        "local_vol_proxy": "ESSVINodalSurface repricing from calibrate_essvi",
        "local_vol_fit_source": "computed" if local_vol_fit is None else "provided",
        "heston_backend_config": backend_config_meta(
            backend=resolved_backend,
            quad_cfg=heston_quad_cfg,
            rule=None,
        ),
        "essvi_success": bool(resolved_local_fit.diag.success),
        "essvi_nfev": int(resolved_local_fit.diag.nfev),
        "essvi_cost": float(resolved_local_fit.diag.cost),
        "notes": notes,
    }
    arrays: dict[str, Any] = {
        "heston_params": heston_params.as_array(),
        "heston_model_price": heston_prices,
        "heston_model_iv": heston_iv,
        "local_vol_model_price": local_prices,
        "local_vol_model_iv": local_iv,
        "market_iv": _market_iv(quotes),
    }

    return HestonModelComparisonDiagnostics(
        meta=meta,
        tables={
            "fit_errors": fit_errors,
            "error_summary": error_summary,
            "tradeoff_summary": tradeoff_summary,
            "held_out_comparison": held_out_comparison,
        },
        arrays=arrays,
    )


__all__ = ["run_heston_vs_local_vol_comparison"]
