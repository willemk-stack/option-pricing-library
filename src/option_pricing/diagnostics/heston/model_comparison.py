"""Heston versus local-vol/eSSVI model comparison diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd

from ...models.heston.calibration.heston_types import (
    HestonMultistartResult,
    HestonQuoteSet,
)
from ...models.heston.fourier import HestonBackend
from ...models.heston.params import HestonParams
from ...numerics.grids import SpacingPolicy
from ...numerics.pde import AdvectionScheme
from ...numerics.pde.domain import Coord
from ...numerics.quadrature import QuadratureConfig
from ...pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from ...pricers.pde_pricer import local_vol_price_pde_european
from ...types import MarketData, OptionSpec, OptionType, PricingInputs
from ...typing import FloatArray
from ...vol.implied_vol_scalar import implied_vol_bs
from ...vol.local_vol_surface import LocalVolSurface
from ...vol.ssvi import (
    ESSVICalibrationConfig,
    ESSVIFitResult,
    ESSVINodalSurface,
    ESSVIProjectionConfig,
    ESSVIProjectionResult,
    calibrate_essvi,
    project_essvi_nodes,
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
_DIRECT_LOCAL_VOL_MODEL_NAME = "Direct local-vol PDE"
_LOCAL_VOL_PROXY_NOTE = (
    "NOTE: The eSSVI row reprices the calibrated implied surface directly and "
    "is retained as a supporting proxy; the direct local-vol PDE table is the "
    "Dupire/PDE repricing audit for the selected validation grid."
)
_MONEYNESS_BUCKET_NOTE = (
    "NOTE: Moneyness buckets use abs(log_moneyness) <= 0.03 for ATM, "
    "log_moneyness < -0.03 for downside wing/skew, and > 0.03 for upside wing."
)
_MATCHED_DIRECT_PDE_NOTE = (
    "NOTE: The full error summary keeps full-set Heston/eSSVI rows and the "
    "selected direct local-vol PDE audit rows together for continuity. Use "
    "direct_local_vol_pde_matched_error_summary for apples-to-apples Heston, "
    "eSSVI proxy, and direct PDE comparisons on the successful direct-PDE "
    "quote subset."
)
_ERROR_SUMMARY_COLUMNS = [
    "model",
    "bucket",
    "n_quotes",
    "price_rmse",
    "price_mae",
    "price_max_abs",
    "iv_rmse_bps",
    "iv_mae_bps",
    "iv_max_abs_bps",
]


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


def _default_projection_cfg() -> ESSVIProjectionConfig:
    return ESSVIProjectionConfig(
        validation_nt=21,
        validation_y_min=-0.60,
        validation_y_max=0.60,
        validation_ny=41,
        dupire_nt=15,
        dupire_y_min=-0.50,
        dupire_y_max=0.50,
        dupire_ny=31,
        strict_validation=False,
    )


def _default_local_vol_pde_solver_cfg() -> dict[str, Any]:
    return {
        "coord": Coord.LOG_S,
        "domain_cfg": BSDomainConfig(
            policy=BSDomainPolicy.LOG_NSIGMA,
            n_sigma=5.5,
            center="strike",
            spacing=SpacingPolicy.CLUSTERED,
            cluster_strength=2.5,
        ),
        "method": "rannacher",
        "advection": AdvectionScheme.CENTRAL,
        "sigma2_floor": 1.0e-12,
        "sigma2_cap": 4.0,
    }


def _serialize_solver_cfg(solver_cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in solver_cfg.items():
        if isinstance(value, BSDomainConfig):
            domain = asdict(value)
            domain["policy"] = str(value.policy.value)
            domain["spacing"] = str(value.spacing.value)
            out[str(key)] = domain
        elif isinstance(value, (Coord, AdvectionScheme)):
            out[str(key)] = str(value.value)
        else:
            out[str(key)] = value
    return out


def _select_direct_pde_quote_indices(
    quotes: HestonQuoteSet,
    *,
    max_quotes: int,
) -> np.ndarray:
    if int(max_quotes) <= 0 or quotes.n_quotes == 0:
        return np.array([], dtype=np.int64)

    expiries = np.sort(np.unique(np.asarray(quotes.expiry, dtype=np.float64)))
    if expiries.size <= 3:
        selected_expiries = expiries
    else:
        selected_expiries = expiries[
            np.unique(np.round(np.linspace(0, expiries.size - 1, 3)).astype(np.int64))
        ]

    targets = np.array([-0.10, 0.0, 0.10], dtype=np.float64)
    selected: list[int] = []
    log_mny = np.asarray(quotes.log_moneyness, dtype=np.float64)
    for expiry in selected_expiries:
        idx = np.flatnonzero(np.isclose(quotes.expiry, float(expiry)))
        used: set[int] = set()
        for target in targets:
            available = [int(i) for i in idx if int(i) not in used]
            if not available:
                continue
            best = min(
                available,
                key=lambda i: (abs(float(log_mny[i]) - float(target)), i),
            )
            selected.append(best)
            used.add(best)
            if len(selected) >= int(max_quotes):
                return np.array(sorted(set(selected)), dtype=np.int64)

    return np.array(sorted(set(selected))[: int(max_quotes)], dtype=np.int64)


def _quote_market(quotes: HestonQuoteSet, tau: float) -> MarketData:
    return MarketData(
        spot=float(quotes.ctx.spot),
        rate=float(quotes.ctx.r_avg(float(tau))),
        dividend_yield=float(quotes.ctx.q_avg(float(tau))),
    )


def _local_vol_surface_from_essvi(
    fit: ESSVIFitResult,
    quotes: HestonQuoteSet,
    projection_cfg: ESSVIProjectionConfig | None,
) -> tuple[LocalVolSurface | None, ESSVIProjectionResult, str]:
    projection = project_essvi_nodes(
        fit.nodes,
        cfg=_default_projection_cfg() if projection_cfg is None else projection_cfg,
    )
    if projection.surface is None:
        return None, projection, "essvi_projection_failed_no_local_vol_surface"
    return (
        LocalVolSurface.from_implied(
            projection.surface,
            forward=quotes.ctx.fwd,
            discount=quotes.ctx.df,
        ),
        projection,
        "essvi_smoothed_projection",
    )


def _direct_local_vol_pde_table(
    *,
    quotes: HestonQuoteSet,
    fit: ESSVIFitResult,
    heston_prices: FloatArray,
    heston_iv: FloatArray,
    held_out_mask: np.ndarray,
    mask_provided: bool,
    max_quotes: int,
    Nx: int,
    Nt: int,
    solver_cfg: dict[str, Any] | None,
    projection_cfg: ESSVIProjectionConfig | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = _select_direct_pde_quote_indices(quotes, max_quotes=max_quotes)
    resolved_solver_cfg = (
        _default_local_vol_pde_solver_cfg() if solver_cfg is None else dict(solver_cfg)
    )
    lv, projection, surface_source = _local_vol_surface_from_essvi(
        fit,
        quotes,
        projection_cfg,
    )
    if lv is None:
        return (
            pd.DataFrame(
                columns=[
                    "model",
                    "quote_index",
                    "expiry",
                    "strike",
                    "log_moneyness",
                    "moneyness_bucket",
                    "is_call",
                    "sample",
                    "is_held_out",
                    "target_iv",
                    "target_price",
                    "heston_price",
                    "heston_iv",
                    "heston_price_residual",
                    "heston_iv_residual_bps",
                    "local_vol_pde_price",
                    "local_vol_pde_iv",
                    "local_vol_pde_price_residual",
                    "local_vol_pde_iv_residual_bps",
                    "pde_runtime_ms",
                    "pde_status",
                    "pde_error",
                    "Nx",
                    "Nt",
                    "surface_source",
                ]
            ),
            {
                "direct_local_vol_pde_repricing": True,
                "direct_local_vol_pde_quote_count": 0,
                "direct_local_vol_pde_success_count": 0,
                "direct_local_vol_pde_surface_source": surface_source,
                "direct_local_vol_pde_projection_success": bool(projection.success),
                "direct_local_vol_pde_projection_message": str(projection.diag.message),
                "direct_local_vol_pde_total_runtime_ms": 0.0,
                "direct_local_vol_pde_solver": {
                    "Nx": int(Nx),
                    "Nt": int(Nt),
                    **_serialize_solver_cfg(resolved_solver_cfg),
                },
            },
        )
    market_iv = _market_iv(quotes)
    sample = np.where(held_out_mask, "held_out", "train")
    if not mask_provided:
        sample = np.full(quotes.n_quotes, "fit", dtype=object)

    rows: list[dict[str, Any]] = []
    total_t0 = perf_counter()
    for quote_index in selected:
        i = int(quote_index)
        tau = float(quotes.expiry[i])
        strike = float(quotes.strike[i])
        kind = OptionType.CALL if bool(quotes.is_call[i]) else OptionType.PUT
        market = _quote_market(quotes, tau)
        target_iv = float(market_iv[i])
        sigma_seed = target_iv if np.isfinite(target_iv) and target_iv > 0.0 else 0.20
        spec = OptionSpec(kind=kind, strike=strike, expiry=tau)
        p = PricingInputs(spec=spec, market=market, sigma=max(float(sigma_seed), 1e-4))
        row: dict[str, Any] = {
            "model": _DIRECT_LOCAL_VOL_MODEL_NAME,
            "quote_index": i,
            "expiry": tau,
            "strike": strike,
            "log_moneyness": float(quotes.log_moneyness[i]),
            "moneyness_bucket": str(
                _moneyness_bucket(np.array([quotes.log_moneyness[i]]))[0]
            ),
            "is_call": bool(quotes.is_call[i]),
            "sample": str(sample[i]),
            "is_held_out": bool(held_out_mask[i]),
            "target_iv": target_iv,
            "target_price": float(quotes.mid[i]),
            "heston_price": float(np.asarray(heston_prices)[i]),
            "heston_iv": float(np.asarray(heston_iv)[i]),
            "heston_price_residual": float(
                np.asarray(heston_prices)[i] - quotes.mid[i]
            ),
            "heston_iv_residual_bps": float(
                (np.asarray(heston_iv)[i] - target_iv) * 1.0e4
            ),
            "local_vol_pde_price": np.nan,
            "local_vol_pde_iv": np.nan,
            "local_vol_pde_price_residual": np.nan,
            "local_vol_pde_iv_residual_bps": np.nan,
            "pde_runtime_ms": np.nan,
            "pde_status": "not_run",
            "pde_error": "",
            "Nx": int(Nx),
            "Nt": int(Nt),
            "surface_source": surface_source,
        }
        t0 = perf_counter()
        try:
            pde_price_raw = local_vol_price_pde_european(
                p,
                lv=lv,
                Nx=int(Nx),
                Nt=int(Nt),
                return_solution=False,
                **resolved_solver_cfg,
            )
            pde_price = float(cast(float, pde_price_raw))
            pde_iv = float(
                implied_vol_bs(
                    pde_price,
                    spec,
                    market,
                    sigma0=max(float(sigma_seed), 1.0e-4),
                )
            )
            row.update(
                {
                    "local_vol_pde_price": pde_price,
                    "local_vol_pde_iv": pde_iv,
                    "local_vol_pde_price_residual": pde_price - float(quotes.mid[i]),
                    "local_vol_pde_iv_residual_bps": (pde_iv - target_iv) * 1.0e4,
                    "pde_status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["pde_status"] = "error"
            row["pde_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["pde_runtime_ms"] = 1.0e3 * (perf_counter() - t0)
        rows.append(row)

    table = pd.DataFrame(rows)
    ok_count = int((table["pde_status"] == "ok").sum()) if not table.empty else 0
    meta = {
        "direct_local_vol_pde_repricing": True,
        "direct_local_vol_pde_quote_count": int(len(table)),
        "direct_local_vol_pde_success_count": ok_count,
        "direct_local_vol_pde_surface_source": surface_source,
        "direct_local_vol_pde_projection_success": bool(projection.success),
        "direct_local_vol_pde_projection_message": str(projection.diag.message),
        "direct_local_vol_pde_total_runtime_ms": 1.0e3 * (perf_counter() - total_t0),
        "direct_local_vol_pde_solver": {
            "Nx": int(Nx),
            "Nt": int(Nt),
            **_serialize_solver_cfg(resolved_solver_cfg),
        },
    }
    return table, meta


def _direct_pde_fit_errors(
    direct_table: pd.DataFrame,
) -> pd.DataFrame:
    if direct_table.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "quote_index",
                "expiry",
                "strike",
                "log_moneyness",
                "moneyness_bucket",
                "is_call",
                "market_iv",
                "model_iv",
                "iv_residual_bps",
                "market_price",
                "model_price",
                "price_residual",
                "is_held_out",
                "sample",
            ]
        )
    return pd.DataFrame(
        {
            "model": direct_table["model"],
            "quote_index": direct_table["quote_index"],
            "expiry": direct_table["expiry"],
            "strike": direct_table["strike"],
            "log_moneyness": direct_table["log_moneyness"],
            "moneyness_bucket": direct_table["moneyness_bucket"],
            "is_call": direct_table["is_call"],
            "market_iv": direct_table["target_iv"],
            "model_iv": direct_table["local_vol_pde_iv"],
            "iv_residual_bps": direct_table["local_vol_pde_iv_residual_bps"],
            "market_price": direct_table["target_price"],
            "model_price": direct_table["local_vol_pde_price"],
            "price_residual": direct_table["local_vol_pde_price_residual"],
            "is_held_out": direct_table["is_held_out"],
            "sample": direct_table["sample"],
        }
    )


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
    return pd.DataFrame(rows, columns=_ERROR_SUMMARY_COLUMNS)


def _matched_direct_pde_error_summary(
    fit_errors: pd.DataFrame,
    direct_table: pd.DataFrame,
) -> pd.DataFrame:
    if direct_table.empty:
        table = pd.DataFrame(columns=_ERROR_SUMMARY_COLUMNS)
        table.attrs["notes"] = [
            "No direct local-vol PDE rows were emitted; matched subset is empty."
        ]
        return table

    ok_rows = direct_table
    if "pde_status" in direct_table.columns:
        ok_rows = direct_table.loc[direct_table["pde_status"].astype(str) == "ok"]
    selected_quote_indices = sorted(
        {int(value) for value in ok_rows["quote_index"].dropna().to_numpy()}
    )
    if not selected_quote_indices:
        table = pd.DataFrame(columns=_ERROR_SUMMARY_COLUMNS)
        table.attrs["notes"] = [
            "Direct local-vol PDE rows were emitted, but none succeeded; "
            "matched subset is empty."
        ]
        return table

    model_order = [
        _HESTON_MODEL_NAME,
        _LOCAL_VOL_MODEL_NAME,
        _DIRECT_LOCAL_VOL_MODEL_NAME,
    ]
    matched = fit_errors.loc[
        fit_errors["quote_index"].isin(selected_quote_indices)
        & fit_errors["model"].isin(model_order)
    ].copy()
    summary = _error_summary(matched)
    if summary.empty:
        summary.attrs["notes"] = [
            "Matched direct local-vol PDE subset produced no summary rows."
        ]
        return summary

    count_matrix = summary.pivot(
        index="bucket",
        columns="model",
        values="n_quotes",
    )
    missing_models = [model for model in model_order if model not in count_matrix]
    if missing_models:
        raise ValueError(
            "Matched direct local-vol PDE subset is missing model rows for "
            f"{', '.join(missing_models)}."
        )
    count_matrix = count_matrix.loc[:, model_order]
    inconsistent = count_matrix.loc[
        count_matrix.isna().any(axis=1)
        | (count_matrix.nunique(axis=1, dropna=False) != 1)
    ]
    if not inconsistent.empty:
        raise ValueError(
            "Matched direct local-vol PDE subset has inconsistent per-bucket "
            f"quote counts: {inconsistent.to_dict(orient='index')}"
        )
    summary.attrs["notes"] = [_MATCHED_DIRECT_PDE_NOTE]
    return summary


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
    model_names = [str(name) for name in fit_errors["model"].drop_duplicates()]
    for model_name in model_names:
        model_group = fit_errors.loc[fit_errors["model"] == model_name]
        for sample_name in ("train", "held_out"):
            group = model_group.loc[model_group["sample"] == sample_name]
            row = _summary_row(
                model_name=model_name,
                bucket=sample_name,
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
            {
                "model": _DIRECT_LOCAL_VOL_MODEL_NAME,
                "fit_quality_note": (
                    "Direct PDE repricing audits a small validation grid against "
                    "the same target quotes rather than relying only on implied-"
                    "surface proxy prices."
                ),
                "interpretability_note": (
                    "Dupire local vol is implied by the calibrated vanilla "
                    "surface; it is a surface model rather than a variance-factor "
                    "model."
                ),
                "smoothness_extrapolation_note": (
                    "Numerics depend on the eSSVI projection, local-vol "
                    "regularity, boundaries, and grid settings."
                ),
                "dynamic_plausibility_note": (
                    "Local volatility matches vanilla marginals under its surface "
                    "assumptions but does not model independent variance risk."
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
    essvi_projection_cfg: ESSVIProjectionConfig | None = None,
    sqrt_weights: FloatArray | None = None,
    run_direct_local_vol_pde: bool = True,
    local_vol_pde_max_quotes: int = 9,
    local_vol_pde_Nx: int = 81,
    local_vol_pde_Nt: int = 121,
    local_vol_pde_solver_cfg: dict[str, Any] | None = None,
) -> HestonModelComparisonDiagnostics:
    """Compare Heston against eSSVI and direct local-vol PDE diagnostics.

    The eSSVI proxy row reprices the calibrated implied surface directly. When
    ``run_direct_local_vol_pde`` is true, the diagnostic also projects eSSVI
    nodes into a Dupire-oriented surface and reprices a small deterministic
    validation grid with the local-vol PDE pricer.
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

    fit_error_parts = [
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
    ]

    direct_local_vol_pde = pd.DataFrame()
    direct_meta: dict[str, Any] = {
        "direct_local_vol_pde_repricing": False,
        "direct_local_vol_pde_quote_count": 0,
        "direct_local_vol_pde_success_count": 0,
    }
    if run_direct_local_vol_pde:
        direct_local_vol_pde, direct_meta = _direct_local_vol_pde_table(
            quotes=quotes,
            fit=resolved_local_fit,
            heston_prices=heston_prices,
            heston_iv=heston_iv,
            held_out_mask=mask,
            mask_provided=mask_provided,
            max_quotes=int(local_vol_pde_max_quotes),
            Nx=int(local_vol_pde_Nx),
            Nt=int(local_vol_pde_Nt),
            solver_cfg=local_vol_pde_solver_cfg,
            projection_cfg=essvi_projection_cfg,
        )
        fit_error_parts.append(_direct_pde_fit_errors(direct_local_vol_pde))

    fit_errors = pd.concat(fit_error_parts, ignore_index=True)
    fit_errors.attrs["notes"] = [
        _LOCAL_VOL_PROXY_NOTE,
        _MONEYNESS_BUCKET_NOTE,
        _MATCHED_DIRECT_PDE_NOTE,
    ]

    direct_local_vol_pde_summary = pd.DataFrame(
        [
            {
                "model": _DIRECT_LOCAL_VOL_MODEL_NAME,
                "n_quotes": int(direct_meta["direct_local_vol_pde_quote_count"]),
                "n_success": int(direct_meta["direct_local_vol_pde_success_count"]),
                "Nx": int(local_vol_pde_Nx),
                "Nt": int(local_vol_pde_Nt),
                "surface_source": str(
                    direct_meta.get("direct_local_vol_pde_surface_source", "not_run")
                ),
                "mean_abs_price_residual": (
                    float(
                        direct_local_vol_pde["local_vol_pde_price_residual"]
                        .abs()
                        .mean()
                    )
                    if not direct_local_vol_pde.empty
                    else np.nan
                ),
                "mean_abs_iv_residual_bps": (
                    float(
                        direct_local_vol_pde["local_vol_pde_iv_residual_bps"]
                        .abs()
                        .mean()
                    )
                    if not direct_local_vol_pde.empty
                    else np.nan
                ),
            }
        ]
    )

    error_summary = _error_summary(fit_errors)
    direct_local_vol_pde_matched_error_summary = _matched_direct_pde_error_summary(
        fit_errors,
        direct_local_vol_pde,
    )
    tradeoff_summary = _tradeoff_summary()
    held_out_comparison = _held_out_comparison(
        fit_errors,
        mask_provided=mask_provided,
    )

    train_quote_count = (
        int(np.count_nonzero(~mask)) if mask_provided else int(quotes.n_quotes)
    )
    held_out_quote_count = int(np.count_nonzero(mask)) if mask_provided else 0

    models = [_HESTON_MODEL_NAME, _LOCAL_VOL_MODEL_NAME]
    if bool(direct_meta["direct_local_vol_pde_repricing"]):
        models.append(_DIRECT_LOCAL_VOL_MODEL_NAME)

    notes = [
        _LOCAL_VOL_PROXY_NOTE,
        _MONEYNESS_BUCKET_NOTE,
        _MATCHED_DIRECT_PDE_NOTE,
        "LIMITATION: Comparison conclusions depend on the target quotes, fit "
        "partition, eSSVI projection, local-vol PDE grid, and boundary policy.",
    ]
    meta: dict[str, Any] = {
        "diagnostic": "heston_vs_local_vol_comparison",
        "models": models,
        "quote_count": int(quotes.n_quotes),
        "train_quote_count": train_quote_count,
        "held_out_quote_count": held_out_quote_count,
        "expiry_count": int(np.unique(quotes.expiry).size),
        "log_moneyness_count": int(np.unique(quotes.log_moneyness).size),
        "held_out_mask_provided": mask_provided,
        "sample_labels": ["train", "held_out"] if mask_provided else ["fit"],
        "comparison_target": "same HestonQuoteSet quotes repriced by both models",
        "comparison_fixture_label": (
            None
            if quotes.metadata is None
            else quotes.metadata.get("fixture_label", None)
        ),
        "local_vol_proxy_kind": "essvi_nodal_implied_surface",
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
    meta.update(direct_meta)
    arrays: dict[str, Any] = {
        "heston_params": heston_params.as_array(),
        "heston_model_price": heston_prices,
        "heston_model_iv": heston_iv,
        "local_vol_model_price": local_prices,
        "local_vol_model_iv": local_iv,
        "market_iv": _market_iv(quotes),
    }
    if not direct_local_vol_pde.empty:
        arrays["direct_local_vol_pde_quote_index"] = direct_local_vol_pde[
            "quote_index"
        ].to_numpy(dtype=np.int64)
        arrays["direct_local_vol_pde_price"] = direct_local_vol_pde[
            "local_vol_pde_price"
        ].to_numpy(dtype=np.float64)
        arrays["direct_local_vol_pde_iv"] = direct_local_vol_pde[
            "local_vol_pde_iv"
        ].to_numpy(dtype=np.float64)

    return HestonModelComparisonDiagnostics(
        meta=meta,
        tables={
            "fit_errors": fit_errors,
            "error_summary": error_summary,
            "tradeoff_summary": tradeoff_summary,
            "held_out_comparison": held_out_comparison,
            "direct_local_vol_pde": direct_local_vol_pde,
            "direct_local_vol_pde_summary": direct_local_vol_pde_summary,
            "direct_local_vol_pde_matched_error_summary": (
                direct_local_vol_pde_matched_error_summary
            ),
        },
        arrays=arrays,
    )


__all__ = ["run_heston_vs_local_vol_comparison"]
