"""Benchmark diagnostics for Heston analytic-calibration Jacobians."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, least_squares

from ...models.black_scholes.bs import (
    black76_call_price_vec,
    black76_call_price_vega_vec,
)
from ...models.heston.calibration.heston_types import HestonQuoteSet
from ...models.heston.calibration.objective import HestonObjective
from ...models.heston.params import HESTON_PARAM_NAMES, HestonParams
from ...numerics.quadrature import QuadratureConfig
from ...pricers.heston import heston_price_from_ctx
from ...types import MarketData, OptionType, PricingContext
from ...typing import FloatArray
from ...vol.implied_vol_slice import implied_vol_black76_slice
from ._provenance import backend_config_meta
from .contracts import HestonDiagnosticsBackend
from .models import HestonCalibrationBenchmarkDiagnostics

type OptimizerMethod = Literal["trf", "dogbox", "lm"]
type OptimizerLoss = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]
type CalibrationBenchmarkMode = Literal["finite_difference", "analytic"]


@dataclass(frozen=True, slots=True)
class _CalibrationRunResult:
    mode: CalibrationBenchmarkMode
    use_analytic_jac: bool
    repeat_index: int
    warmup: bool
    runtime_ms: float
    result: OptimizeResult
    fitted_params: HestonParams
    residual: FloatArray


def _default_true_params() -> HestonParams:
    # REVIEW: Smoke-benchmark synthetic truth; not a production calibration scenario.
    return HestonParams(kappa=1.5, vbar=0.04, eta=0.45, rho=-0.55, v=0.045)


def _default_seed_params() -> HestonParams:
    # REVIEW: Smoke-benchmark seed chosen to require optimization without stress testing.
    return HestonParams(kappa=1.1, vbar=0.035, eta=0.35, rho=-0.35, v=0.04)


def _default_quad_cfg() -> QuadratureConfig:
    # REVIEW: Smoke-benchmark quadrature keeps CI/runtime costs low; it is not a
    # production accuracy policy.
    return QuadratureConfig(u_max=50.0, n_panels=6, nodes_per_panel=6)


def _default_expiries() -> FloatArray:
    # REVIEW: Synthetic quote grid is intentionally small for smoke diagnostics.
    return np.array([0.5, 1.0], dtype=np.float64)


def _default_log_moneyness() -> FloatArray:
    # REVIEW: Synthetic quote grid is intentionally small for smoke diagnostics.
    return np.array([-0.08, 0.0, 0.08], dtype=np.float64)


def _to_ctx(market: MarketData | PricingContext | None) -> PricingContext:
    if market is None:
        market = MarketData(spot=100.0, rate=0.015, dividend_yield=0.005)
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def _coerce_positive_vector(
    values: Sequence[float] | np.ndarray | None,
    *,
    default: FloatArray,
    label: str,
) -> FloatArray:
    if values is None:
        arr = default.copy()
    else:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{label} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite.")
    if np.any(arr <= 0.0):
        raise ValueError(f"{label} values must be positive.")
    return np.asarray(arr, dtype=np.float64)


def _coerce_log_moneyness(
    values: Sequence[float] | np.ndarray | None,
) -> FloatArray:
    if values is None:
        arr = _default_log_moneyness()
    else:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("log_moneyness must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("log_moneyness values must be finite.")
    return np.asarray(arr, dtype=np.float64)


def _params_to_dict(params: HestonParams) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(HESTON_PARAM_NAMES, params.as_array(), strict=True)
    }


def _x_scale_for_optimizer(x_scale: str | np.ndarray | None) -> str | np.ndarray | None:
    if x_scale is None or isinstance(x_scale, str):
        return x_scale
    return np.asarray(x_scale, dtype=np.float64).copy()


def _x_scale_for_meta(x_scale: str | np.ndarray | None) -> Any:
    if x_scale is None or isinstance(x_scale, str):
        return x_scale
    return np.asarray(x_scale, dtype=np.float64).tolist()


def _build_synthetic_heston_quotes(
    *,
    market: MarketData | PricingContext | None,
    true_params: HestonParams,
    expiries: Sequence[float] | np.ndarray | None,
    log_moneyness: Sequence[float] | np.ndarray | None,
    backend: HestonDiagnosticsBackend,
    quad_cfg: QuadratureConfig | None,
    random_seed: int,
    noise_vol_bps: float,
) -> HestonQuoteSet:
    ctx = _to_ctx(market)
    expiry_grid = _coerce_positive_vector(
        expiries,
        default=_default_expiries(),
        label="expiries",
    )
    log_moneyness_grid = _coerce_log_moneyness(log_moneyness)

    strikes: list[float] = []
    expiry_values: list[float] = []
    mid_values: list[float] = []
    iv_values: list[float] = []
    vega_values: list[float] = []
    labels: list[str] = []

    rng = np.random.default_rng(int(random_seed))
    for tau in expiry_grid:
        forward = float(ctx.fwd(float(tau)))
        df = float(ctx.df(float(tau)))
        strike_slice = np.asarray(
            forward * np.exp(log_moneyness_grid),
            dtype=np.float64,
        )
        clean_prices = np.asarray(
            heston_price_from_ctx(
                kind=OptionType.CALL,
                strike=strike_slice,
                tau=float(tau),
                ctx=ctx,
                params=true_params,
                backend=backend,
                quad_cfg=quad_cfg,
            ),
            dtype=np.float64,
        )
        iv_slice = np.asarray(
            implied_vol_black76_slice(
                forward=forward,
                strikes=strike_slice,
                tau=float(tau),
                df=df,
                prices=clean_prices,
                is_call=True,
                initial_sigma=0.2,
            ),
            dtype=np.float64,
        )
        if noise_vol_bps > 0.0:
            # REVIEW: Noise is applied in absolute Black-vol basis points and
            # repriced to keep quote prices arbitrage-compatible pointwise.
            iv_slice = np.maximum(
                iv_slice
                + rng.normal(
                    loc=0.0,
                    scale=float(noise_vol_bps) * 1.0e-4,
                    size=iv_slice.shape,
                ),
                1.0e-6,
            )
            mid_slice = black76_call_price_vec(
                forward=forward,
                strikes=strike_slice,
                sigma=iv_slice,
                tau=float(tau),
                df=df,
            )
        else:
            mid_slice = clean_prices

        _, vega_slice = black76_call_price_vega_vec(
            forward=forward,
            strikes=strike_slice,
            sigma=iv_slice,
            tau=float(tau),
            df=df,
        )

        for point_idx, (strike, log_m, mid, iv, vega) in enumerate(
            zip(
                strike_slice,
                log_moneyness_grid,
                mid_slice,
                iv_slice,
                vega_slice,
                strict=True,
            )
        ):
            strikes.append(float(strike))
            expiry_values.append(float(tau))
            mid_values.append(float(mid))
            iv_values.append(float(iv))
            vega_values.append(float(vega))
            labels.append(f"call-T{float(tau):.6g}-k{float(log_m):+.6g}-{point_idx}")

    return HestonQuoteSet(
        ctx=ctx,
        strike=np.asarray(strikes, dtype=np.float64),
        expiry=np.asarray(expiry_values, dtype=np.float64),
        is_call=np.ones(len(strikes), dtype=np.bool_),
        mid=np.asarray(mid_values, dtype=np.float64),
        bs_vega=np.asarray(vega_values, dtype=np.float64),
        iv_mid=np.asarray(iv_values, dtype=np.float64),
        labels=tuple(labels),
    )


def _run_single_calibration(
    *,
    objective: HestonObjective,
    seed_params: HestonParams,
    mode: CalibrationBenchmarkMode,
    use_analytic_jac: bool,
    repeat_index: int,
    warmup: bool,
    loss: OptimizerLoss,
    method: OptimizerMethod,
    x_scale: str | np.ndarray | None,
) -> _CalibrationRunResult:
    def analytic_jac(u: np.ndarray, *_args: object, **_kwargs: object) -> FloatArray:
        return objective.jac(np.asarray(u, dtype=np.float64))

    jac: str | Any = analytic_jac if use_analytic_jac else "2-point"
    optimizer_kwargs: dict[str, Any] = {
        "fun": objective.residual,
        "jac": jac,
        "x0": seed_params.transform_to_unconstrained(),
        "loss": loss,
        "method": method,
    }
    optimizer_x_scale = _x_scale_for_optimizer(x_scale)
    if optimizer_x_scale is not None:
        optimizer_kwargs["x_scale"] = optimizer_x_scale

    t0 = perf_counter()
    result = least_squares(**optimizer_kwargs)
    runtime_ms = 1.0e3 * (perf_counter() - t0)

    fitted_params = HestonParams.transform_to_constrained(
        np.asarray(result.x, dtype=np.float64)
    )
    residual = np.asarray(result.fun, dtype=np.float64)

    return _CalibrationRunResult(
        mode=mode,
        use_analytic_jac=use_analytic_jac,
        repeat_index=repeat_index,
        warmup=warmup,
        runtime_ms=float(runtime_ms),
        result=result,
        fitted_params=fitted_params,
        residual=residual,
    )


def _result_row(
    run: _CalibrationRunResult,
    *,
    method: OptimizerMethod,
    loss: OptimizerLoss,
    backend: HestonDiagnosticsBackend,
    n_quotes: int,
) -> dict[str, Any]:
    residual = np.asarray(run.residual, dtype=np.float64)
    njev = getattr(run.result, "njev", None)
    return {
        "mode": run.mode,
        "use_analytic_jac": bool(run.use_analytic_jac),
        "repeat_index": int(run.repeat_index),
        "warmup": bool(run.warmup),
        "runtime_ms": float(run.runtime_ms),
        "nfev": int(getattr(run.result, "nfev", 0)),
        "njev": pd.NA if njev is None else int(njev),
        "cost": float(getattr(run.result, "cost", np.nan)),
        "residual_norm": float(np.linalg.norm(residual)),
        "max_abs_residual": (float(np.max(np.abs(residual))) if residual.size else 0.0),
        "success": bool(getattr(run.result, "success", False)),
        "status": int(getattr(run.result, "status", 0)),
        "message": str(getattr(run.result, "message", "")),
        "method": method,
        "loss": loss,
        "backend": backend,
        "n_quotes": int(n_quotes),
    }


def _median_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return float(numeric.median())
    return None


def _summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    measured = runs.loc[~runs["warmup"].astype(bool)].copy()
    baseline = measured.loc[measured["mode"] == "finite_difference"]
    baseline_runtime = _median_or_none(baseline["runtime_ms"])
    baseline_nfev = _median_or_none(baseline["nfev"])

    rows: list[dict[str, Any]] = []
    for mode, group in measured.groupby("mode", sort=False):
        median_runtime = _median_or_none(group["runtime_ms"])
        median_nfev = _median_or_none(group["nfev"])
        speedup = (
            None
            if baseline_runtime is None
            or median_runtime is None
            or median_runtime <= 0.0
            else baseline_runtime / median_runtime
        )
        nfev_ratio = (
            None
            if baseline_nfev is None or median_nfev is None or median_nfev <= 0.0
            else baseline_nfev / median_nfev
        )
        notes = (
            "finite-difference baseline"
            if mode == "finite_difference"
            else (
                "REVIEW: Speedup ratios are smoke-benchmark measurements and "
                "environment-dependent."
            )
        )
        rows.append(
            {
                "mode": str(mode),
                "median_runtime_ms": median_runtime,
                "mean_runtime_ms": float(group["runtime_ms"].mean()),
                "std_runtime_ms": float(group["runtime_ms"].std(ddof=0)),
                "median_nfev": median_nfev,
                "median_njev": _median_or_none(group["njev"]),
                "median_residual_norm": _median_or_none(group["residual_norm"]),
                "median_max_abs_residual": _median_or_none(group["max_abs_residual"]),
                "speedup_vs_finite_difference": speedup,
                "nfev_ratio_vs_finite_difference": nfev_ratio,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows)


def _parameter_recovery_table(
    selected_runs: dict[CalibrationBenchmarkMode, _CalibrationRunResult],
    *,
    true_params: HestonParams,
    seed_params: HestonParams,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    true_values = _params_to_dict(true_params)
    seed_values = _params_to_dict(seed_params)
    for mode, run in selected_runs.items():
        fitted_values = _params_to_dict(run.fitted_params)
        for parameter in HESTON_PARAM_NAMES:
            fitted = fitted_values[parameter]
            truth = true_values[parameter]
            rows.append(
                {
                    "mode": mode,
                    "parameter": parameter,
                    "true": truth,
                    "seed": seed_values[parameter],
                    "fitted": fitted,
                    "fit_minus_true": fitted - truth,
                    "abs_fit_minus_true": abs(fitted - truth),
                }
            )
    return pd.DataFrame(rows)


def _residual_table(
    selected_runs: dict[CalibrationBenchmarkMode, _CalibrationRunResult],
    *,
    quotes: HestonQuoteSet,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    log_moneyness = quotes.log_moneyness
    for mode, run in selected_runs.items():
        for quote_index, residual in enumerate(run.residual):
            rows.append(
                {
                    "mode": mode,
                    "quote_index": int(quote_index),
                    "expiry": float(quotes.expiry[quote_index]),
                    "strike": float(quotes.strike[quote_index]),
                    "log_moneyness": float(log_moneyness[quote_index]),
                    "residual": float(residual),
                }
            )
    return pd.DataFrame(rows)


def run_heston_calibration_benchmark_diagnostics(
    *,
    market: MarketData | PricingContext | None = None,
    true_params: HestonParams | None = None,
    seed_params: HestonParams | None = None,
    expiries: Sequence[float] | np.ndarray | None = None,
    log_moneyness: Sequence[float] | np.ndarray | None = None,
    backend: HestonDiagnosticsBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    repeat: int = 3,
    warmup: int = 1,
    loss: OptimizerLoss = "linear",
    method: OptimizerMethod = "trf",
    x_scale: str | np.ndarray | None = None,
    random_seed: int = 42,
    noise_vol_bps: float = 0.0,
) -> HestonCalibrationBenchmarkDiagnostics:
    """Compare Heston calibration with analytic vs numerical Jacobians.

    The runner builds one synthetic call quote grid, then runs SciPy
    ``least_squares`` twice on identical data: once with ``jac="2-point"`` and
    once with ``HestonObjective.jac``. It captures optimizer diagnostics and
    packages them into notebook-facing tables without plotting objects.

    REVIEW: Defaults define a smoke benchmark only. They intentionally exercise
    analytic-vs-finite-difference wiring without claiming production speedups
    or paper-equivalent calibration coverage.
    """
    if backend != "gauss_legendre":
        raise NotImplementedError(
            "Analytic Heston calibration Jacobian diagnostics currently require "
            "backend='gauss_legendre'. The quad backend does not expose the "
            "analytic probability Jacobian."
        )
    if method == "lm" and loss != "linear":
        raise ValueError(
            "method='lm' requires loss='linear' in scipy.optimize.least_squares."
        )
    if int(repeat) < 1:
        raise ValueError("repeat must be >= 1.")
    if int(warmup) < 0:
        raise ValueError("warmup must be >= 0.")
    if float(noise_vol_bps) < 0.0:
        raise ValueError("noise_vol_bps must be nonnegative.")

    resolved_true_params = (
        _default_true_params() if true_params is None else true_params
    )
    resolved_seed_params = (
        _default_seed_params() if seed_params is None else seed_params
    )
    resolved_quad_cfg = _default_quad_cfg() if quad_cfg is None else quad_cfg

    quotes = _build_synthetic_heston_quotes(
        market=market,
        true_params=resolved_true_params,
        expiries=expiries,
        log_moneyness=log_moneyness,
        backend=backend,
        quad_cfg=resolved_quad_cfg,
        random_seed=random_seed,
        noise_vol_bps=float(noise_vol_bps),
    )
    objective = HestonObjective(
        quotes=quotes,
        backend=backend,
        quad_cfg=resolved_quad_cfg,
    )

    mode_configs: tuple[tuple[CalibrationBenchmarkMode, bool], ...] = (
        ("finite_difference", False),
        ("analytic", True),
    )
    runs: list[_CalibrationRunResult] = []
    for mode, use_analytic_jac in mode_configs:
        for warmup_index in range(int(warmup)):
            runs.append(
                _run_single_calibration(
                    objective=objective,
                    seed_params=resolved_seed_params,
                    mode=mode,
                    use_analytic_jac=use_analytic_jac,
                    repeat_index=warmup_index - int(warmup),
                    warmup=True,
                    loss=loss,
                    method=method,
                    x_scale=x_scale,
                )
            )
        for repeat_index in range(int(repeat)):
            runs.append(
                _run_single_calibration(
                    objective=objective,
                    seed_params=resolved_seed_params,
                    mode=mode,
                    use_analytic_jac=use_analytic_jac,
                    repeat_index=repeat_index,
                    warmup=False,
                    loss=loss,
                    method=method,
                    x_scale=x_scale,
                )
            )

    runs_table = pd.DataFrame(
        [
            _result_row(
                run,
                method=method,
                loss=loss,
                backend=backend,
                n_quotes=quotes.n_quotes,
            )
            for run in runs
        ]
    )
    summary = _summary_table(runs_table)
    selected_runs: dict[CalibrationBenchmarkMode, _CalibrationRunResult] = {}
    for run in runs:
        if not run.warmup and run.mode not in selected_runs:
            selected_runs[run.mode] = run

    parameter_recovery = _parameter_recovery_table(
        selected_runs,
        true_params=resolved_true_params,
        seed_params=resolved_seed_params,
    )
    residuals = _residual_table(selected_runs, quotes=quotes)

    backend_meta = backend_config_meta(
        backend=backend,
        quad_cfg=resolved_quad_cfg,
        rule=None,
    )
    meta: dict[str, Any] = {
        "diagnostic": "heston_calibration_jacobian_benchmark",
        "benchmark_label": "smoke",
        "objective": "existing HestonObjective vega-scaled price residual",
        "modes": [mode for mode, _ in mode_configs],
        "repeat": int(repeat),
        "warmup": int(warmup),
        "loss": loss,
        "method": method,
        "x_scale": _x_scale_for_meta(x_scale),
        "random_seed": int(random_seed),
        "noise_vol_bps": float(noise_vol_bps),
        "n_quotes": int(quotes.n_quotes),
        "true_params": _params_to_dict(resolved_true_params),
        "seed_params": _params_to_dict(resolved_seed_params),
        "backend_config": backend_meta,
        "notes": [
            "REVIEW: Synthetic quote grid and quadrature settings are smoke-benchmark choices.",
            "REVIEW: Optimizer method/loss choices determine benchmark comparability.",
            "REVIEW: Residuals use the repo's existing vega-scaled price objective.",
            "REVIEW: Timing and speedup ratios are environment-dependent.",
        ],
    }
    arrays = {
        "quotes": {
            "strike": quotes.strike,
            "expiry": quotes.expiry,
            "log_moneyness": quotes.log_moneyness,
            "mid": quotes.mid,
            "iv_mid": quotes.iv_mid,
            "bs_vega": quotes.bs_vega,
        },
        "true_params": resolved_true_params.as_array(),
        "seed_params": resolved_seed_params.as_array(),
        "fitted_params": {
            mode: run.fitted_params.as_array() for mode, run in selected_runs.items()
        },
        "residuals": {mode: run.residual for mode, run in selected_runs.items()},
    }

    return HestonCalibrationBenchmarkDiagnostics(
        meta=meta,
        tables={
            "runs": runs_table,
            "summary": summary,
            "parameter_recovery": parameter_recovery,
            "residuals": residuals,
        },
        arrays=arrays,
    )


__all__ = ["run_heston_calibration_benchmark_diagnostics"]
