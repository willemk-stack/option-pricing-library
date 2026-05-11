"""
Calibration helpers and main object
"""

from __future__ import annotations

from typing import Literal, SupportsFloat, SupportsIndex, SupportsInt, cast, overload

import numpy as np

from ...._scipy_compat import OptimizeResult, least_squares
from ....exceptions import NoConvergenceError
from ....typing import FloatArray
from ..charfunc import HESTON_ANALYTIC_JAC_ETA_MIN
from ..fourier import HestonBackend, QuadratureConfig
from ..params import HestonParams
from .bounds import HestonCalibrationBounds, transform_to_bounded_constrained
from .heston_types import (
    HESTON_PARAMETER_TRANSFORMS,
    HestonCalibrationRun,
    HestonMultistartResult,
    HestonObjectiveType,
    HestonParameterTransform,
    HestonQuoteSet,
    HestonRegConfig,
)
from .objective import HestonObjective
from .seeding import default_heston_seed, heston_seed_grid


def _dedupe_multistart_seeds(
    seeds: list[HestonParams],
    *,
    decimals: int = 10,
) -> tuple[HestonParams, ...]:
    # NOTE: Deduplicate by rounded constrained parameter values so repeated
    # starts do not waste optimizer budget or skew initialization diagnostics.
    seen: set[tuple[float, ...]] = set()
    out: list[HestonParams] = []

    for seed in seeds:
        key = tuple(np.round(seed.as_array(), decimals=decimals))
        if key in seen:
            continue
        seen.add(key)
        out.append(seed)

    return tuple(out)


def _build_multistart_seeds(
    quotes: HestonQuoteSet,
    *,
    seeds: tuple[HestonParams, ...] | list[HestonParams] | None,
    x0_params: HestonParams | None,
    bounds: HestonCalibrationBounds | None,
    max_seeds: int | None,
    include_default_seed: bool,
) -> tuple[HestonParams, ...]:
    # NOTE: Reject nonpositive limits before seed construction so explicit
    # and generated seed paths expose the same public API validation.
    if max_seeds is not None and max_seeds <= 0:
        raise ValueError("max_seeds must be positive or None.")

    resolved_bounds = bounds if bounds is not None else HestonCalibrationBounds()

    if seeds is None:
        generated = list(
            heston_seed_grid(
                quotes,
                bounds=resolved_bounds,
                max_seeds=max_seeds,
                include_default=include_default_seed,
            )
        )
    else:
        generated = list(seeds)

    # NOTE: Prepend x0_params so callers migrating from single-start
    # calibration keep their explicit initial guess as the first attempted run.
    if x0_params is not None:
        generated.insert(0, x0_params)

    deduped = _dedupe_multistart_seeds(generated)

    # NOTE: Apply max_seeds after x0_params insertion and deduplication so the
    # final optimizer-run count is bounded while preserving caller priority.
    if max_seeds is not None:
        deduped = deduped[:max_seeds]

    if not deduped:
        raise ValueError("calibrate_heston_multistart requires at least one seed.")

    return deduped


def _optional_finite_float(value: SupportsFloat | None) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float


def _optional_int(value: SupportsInt | SupportsIndex | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _run_sort_key(run: HestonCalibrationRun) -> tuple[int, float, int]:
    # NOTE: Sort successful runs before failed runs, then by cost and original
    # seed order, so best_run is deterministic and failures remain inspectable.
    return (0 if run.success else 1, run.cost, run.seed_index)


def _analytic_jacobian_policy_metadata(
    *,
    use_analytic_jac: bool,
    backend: HestonBackend,
    parameter_transform: HestonParameterTransform,
) -> dict[str, object]:
    mode = "analytic" if use_analytic_jac else "finite_difference"
    return {
        "heston_jacobian_mode": mode,
        "heston_analytic_jacobian_requested": bool(use_analytic_jac),
        "heston_analytic_jacobian_backend": backend,
        "heston_analytic_jacobian_eta_min": float(HESTON_ANALYTIC_JAC_ETA_MIN),
        "heston_analytic_jacobian_supported_domain": (
            "fixed Gauss-Legendre backend, nonzero fixed-rule quadrature nodes, "
            "eta at or above the analytic-Jacobian floor, and documented calibration "
            "bounds"
        ),
        "heston_parameter_transform": parameter_transform,
    }


def _validate_analytic_jacobian_calibration_request(
    *,
    x0_params: HestonParams,
    backend: HestonBackend,
    bounds: HestonCalibrationBounds | None,
) -> HestonCalibrationBounds:
    """Validate the supported analytic-Jacobian calibration domain."""
    if backend != "gauss_legendre":
        raise NotImplementedError(
            "Analytic Heston calibration Jacobians require backend='gauss_legendre'. "
            "Use use_analytic_jac=False for finite-difference calibration on other "
            "backends."
        )

    resolved_bounds = HestonCalibrationBounds() if bounds is None else bounds
    resolved_bounds.require_analytic_jacobian_compatible()

    if x0_params.eta < HESTON_ANALYTIC_JAC_ETA_MIN:
        raise ValueError(
            "Analytic Heston calibration Jacobians require initial eta >= "
            f"{HESTON_ANALYTIC_JAC_ETA_MIN:g}. Price-only deterministic-limit "
            "pricing near eta=0 is separate; use finite-difference calibration "
            "or choose an initial eta above the analytic-Jacobian floor."
        )

    values = x0_params.as_array()
    lower = resolved_bounds.lower_array()
    upper = resolved_bounds.upper_array()
    outside = (values < lower) | (values > upper)
    if np.any(outside):
        idx = int(np.flatnonzero(outside)[0])
        name = ("kappa", "vbar", "eta", "rho", "v")[idx]
        raise ValueError(
            "Analytic Heston calibration Jacobians are default-enabled only for "
            f"documented calibration bounds. Initial {name}={values[idx]} is outside "
            f"[{lower[idx]}, {upper[idx]}]."
        )

    return resolved_bounds


def _all_failed_message(runs: tuple[HestonCalibrationRun, ...]) -> str:
    preview_count = 3
    preview = "; ".join(
        f"seed {run.seed_index}: {run.message}" for run in runs[:preview_count]
    )
    remaining = len(runs) - preview_count
    suffix = f"; ... {remaining} more failure(s)" if remaining > 0 else ""
    return (
        f"Heston multistart calibration failed because all {len(runs)} seed(s) failed. "
        f"Failures: {preview}{suffix}"
    )


def _validate_heston_calibration_inputs(
    quotes: HestonQuoteSet,
    *,
    objective_type: HestonObjectiveType,
) -> None:
    if quotes.n_quotes == 0:
        raise ValueError("Heston calibration requires at least one quote.")

    if objective_type != "bid_ask_normalized":
        return

    if quotes.bid is None or quotes.ask is None:
        raise ValueError(
            "objective_type='bid_ask_normalized' requires quotes.bid and quotes.ask"
        )

    spread = np.asarray(quotes.ask - quotes.bid, dtype=np.float64)
    if not np.all(np.isfinite(spread)):
        raise ValueError("bid/ask spread must be finite")
    if np.any(spread <= 0.0):
        raise ValueError("bid/ask spread must be strictly positive")


# NOTE: Use overloads for return_result so the backward-compatible fitted
# parameter return and the diagnostics tuple return stay type-checkable without
# casts at call sites.
@overload
def calibrate_heston(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None = None,
    objective_type: HestonObjectiveType = "vega_scaled_price",
    x0_params: HestonParams | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    x_scale: Literal["jac"] | float | FloatArray | None = "jac",
    parameter_transform: HestonParameterTransform = "unconstrained",
    vega_floor: float | None = None,
    price_floor: float | None = None,
    spread_floor: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    reg: HestonRegConfig | None = None,
    bounds: HestonCalibrationBounds | None = None,
    use_analytic_jac: bool = True,
    method: Literal["trf", "dogbox", "lm"] = "trf",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    return_result: Literal[False] = False,
) -> HestonParams: ...


@overload
def calibrate_heston(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None = None,
    objective_type: HestonObjectiveType = "vega_scaled_price",
    x0_params: HestonParams | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    x_scale: Literal["jac"] | float | FloatArray | None = "jac",
    parameter_transform: HestonParameterTransform = "unconstrained",
    vega_floor: float | None = None,
    price_floor: float | None = None,
    spread_floor: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    reg: HestonRegConfig | None = None,
    bounds: HestonCalibrationBounds | None = None,
    use_analytic_jac: bool = True,
    method: Literal["trf", "dogbox", "lm"] = "trf",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    return_result: Literal[True] = True,
) -> tuple[HestonParams, OptimizeResult]: ...


@overload
def calibrate_heston(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None = None,
    objective_type: HestonObjectiveType = "vega_scaled_price",
    x0_params: HestonParams | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    x_scale: Literal["jac"] | float | FloatArray | None = "jac",
    parameter_transform: HestonParameterTransform = "unconstrained",
    vega_floor: float | None = None,
    price_floor: float | None = None,
    spread_floor: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    reg: HestonRegConfig | None = None,
    bounds: HestonCalibrationBounds | None = None,
    use_analytic_jac: bool = True,
    method: Literal["trf", "dogbox", "lm"] = "trf",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    return_result: bool = False,
) -> HestonParams | tuple[HestonParams, OptimizeResult]: ...


def calibrate_heston(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None = None,
    objective_type: HestonObjectiveType = "vega_scaled_price",
    x0_params: HestonParams | None = None,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    x_scale: Literal["jac"] | float | FloatArray | None = "jac",
    parameter_transform: HestonParameterTransform = "unconstrained",
    vega_floor: float | None = None,
    price_floor: float | None = None,
    spread_floor: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    reg: HestonRegConfig | None = None,
    bounds: HestonCalibrationBounds | None = None,
    use_analytic_jac: bool = True,
    method: Literal["trf", "dogbox", "lm"] = "trf",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    return_result: bool = False,
) -> HestonParams | tuple[HestonParams, OptimizeResult]:
    if parameter_transform not in HESTON_PARAMETER_TRANSFORMS:
        supported = ", ".join(repr(value) for value in HESTON_PARAMETER_TRANSFORMS)
        raise ValueError(
            f"parameter_transform must be one of {supported}; "
            f"got {parameter_transform!r}"
        )
    _validate_heston_calibration_inputs(quotes, objective_type=objective_type)
    if parameter_transform == "unconstrained" and bounds is not None:
        raise ValueError("bounds are only used with parameter_transform='bounded'.")
    if x0_params is None:
        x0_params = default_heston_seed(
            quotes,
            bounds=bounds if bounds is not None else HestonCalibrationBounds(),
        )
    analytic_bounds = (
        _validate_analytic_jacobian_calibration_request(
            x0_params=x0_params,
            backend=backend,
            bounds=bounds,
        )
        if use_analytic_jac
        else None
    )
    # NOTE: Keep the default optimizer at "trf" because the default
    # loss="soft_l1" is invalid for SciPy's method="lm"; explicit LM is still
    # allowed with loss="linear".
    if method == "lm" and loss != "linear":
        raise ValueError(
            "method='lm' requires loss='linear' in scipy.optimize.least_squares."
        )

    obj = HestonObjective(
        quotes=quotes,
        objective_type=objective_type,
        sqrt_weights=sqrt_weights,
        vega_floor=1e-6 if vega_floor is None else float(vega_floor),
        price_floor=1e-10 if price_floor is None else float(price_floor),
        spread_floor=1e-6 if spread_floor is None else float(spread_floor),
        backend=backend,
        quad_cfg=quad_cfg,
        reg=reg,
        parameter_transform=parameter_transform,
        bounds=bounds,
    )

    def analytic_jac(u: np.ndarray, *_args: object, **_kwargs: object) -> FloatArray:
        return obj.jac(np.asarray(u, dtype=np.float64))

    if parameter_transform == "bounded":
        resolved_bounds = (
            analytic_bounds
            if analytic_bounds is not None
            else (bounds if bounds is not None else HestonCalibrationBounds())
        )
        x0 = x0_params.transform_to_bounded_unconstrained(resolved_bounds)
    else:
        resolved_bounds = None
        x0 = x0_params.transform_to_unconstrained()
    if any(tol is not None for tol in (ftol, xtol, gtol)):
        if use_analytic_jac:
            res = least_squares(
                fun=obj.residual,
                jac=analytic_jac,
                x0=x0,
                loss=loss,
                x_scale=x_scale,
                method=method,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )
        else:
            res = least_squares(
                fun=obj.residual,
                jac="2-point",
                x0=x0,
                loss=loss,
                x_scale=x_scale,
                method=method,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )
    else:
        if use_analytic_jac:
            res = least_squares(
                fun=obj.residual,
                jac=analytic_jac,
                x0=x0,
                loss=loss,
                x_scale=x_scale,
                method=method,
                max_nfev=max_nfev,
            )
        else:
            res = least_squares(
                fun=obj.residual,
                jac="2-point",
                x0=x0,
                loss=loss,
                x_scale=x_scale,
                method=method,
                max_nfev=max_nfev,
            )

    if not res.success or not np.all(np.isfinite(res.x)):
        raise NoConvergenceError(f"Heston calibration failed: {res.message}")

    res.update(
        _analytic_jacobian_policy_metadata(
            use_analytic_jac=use_analytic_jac,
            backend=backend,
            parameter_transform=parameter_transform,
        )
    )

    raw_fit = np.asarray(res.x, dtype=np.float64)
    if parameter_transform == "bounded":
        assert resolved_bounds is not None
        fitted_params = transform_to_bounded_constrained(raw_fit, resolved_bounds)
    else:
        fitted_params = HestonParams.transform_to_constrained(raw_fit)
    if return_result:
        return fitted_params, res
    return fitted_params


def calibrate_heston_multistart(
    quotes: HestonQuoteSet,
    sqrt_weights: FloatArray | None = None,
    objective_type: HestonObjectiveType = "vega_scaled_price",
    seeds: tuple[HestonParams, ...] | list[HestonParams] | None = None,
    x0_params: HestonParams | None = None,
    max_seeds: int | None = 12,
    include_default_seed: bool = True,
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
    x_scale: Literal["jac"] | float | FloatArray | None = "jac",
    parameter_transform: HestonParameterTransform = "bounded",
    vega_floor: float | None = None,
    price_floor: float | None = None,
    spread_floor: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    reg: HestonRegConfig | None = None,
    bounds: HestonCalibrationBounds | None = None,
    use_analytic_jac: bool = True,
    method: Literal["trf", "dogbox", "lm"] = "trf",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
) -> HestonMultistartResult:
    """Run Heston calibration from multiple deterministic starting points.

    Invalid calibration inputs raise before any optimizer run. When at least
    one seed succeeds, failed seeds are still retained in ``runs`` because they
    are useful initialization-sensitivity diagnostics. If every seed fails,
    this function raises ``NoConvergenceError`` instead of returning a
    ``HestonMultistartResult`` without valid ``best_run`` / ``best_params``,
    preserving the successful-result type contract.
    """
    # NOTE: Multi-start defaults to the bounded transform because seed-grid
    # diagnostics are most useful when every optimizer run stays inside the
    # same practical calibration box; calibrate_heston keeps its old default.
    _validate_heston_calibration_inputs(quotes, objective_type=objective_type)
    resolved_seeds = _build_multistart_seeds(
        quotes,
        seeds=seeds,
        x0_params=x0_params,
        bounds=bounds,
        max_seeds=max_seeds,
        include_default_seed=include_default_seed,
    )

    runs: list[HestonCalibrationRun] = []

    for seed_index, seed in enumerate(resolved_seeds):
        try:
            fitted_params, scipy_result = calibrate_heston(
                quotes=quotes,
                sqrt_weights=sqrt_weights,
                objective_type=objective_type,
                x0_params=seed,
                loss=loss,
                x_scale=x_scale,
                parameter_transform=parameter_transform,
                vega_floor=vega_floor,
                price_floor=price_floor,
                spread_floor=spread_floor,
                backend=backend,
                quad_cfg=quad_cfg,
                reg=reg,
                bounds=bounds,
                use_analytic_jac=use_analytic_jac,
                method=method,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                return_result=True,
            )
            run = HestonCalibrationRun(
                seed_index=seed_index,
                seed_params=seed,
                fitted_params=fitted_params,
                success=True,
                cost=float(cast(SupportsFloat, scipy_result.cost)),
                optimality=_optional_finite_float(
                    getattr(scipy_result, "optimality", None)
                ),
                nfev=_optional_int(getattr(scipy_result, "nfev", None)),
                njev=_optional_int(getattr(scipy_result, "njev", None)),
                status=_optional_int(getattr(scipy_result, "status", None)),
                message=str(scipy_result.message),
                raw_x=np.asarray(scipy_result.x, dtype=np.float64),
            )
        except Exception as exc:
            # NOTE: Do not raise on individual seed failures; retain failed
            # runs because initialization sensitivity is a calibration diagnostic.
            run = HestonCalibrationRun(
                seed_index=seed_index,
                seed_params=seed,
                fitted_params=None,
                success=False,
                cost=np.inf,
                optimality=None,
                nfev=None,
                njev=None,
                status=None,
                message=f"{type(exc).__name__}: {exc}",
                raw_x=None,
            )

        runs.append(run)

    sorted_runs = tuple(sorted(runs, key=_run_sort_key))
    successful_runs = tuple(run for run in sorted_runs if run.success)

    if not successful_runs:
        # NOTE: Always raise when all seeds fail because a result object
        # without a successful best_run would violate its own type contract.
        raise NoConvergenceError(_all_failed_message(sorted_runs))

    best_run = successful_runs[0]
    assert best_run.fitted_params is not None

    return HestonMultistartResult(
        best_params=best_run.fitted_params,
        best_run=best_run,
        runs=sorted_runs,
        objective_type=objective_type,
        parameter_transform=parameter_transform,
        backend=backend,
        quote_count=quotes.n_quotes,
        success_count=len(successful_runs),
        failure_count=len(sorted_runs) - len(successful_runs),
        jacobian_mode="analytic" if use_analytic_jac else "finite_difference",
        analytic_jacobian_eta_min=HESTON_ANALYTIC_JAC_ETA_MIN,
    )
