"""High-level volatility-surface fitting workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from option_pricing.marketdata.bundles import (
    LoadedModelValidationBundle,
    load_model_validation_bundle,
)
from option_pricing.marketdata.surface_ready import (
    PreparedSVIMarketFit,
    prepare_svi_market_fit,
)
from option_pricing.vol.surface_core import VolSurface
from option_pricing.vol.svi import (
    DomainCheckConfig,
    SVIFitDiagnostics,
    SVIFitResult,
    SVIParams,
    SVISmile,
    calibrate_svi,
)
from option_pricing.vol.svi.regularization import RegOverride

_SVI_FIT_WORKFLOW_GUIDANCE = (
    "Use fit_svi_from_bundle(path) for the one-shot workflow, or "
    "load_model_validation_bundle(path) -> prepare_svi_market_fit(bundle) "
    "-> fit_svi_market(prepared) when you need each step."
)
_MIN_SVI_SLICE_POINTS = 5
_ACT_365_DAYS = 365.0
_REQUIRED_PREPARED_COLUMNS = (
    "expiry_years",
    "log_moneyness",
    "total_variance",
    "sqrt_weight",
)
_PARAMETER_TABLE_COLUMNS = (
    "expiry_years",
    "expiry_days",
    "status",
    "point_count",
    "a",
    "b",
    "rho",
    "m",
    "sigma",
    "diagnostics_ok",
    "failure_reason",
    "rmse_w",
    "rmse_unw",
    "max_abs_werr",
    "solver_cost",
    "solver_nfev",
    "error",
)


@dataclass(frozen=True, slots=True)
class SVIMarketFitConfig:
    """Optional overrides for per-expiry SVI slice calibration."""

    x0: SVIParams | None = None
    reg_override: RegOverride | None = None
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    f_scale: float = 1.0
    domain_check: DomainCheckConfig | None = None
    robust_data_only: bool = True
    irls_max_outer: int = 8
    irls_w_floor: float = 1e-4
    irls_damp: float = 0.0
    irls_tol: float = 1e-8
    repair_butterfly: bool = False
    repair_method: Literal["project", "line_search"] = "line_search"
    refit_after_repair: bool = True
    calibrate_kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SVISliceMarketFitResult:
    expiry_years: float
    status: str
    point_count: int
    selected_points: pd.DataFrame
    fit_result: SVIFitResult | None
    params: SVIParams | None
    diagnostics: SVIFitDiagnostics | None
    warnings: tuple[str, ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class SVIMarketFitResult:
    model_name: str
    status: str
    prepared: PreparedSVIMarketFit
    slice_results: tuple[SVISliceMarketFitResult, ...]
    parameter_table: pd.DataFrame
    surface: VolSurface | None
    summary: dict[str, Any]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def selected_points(self) -> pd.DataFrame:
        """Prepared SVI points selected for calibration."""

        return self.prepared.selected_points

    @property
    def rejected_points(self) -> pd.DataFrame:
        """Prepared SVI points rejected before calibration."""

        return self.prepared.rejected_points


class SVIMarketFitError(RuntimeError):
    """Raised when an SVI market-fit workflow fails with raise_on_failure."""


def fit_svi_market(
    prepared: PreparedSVIMarketFit,
    *,
    fit_config: SVIMarketFitConfig | None = None,
    raise_on_failure: bool = False,
    allow_partial: bool = True,
    allow_blocked: bool = False,
) -> SVIMarketFitResult:
    """Fit one raw-SVI slice per prepared market expiry."""

    if not isinstance(prepared, PreparedSVIMarketFit):
        raise TypeError(
            "prepared must be a PreparedSVIMarketFit. "
            "Use prepare_svi_market_fit(bundle) before fit_svi_market(...). "
            f"{_SVI_FIT_WORKFLOW_GUIDANCE}"
        )

    raise_on_failure = _validate_bool("raise_on_failure", raise_on_failure)
    allow_partial = _validate_bool("allow_partial", allow_partial)
    allow_blocked = _validate_bool("allow_blocked", allow_blocked)

    warnings = tuple(prepared.warnings)
    errors: tuple[str, ...] = ()

    if prepared.status == "empty":
        message = (
            "SVI market fit skipped because no selected surface points are "
            "available. Inspect prepared.rejected_points and "
            f"prepared.stats.rejection_counts. {_SVI_FIT_WORKFLOW_GUIDANCE}"
        )
        warnings = _dedupe_strings((*warnings, message))
        errors = (message,)
        return _result(
            prepared,
            status="empty",
            slice_results=(),
            warnings=warnings,
            errors=errors,
        )

    if prepared.status == "blocked" and not allow_blocked:
        message = (
            "SVI market fit blocked by preparation status. Inspect "
            "prepared.warnings and pass allow_blocked=True only for an "
            f"explicit advanced rerun. {_SVI_FIT_WORKFLOW_GUIDANCE}"
        )
        warnings = _dedupe_strings((*warnings, message))
        errors = (message,)
        return _result(
            prepared,
            status="blocked",
            slice_results=(),
            warnings=warnings,
            errors=errors,
        )

    if prepared.status == "blocked" and allow_blocked:
        warnings = _dedupe_strings(
            (*warnings, "SVI calibration ran despite blocked preparation status.")
        )

    try:
        kwargs = _fit_config_kwargs(fit_config)
        slice_results = tuple(_fit_expiry_slices(prepared, kwargs))
    except Exception as exc:
        message = _failure_message(prepared, f"{type(exc).__name__}: {exc}")
        if raise_on_failure:
            raise SVIMarketFitError(message) from exc
        return _result(
            prepared,
            status="failed",
            slice_results=(),
            warnings=warnings,
            errors=(message,),
        )

    slice_warnings = tuple(
        warning for slice_result in slice_results for warning in slice_result.warnings
    )
    slice_errors = tuple(
        slice_result.error for slice_result in slice_results if slice_result.error
    )
    warnings = _dedupe_strings((*warnings, *slice_warnings))
    errors = _dedupe_strings((*errors, *slice_errors))

    status = _calibration_status(
        slice_results,
        allow_partial=allow_partial,
    )
    if status == "failed" and slice_results and not errors:
        errors = ("SVI market fit failed without returning fitted expiry slices.",)

    if status in {"failed", "partial"} and raise_on_failure:
        detail = "; ".join(errors) if errors else f"status={status!r}"
        raise SVIMarketFitError(_failure_message(prepared, detail))

    return _result(
        prepared,
        status=status,
        slice_results=slice_results,
        warnings=warnings,
        errors=errors,
    )


def fit_svi_from_bundle(
    path_or_bundle: str | Path | LoadedModelValidationBundle,
    *,
    min_expiry_days: float = 7.0,
    max_expiry_days: float | None = None,
    min_points_per_expiry: int = 5,
    min_moneyness: float | None = 0.5,
    max_moneyness: float | None = 2.0,
    require_iv: bool = True,
    require_mid: bool = False,
    raise_on_block: bool = False,
    fit_config: SVIMarketFitConfig | None = None,
    raise_on_failure: bool = False,
    allow_partial: bool = True,
    allow_blocked: bool = False,
) -> SVIMarketFitResult:
    """Load or accept a bundle, prepare SVI points, and fit per-expiry SVI."""

    bundle = (
        path_or_bundle
        if isinstance(path_or_bundle, LoadedModelValidationBundle)
        else load_model_validation_bundle(path_or_bundle)
    )
    prepared = prepare_svi_market_fit(
        bundle,
        min_expiry_days=min_expiry_days,
        max_expiry_days=max_expiry_days,
        min_points_per_expiry=min_points_per_expiry,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
        require_iv=require_iv,
        require_mid=require_mid,
        raise_on_block=raise_on_block,
    )
    return fit_svi_market(
        prepared,
        fit_config=fit_config,
        raise_on_failure=raise_on_failure,
        allow_partial=allow_partial,
        allow_blocked=allow_blocked,
    )


def _fit_expiry_slices(
    prepared: PreparedSVIMarketFit,
    calibrate_kwargs: Mapping[str, Any],
) -> tuple[SVISliceMarketFitResult, ...]:
    selected = prepared.selected_points.copy(deep=True)
    _validate_prepared_columns(selected)
    if selected.empty:
        return ()

    expiry_values = _numeric_series(selected["expiry_years"])
    selected["_svi_fit_expiry_years"] = expiry_values
    finite_expiries = sorted(
        {
            float(expiry)
            for expiry in expiry_values
            if math.isfinite(float(expiry)) and float(expiry) > 0.0
        }
    )

    results: list[SVISliceMarketFitResult] = []
    invalid_expiry_mask = ~np.isfinite(expiry_values) | (expiry_values <= 0.0)
    if bool(invalid_expiry_mask.any()):
        invalid_points = selected.loc[invalid_expiry_mask].drop(
            columns=["_svi_fit_expiry_years"]
        )
        results.append(
            _failed_slice_result(
                prepared,
                expiry_years=math.nan,
                points=invalid_points,
                detail="selected_points contain nonpositive or nonfinite expiry_years",
            )
        )

    for expiry in finite_expiries:
        mask = np.isclose(expiry_values, expiry, rtol=0.0, atol=0.0)
        points = selected.loc[mask].drop(columns=["_svi_fit_expiry_years"])
        results.append(
            _fit_single_expiry(
                prepared,
                expiry_years=expiry,
                points=points,
                calibrate_kwargs=calibrate_kwargs,
            )
        )

    return tuple(results)


def _fit_single_expiry(
    prepared: PreparedSVIMarketFit,
    *,
    expiry_years: float,
    points: pd.DataFrame,
    calibrate_kwargs: Mapping[str, Any],
) -> SVISliceMarketFitResult:
    selected_points = points.copy(deep=True).reset_index(drop=True)
    try:
        y = _finite_array(selected_points, "log_moneyness")
        w_obs = _finite_array(selected_points, "total_variance")
        sqrt_weights = _finite_array(selected_points, "sqrt_weight")
        if y.size < _MIN_SVI_SLICE_POINTS:
            raise ValueError(
                "SVI slice requires at least "
                f"{_MIN_SVI_SLICE_POINTS} selected points, got {y.size}"
            )
        if not math.isfinite(expiry_years) or expiry_years <= 0.0:
            raise ValueError(f"expiry_years must be positive, got {expiry_years!r}")

        order = np.argsort(y)
        y_fit = np.asarray(y[order], dtype=np.float64)
        w_fit = np.asarray(w_obs[order], dtype=np.float64)
        sqrt_weights_fit = np.asarray(sqrt_weights[order], dtype=np.float64)
        fit_result = calibrate_svi(
            y=y_fit,
            w_obs=w_fit,
            sqrt_weights=sqrt_weights_fit,
            slice_T=float(expiry_years),
            **dict(calibrate_kwargs),
        )
    except Exception as exc:
        return _failed_slice_result(
            prepared,
            expiry_years=expiry_years,
            points=selected_points,
            detail=f"{type(exc).__name__}: {exc}",
        )

    warnings: tuple[str, ...] = ()
    diagnostics = fit_result.diag
    if not diagnostics.ok:
        warnings = (
            "SVI slice diagnostics reported a model warning for "
            f"expiry_years={expiry_years:.12g}: {diagnostics.failure_reason}",
        )
    return SVISliceMarketFitResult(
        expiry_years=float(expiry_years),
        status="ok",
        point_count=int(len(selected_points)),
        selected_points=selected_points,
        fit_result=fit_result,
        params=fit_result.params,
        diagnostics=diagnostics,
        warnings=warnings,
        error=None,
    )


def _failed_slice_result(
    prepared: PreparedSVIMarketFit,
    *,
    expiry_years: float,
    points: pd.DataFrame,
    detail: str,
) -> SVISliceMarketFitResult:
    selected_points = points.copy(deep=True).reset_index(drop=True)
    return SVISliceMarketFitResult(
        expiry_years=float(expiry_years),
        status="failed",
        point_count=int(len(selected_points)),
        selected_points=selected_points,
        fit_result=None,
        params=None,
        diagnostics=None,
        warnings=(),
        error=_failure_message(
            prepared,
            f"expiry_years={expiry_years!r}, point_count={len(selected_points)}: "
            f"{detail}",
        ),
    )


def _result(
    prepared: PreparedSVIMarketFit,
    *,
    status: str,
    slice_results: tuple[SVISliceMarketFitResult, ...],
    warnings: tuple[str, ...],
    errors: tuple[str, ...],
) -> SVIMarketFitResult:
    parameter_table = _parameter_table(slice_results)
    surface, surface_warnings = _surface_from_slice_results(prepared, slice_results)
    warnings = _dedupe_strings((*warnings, *surface_warnings))
    summary = _summary(
        prepared,
        status=status,
        slice_results=slice_results,
        warnings=warnings,
        errors=errors,
    )
    return SVIMarketFitResult(
        model_name=prepared.model_name,
        status=status,
        prepared=prepared,
        slice_results=slice_results,
        parameter_table=parameter_table,
        surface=surface,
        summary=summary,
        warnings=warnings,
        errors=errors,
    )


def _calibration_status(
    slice_results: tuple[SVISliceMarketFitResult, ...],
    *,
    allow_partial: bool,
) -> str:
    if not slice_results:
        return "failed"
    fitted_count = sum(
        1 for slice_result in slice_results if slice_result.status == "ok"
    )
    failed_count = sum(
        1 for slice_result in slice_results if slice_result.status == "failed"
    )
    if fitted_count > 0 and failed_count == 0:
        return "ok"
    if fitted_count > 0 and failed_count > 0:
        return "partial" if allow_partial else "failed"
    return "failed"


def _summary(
    prepared: PreparedSVIMarketFit,
    *,
    status: str,
    slice_results: tuple[SVISliceMarketFitResult, ...],
    warnings: tuple[str, ...],
    errors: tuple[str, ...],
) -> dict[str, Any]:
    fitted_count = sum(
        1 for slice_result in slice_results if slice_result.status == "ok"
    )
    failed_count = sum(
        1 for slice_result in slice_results if slice_result.status == "failed"
    )
    summary: dict[str, Any] = {
        "model_name": prepared.model_name,
        "calibration_status": status,
        "input_point_count": int(prepared.stats.input_point_count),
        "selected_point_count": int(prepared.stats.selected_point_count),
        "rejected_point_count": int(prepared.stats.rejected_point_count),
        "fitted_expiry_count": int(fitted_count),
        "failed_expiry_count": int(failed_count),
        "warning_count": len(warnings),
        "error_count": len(errors),
    }
    if prepared.stats.rejection_counts:
        summary["rejection_counts"] = dict(prepared.stats.rejection_counts)
    underlying = _selected_underlying(prepared.selected_points)
    if underlying is not None:
        summary["underlying"] = underlying
    spot = getattr(prepared.market_data, "spot", None)
    if spot is not None:
        summary["spot"] = float(spot)
    return summary


def _parameter_table(
    slice_results: tuple[SVISliceMarketFitResult, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slice_result in slice_results:
        params = slice_result.params
        diagnostics = slice_result.diagnostics
        checks = None if diagnostics is None else diagnostics.checks
        solver = None if diagnostics is None else diagnostics.solver
        expiry_years = float(slice_result.expiry_years)
        rows.append(
            {
                "expiry_years": expiry_years,
                "expiry_days": (
                    math.nan
                    if not math.isfinite(expiry_years)
                    else expiry_years * _ACT_365_DAYS
                ),
                "status": slice_result.status,
                "point_count": int(slice_result.point_count),
                "a": None if params is None else float(params.a),
                "b": None if params is None else float(params.b),
                "rho": None if params is None else float(params.rho),
                "m": None if params is None else float(params.m),
                "sigma": None if params is None else float(params.sigma),
                "diagnostics_ok": None if diagnostics is None else bool(diagnostics.ok),
                "failure_reason": (
                    None if diagnostics is None else diagnostics.failure_reason
                ),
                "rmse_w": None if checks is None else float(checks.rmse_w),
                "rmse_unw": None if checks is None else float(checks.rmse_unw),
                "max_abs_werr": (
                    None if checks is None else float(checks.max_abs_werr)
                ),
                "solver_cost": None if solver is None else float(solver.cost),
                "solver_nfev": None if solver is None else int(solver.nfev),
                "error": slice_result.error,
            }
        )
    return pd.DataFrame(rows, columns=list(_PARAMETER_TABLE_COLUMNS))


def _surface_from_slice_results(
    prepared: PreparedSVIMarketFit,
    slice_results: tuple[SVISliceMarketFitResult, ...],
) -> tuple[VolSurface | None, tuple[str, ...]]:
    fitted = [
        slice_result
        for slice_result in slice_results
        if slice_result.status == "ok"
        and slice_result.params is not None
        and math.isfinite(slice_result.expiry_years)
        and slice_result.expiry_years > 0.0
    ]
    if not fitted:
        return None, ()

    fitted.sort(key=lambda slice_result: slice_result.expiry_years)
    expiries: list[float] = []
    smiles: list[SVISmile] = []
    for slice_result in fitted:
        params = slice_result.params
        if params is None:
            continue
        y_min, y_max = _smile_domain(slice_result)
        expiries.append(float(slice_result.expiry_years))
        smiles.append(
            SVISmile(
                T=float(slice_result.expiry_years),
                params=params,
                y_min=y_min,
                y_max=y_max,
                diagnostics=slice_result.diagnostics,
            )
        )

    try:
        surface = VolSurface(
            expiries=np.asarray(expiries, dtype=np.float64),
            smiles=tuple(smiles),
            forward=prepared.market_data.forward,
        )
    except Exception as exc:
        return None, (
            "SVI calibration succeeded but VolSurface construction failed: "
            f"{type(exc).__name__}: {exc}",
        )
    return surface, ()


def _smile_domain(slice_result: SVISliceMarketFitResult) -> tuple[float, float]:
    if slice_result.diagnostics is not None:
        y_lo, y_hi = slice_result.diagnostics.checks.y_domain
        return float(y_lo), float(y_hi)

    values = pd.to_numeric(
        slice_result.selected_points.get("log_moneyness", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if values.empty:
        return -1.25, 1.25
    return float(values.min()), float(values.max())


def _fit_config_kwargs(fit_config: SVIMarketFitConfig | None) -> dict[str, Any]:
    if fit_config is None:
        return {}
    if not isinstance(fit_config, SVIMarketFitConfig):
        raise TypeError("fit_config must be an SVIMarketFitConfig or None")

    kwargs = {
        field.name: getattr(fit_config, field.name)
        for field in fields(SVIMarketFitConfig)
        if field.name != "calibrate_kwargs"
    }
    extra = (
        {} if fit_config.calibrate_kwargs is None else dict(fit_config.calibrate_kwargs)
    )
    managed = {"y", "w_obs", "sqrt_weights", "slice_T"}
    managed_overlap = managed.intersection(extra)
    if managed_overlap:
        names = ", ".join(sorted(managed_overlap))
        raise ValueError(f"fit_config.calibrate_kwargs cannot define {names}")

    mirrored_overlap = set(kwargs).intersection(extra)
    if mirrored_overlap:
        names = ", ".join(sorted(mirrored_overlap))
        raise ValueError(
            "fit_config.calibrate_kwargs duplicates explicit SVIMarketFitConfig "
            f"fields: {names}"
        )

    kwargs.update(extra)
    return kwargs


def _validate_prepared_columns(selected: pd.DataFrame) -> None:
    missing = [
        column for column in _REQUIRED_PREPARED_COLUMNS if column not in selected
    ]
    if missing:
        joined = ", ".join(repr(column) for column in missing)
        raise ValueError(
            "prepared.selected_points is missing required SVI columns: "
            f"{joined}. Use prepare_svi_market_fit(bundle) before fitting. "
            f"{_SVI_FIT_WORKFLOW_GUIDANCE}"
        )


def _finite_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numeric_series(frame[column])
    if not np.all(np.isfinite(values)):
        raise ValueError(f"selected_points[{column!r}] must be finite")
    return np.asarray(values, dtype=np.float64)


def _numeric_series(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )


def _failure_message(prepared: PreparedSVIMarketFit, detail: str) -> str:
    context = {
        "selected_point_count": int(prepared.stats.selected_point_count),
        "rejected_point_count": int(prepared.stats.rejected_point_count),
        "preparation_status": prepared.status,
        "underlying": _selected_underlying(prepared.selected_points),
        "spot": getattr(prepared.market_data, "spot", None),
    }
    rendered = ", ".join(
        f"{key}={value!r}" for key, value in context.items() if value is not None
    )
    return f"SVI market fit failed ({rendered}): {detail}. {_SVI_FIT_WORKFLOW_GUIDANCE}"


def _selected_underlying(selected_points: pd.DataFrame) -> str | None:
    if "underlying" not in selected_points.columns or selected_points.empty:
        return None
    values = selected_points["underlying"].dropna().astype(str).unique()
    if len(values) == 0:
        return None
    if len(values) == 1:
        return str(values[0])
    return ",".join(sorted(str(value) for value in values))


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = [
    "SVIMarketFitConfig",
    "SVIMarketFitError",
    "SVIMarketFitResult",
    "SVISliceMarketFitResult",
    "fit_svi_from_bundle",
    "fit_svi_market",
]
