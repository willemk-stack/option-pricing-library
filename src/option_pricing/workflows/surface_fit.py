"""High-level volatility-surface fitting workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from option_pricing.marketdata.bundles import (
    LoadedModelValidationBundle,
    load_model_validation_bundle,
)
from option_pricing.marketdata.surface_ready import (
    PreparedESSVIMarketFit,
    PreparedSVIMarketFit,
    prepare_essvi_market_fit,
    prepare_svi_market_fit,
)
from option_pricing.vol.arbitrage import SurfaceNoArbReport, check_surface_noarb
from option_pricing.vol.ssvi import (
    ESSVIFitResult,
    ESSVIGlobalCalibrationConfig,
    ESSVINodalSmileSlice,
    ESSVINodalSurface,
    ESSVINodeConstraintReport,
    calibrate_essvi_global,
    validate_essvi_nodes,
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
_ESSVI_FIT_WORKFLOW_GUIDANCE = (
    "Use fit_essvi_from_bundle(path) for the one-shot workflow, or "
    "load_model_validation_bundle(path) -> prepare_essvi_market_fit(bundle) "
    "-> fit_essvi_market(prepared) when you need each step."
)
_MIN_SVI_SLICE_POINTS = 5
_ACT_365_DAYS = 365.0
_REQUIRED_PREPARED_COLUMNS = (
    "expiry_years",
    "log_moneyness",
    "total_variance",
    "sqrt_weight",
)
_ESSVI_REQUIRED_PREPARED_COLUMNS = (
    "y",
    "T",
    "price_mkt",
    "sqrt_weight",
    "is_call",
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
class ESSVIMarketFitConfig:
    """Optional eSSVI market-fit workflow settings.

    The low-level calibration options remain grouped in
    ``ESSVIGlobalCalibrationConfig`` instead of being mirrored field by field.
    """

    calibration_config: ESSVIGlobalCalibrationConfig | None = None
    validate_nodes: bool = True
    validate_static_noarb: bool = True

    def __post_init__(self) -> None:
        if self.calibration_config is not None and not isinstance(
            self.calibration_config,
            ESSVIGlobalCalibrationConfig,
        ):
            raise TypeError(
                "calibration_config must be an ESSVIGlobalCalibrationConfig or None"
            )
        _validate_bool("validate_nodes", self.validate_nodes)
        _validate_bool("validate_static_noarb", self.validate_static_noarb)


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


@dataclass(frozen=True, slots=True)
class ESSVIMarketFitValidation:
    node_report: ESSVINodeConstraintReport | None
    surface_noarb_report: SurfaceNoArbReport | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ESSVIMarketFitResult:
    model_name: str
    status: str
    prepared: PreparedESSVIMarketFit
    fit_result: ESSVIFitResult | None
    validation: ESSVIMarketFitValidation | None
    surface: ESSVINodalSurface | None
    summary: dict[str, Any]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def selected_points(self) -> pd.DataFrame:
        """Prepared eSSVI points selected for calibration."""

        return self.prepared.selected_points

    @property
    def rejected_points(self) -> pd.DataFrame:
        """Prepared eSSVI points rejected before calibration."""

        return self.prepared.rejected_points


class SVIMarketFitError(RuntimeError):
    """Raised when an SVI market-fit workflow fails with raise_on_failure."""


class ESSVIMarketFitError(RuntimeError):
    """Raised when an eSSVI market-fit workflow fails with raise_on_failure."""


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
    """Load or accept a bundle, prepare SVI points, and fit per-expiry SVI.

    ``raise_on_block`` is accepted for symmetry with other market-fit workflows.
    SVI preparation currently returns ``ready`` or ``empty`` in normal use; the
    flag is reserved for future structural SVI preparation blocks.
    """

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


def fit_essvi_market(
    prepared: PreparedESSVIMarketFit,
    *,
    fit_config: ESSVIMarketFitConfig | ESSVIGlobalCalibrationConfig | None = None,
    raise_on_failure: bool = False,
) -> ESSVIMarketFitResult:
    """Fit global eSSVI nodes from prepared market surface points."""

    if not isinstance(prepared, PreparedESSVIMarketFit):
        raise TypeError(
            "prepared must be a PreparedESSVIMarketFit. "
            "Use prepare_essvi_market_fit(bundle) before fit_essvi_market(...). "
            f"{_ESSVI_FIT_WORKFLOW_GUIDANCE}"
        )

    raise_on_failure = _validate_bool("raise_on_failure", raise_on_failure)
    config = _essvi_workflow_config(fit_config)
    calibration_config = (
        ESSVIGlobalCalibrationConfig()
        if config.calibration_config is None
        else config.calibration_config
    )

    warnings = tuple(prepared.warnings)
    if prepared.status == "empty":
        message = (
            "eSSVI market fit skipped because no selected surface points are "
            "available. Inspect prepared.rejected_points and "
            f"prepared.stats.rejection_counts. {_ESSVI_FIT_WORKFLOW_GUIDANCE}"
        )
        warnings = _dedupe_strings((*warnings, message))
        return _essvi_result(
            prepared,
            status="empty",
            fit_result=None,
            validation=None,
            surface=None,
            warnings=warnings,
            errors=(message,),
        )

    if prepared.status == "blocked":
        message = (
            "eSSVI market fit blocked by preparation status. Inspect "
            f"prepared.warnings before fitting. {_ESSVI_FIT_WORKFLOW_GUIDANCE}"
        )
        warnings = _dedupe_strings((*warnings, message))
        return _essvi_result(
            prepared,
            status="blocked",
            fit_result=None,
            validation=None,
            surface=None,
            warnings=warnings,
            errors=(message,),
        )

    try:
        selected = _essvi_selected_points(prepared.selected_points)
        fit_result = calibrate_essvi_global(
            y=_finite_array(selected, "y"),
            T=_finite_array(selected, "T"),
            price_mkt=_finite_array(selected, "price_mkt"),
            market=prepared.market_data,
            sqrt_weights=_finite_array(selected, "sqrt_weight"),
            is_call=_bool_array(selected, "is_call"),
            cfg=calibration_config,
        )
        surface, surface_warnings = _essvi_surface_from_fit(prepared, fit_result)
        validation = _essvi_validation(
            prepared,
            fit_result,
            config=config,
            calibration_config=calibration_config,
        )
    except Exception as exc:
        message = _essvi_failure_message(prepared, f"{type(exc).__name__}: {exc}")
        if raise_on_failure:
            raise ESSVIMarketFitError(message) from exc
        return _essvi_result(
            prepared,
            status="failed",
            fit_result=None,
            validation=None,
            surface=None,
            warnings=warnings,
            errors=(message,),
        )

    warnings = _dedupe_strings((*warnings, *surface_warnings, *validation.warnings))
    errors = _dedupe_strings(validation.errors)
    status = "ok" if validation.ok and surface is not None else "failed"
    if surface is None:
        errors = _dedupe_strings(
            (
                *errors,
                "eSSVI calibration succeeded but nodal surface construction failed.",
            )
        )

    if status == "failed" and raise_on_failure:
        detail = "; ".join(errors) if errors else "validation failed"
        raise ESSVIMarketFitError(_essvi_failure_message(prepared, detail))

    return _essvi_result(
        prepared,
        status=status,
        fit_result=fit_result,
        validation=validation,
        surface=surface,
        warnings=warnings,
        errors=errors,
    )


def fit_essvi_from_bundle(
    path_or_bundle: str | Path | LoadedModelValidationBundle,
    *,
    min_expiry_days: float = 7.0,
    max_expiry_days: float | None = None,
    min_points_per_expiry: int = 5,
    min_expiry_count: int = 3,
    min_moneyness: float | None = 0.5,
    max_moneyness: float | None = 2.0,
    require_mid: bool = True,
    require_iv: bool = False,
    raise_on_block: bool = False,
    fit_config: ESSVIMarketFitConfig | ESSVIGlobalCalibrationConfig | None = None,
    raise_on_failure: bool = False,
) -> ESSVIMarketFitResult:
    """Load or accept a bundle, prepare eSSVI points, and fit global eSSVI."""

    bundle = (
        path_or_bundle
        if isinstance(path_or_bundle, LoadedModelValidationBundle)
        else load_model_validation_bundle(path_or_bundle)
    )
    prepared = prepare_essvi_market_fit(
        bundle,
        min_expiry_days=min_expiry_days,
        max_expiry_days=max_expiry_days,
        min_points_per_expiry=min_points_per_expiry,
        min_expiry_count=min_expiry_count,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
        require_mid=require_mid,
        require_iv=require_iv,
        raise_on_block=raise_on_block,
    )
    return fit_essvi_market(
        prepared,
        fit_config=fit_config,
        raise_on_failure=raise_on_failure,
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


def _essvi_result(
    prepared: PreparedESSVIMarketFit,
    *,
    status: str,
    fit_result: ESSVIFitResult | None,
    validation: ESSVIMarketFitValidation | None,
    surface: ESSVINodalSurface | None,
    warnings: tuple[str, ...],
    errors: tuple[str, ...],
) -> ESSVIMarketFitResult:
    summary = _essvi_summary(
        prepared,
        status=status,
        fit_result=fit_result,
        validation=validation,
        warnings=warnings,
        errors=errors,
    )
    return ESSVIMarketFitResult(
        model_name=prepared.model_name,
        status=status,
        prepared=prepared,
        fit_result=fit_result,
        validation=validation,
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


def _essvi_summary(
    prepared: PreparedESSVIMarketFit,
    *,
    status: str,
    fit_result: ESSVIFitResult | None,
    validation: ESSVIMarketFitValidation | None,
    warnings: tuple[str, ...],
    errors: tuple[str, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_name": prepared.model_name,
        "calibration_status": status,
        "input_point_count": int(prepared.stats.input_point_count),
        "selected_point_count": int(prepared.stats.selected_point_count),
        "rejected_point_count": int(prepared.stats.rejected_point_count),
        "expiry_count": int(prepared.stats.expiry_count),
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

    if fit_result is not None:
        summary.update(
            {
                "node_count": int(fit_result.nodes.expiries.size),
                "price_rmse": float(fit_result.diag.price_rmse),
                "max_abs_price_error": float(fit_result.diag.max_abs_price_error),
                "optimizer_success": bool(fit_result.diag.success),
                "optimizer_nfev": int(fit_result.diag.nfev),
                "optimizer_cost": float(fit_result.diag.cost),
            }
        )
    if validation is not None:
        summary["validation_ok"] = bool(validation.ok)
        if validation.node_report is not None:
            summary["node_validation_ok"] = bool(validation.node_report.ok)
        if validation.surface_noarb_report is not None:
            summary["surface_noarb_ok"] = bool(validation.surface_noarb_report.ok)
    return summary


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


def _essvi_surface_from_fit(
    prepared: PreparedESSVIMarketFit,
    fit_result: ESSVIFitResult,
) -> tuple[ESSVINodalSurface | None, tuple[str, ...]]:
    try:
        y_min, y_max = _observed_y_domain(prepared.selected_points)
        return (
            ESSVINodalSurface(
                fit_result.nodes,
                y_min=y_min,
                y_max=y_max,
            ),
            (),
        )
    except Exception as exc:
        return None, (
            "eSSVI calibration succeeded but ESSVINodalSurface construction "
            f"failed: {type(exc).__name__}: {exc}",
        )


def _essvi_validation(
    prepared: PreparedESSVIMarketFit,
    fit_result: ESSVIFitResult,
    *,
    config: ESSVIMarketFitConfig,
    calibration_config: ESSVIGlobalCalibrationConfig,
) -> ESSVIMarketFitValidation:
    node_report: ESSVINodeConstraintReport | None = None
    surface_noarb_report: SurfaceNoArbReport | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    if config.validate_nodes:
        node_report = validate_essvi_nodes(
            fit_result.nodes,
            strict=False,
            tol=calibration_config.constraint_tol,
        )
        if not node_report.ok:
            errors = _dedupe_strings(
                (*errors, f"eSSVI node validation failed: {node_report.message}")
            )

    if config.validate_static_noarb:
        try:
            surface_noarb_report = _essvi_static_noarb_report(prepared, fit_result)
        except Exception as exc:
            errors = _dedupe_strings(
                (
                    *errors,
                    "eSSVI nodal surface validation failed to run: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            if not surface_noarb_report.ok:
                errors = _dedupe_strings(
                    (
                        *errors,
                        "eSSVI nodal surface static no-arbitrage validation "
                        f"failed: {surface_noarb_report.message}",
                    )
                )

    if node_report is None and surface_noarb_report is None:
        warnings = ("eSSVI fit validation was skipped by ESSVIMarketFitConfig.",)

    return ESSVIMarketFitValidation(
        node_report=node_report,
        surface_noarb_report=surface_noarb_report,
        warnings=warnings,
        errors=errors,
    )


def _essvi_static_noarb_report(
    prepared: PreparedESSVIMarketFit,
    fit_result: ESSVIFitResult,
) -> SurfaceNoArbReport:
    expiries = np.asarray(fit_result.nodes.expiries, dtype=np.float64)
    smiles: list[ESSVINodalSmileSlice] = []
    for expiry in expiries:
        y_min, y_max = _observed_y_domain(
            _essvi_points_for_expiry(prepared.selected_points, float(expiry))
        )
        smiles.append(
            ESSVINodalSmileSlice(
                T=float(expiry),
                nodes=fit_result.nodes,
                y_min=y_min,
                y_max=y_max,
            )
        )
    surface = VolSurface(
        expiries=expiries,
        smiles=tuple(smiles),
        forward=prepared.market_data.forward,
    )
    return check_surface_noarb(
        surface,
        df=prepared.market_data.df,
    )


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


def _essvi_selected_points(selected_points: pd.DataFrame) -> pd.DataFrame:
    selected = selected_points.copy(deep=True).reset_index(drop=True)
    _validate_essvi_prepared_columns(selected)
    if selected.empty:
        raise ValueError("prepared.selected_points must not be empty")
    selected["_essvi_fit_T"] = _numeric_series(selected["T"])
    selected["_essvi_fit_y"] = _numeric_series(selected["y"])
    selected = selected.sort_values(
        by=["_essvi_fit_T", "_essvi_fit_y"],
        kind="mergesort",
    )
    return selected.drop(columns=["_essvi_fit_T", "_essvi_fit_y"]).reset_index(
        drop=True
    )


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


def _essvi_points_for_expiry(
    selected_points: pd.DataFrame,
    expiry: float,
) -> pd.DataFrame:
    if selected_points.empty or "T" not in selected_points:
        return cast(pd.DataFrame, selected_points.iloc[0:0])
    T = _numeric_series(selected_points["T"])
    mask = np.isclose(T, float(expiry), rtol=0.0, atol=1e-12)
    return cast(pd.DataFrame, selected_points.loc[mask])


def _observed_y_domain(points: pd.DataFrame) -> tuple[float, float]:
    if points.empty or "y" not in points:
        return -2.5, 2.5
    y = pd.to_numeric(points["y"], errors="coerce").dropna()
    if y.empty:
        return -2.5, 2.5
    y_values = y.to_numpy(dtype=np.float64, copy=False)
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size == 0:
        return -2.5, 2.5
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if not y_min < y_max:
        return y_min - 0.05, y_max + 0.05
    padding = max(0.02, 0.05 * (y_max - y_min))
    return y_min - padding, y_max + padding


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


def _essvi_workflow_config(
    fit_config: ESSVIMarketFitConfig | ESSVIGlobalCalibrationConfig | None,
) -> ESSVIMarketFitConfig:
    if fit_config is None:
        return ESSVIMarketFitConfig()
    if isinstance(fit_config, ESSVIGlobalCalibrationConfig):
        return ESSVIMarketFitConfig(calibration_config=fit_config)
    if isinstance(fit_config, ESSVIMarketFitConfig):
        return fit_config
    raise TypeError(
        "fit_config must be an ESSVIMarketFitConfig, "
        "ESSVIGlobalCalibrationConfig, or None"
    )


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


def _validate_essvi_prepared_columns(selected: pd.DataFrame) -> None:
    missing = [
        column for column in _ESSVI_REQUIRED_PREPARED_COLUMNS if column not in selected
    ]
    if missing:
        joined = ", ".join(repr(column) for column in missing)
        raise ValueError(
            "prepared.selected_points is missing required eSSVI columns: "
            f"{joined}. Use prepare_essvi_market_fit(bundle) before fitting. "
            f"{_ESSVI_FIT_WORKFLOW_GUIDANCE}"
        )


def _finite_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _numeric_series(frame[column])
    if not np.all(np.isfinite(values)):
        raise ValueError(f"selected_points[{column!r}] must be finite")
    return np.asarray(values, dtype=np.float64)


def _bool_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if values.isna().any():
        raise ValueError(f"selected_points[{column!r}] must not contain missing values")
    return values.astype(bool).to_numpy(dtype=np.bool_, copy=False)


def _numeric_series(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )


def _essvi_failure_message(prepared: PreparedESSVIMarketFit, detail: str) -> str:
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
    return (
        f"eSSVI market fit failed ({rendered}): {detail}. "
        f"{_ESSVI_FIT_WORKFLOW_GUIDANCE}"
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
    "ESSVIMarketFitConfig",
    "ESSVIMarketFitError",
    "ESSVIMarketFitResult",
    "SVIMarketFitConfig",
    "SVIMarketFitError",
    "SVIMarketFitResult",
    "SVISliceMarketFitResult",
    "fit_essvi_from_bundle",
    "fit_essvi_market",
    "fit_svi_from_bundle",
    "fit_svi_market",
]
