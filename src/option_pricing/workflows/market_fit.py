"""High-level market-fitting workflows."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from option_pricing.marketdata.bundles import (
    LoadedModelValidationBundle,
    load_model_validation_bundle,
)
from option_pricing.marketdata.model_ready import (
    PreparedHestonMarketFit,
    prepare_heston_market_fit,
)
from option_pricing.models.heston.calibration import (
    HestonCalibrationBounds,
    HestonMultistartResult,
    HestonObjectiveType,
    HestonParameterTransform,
    calibrate_heston_multistart,
)
from option_pricing.models.heston.params import HestonParams

_HESTON_FIT_WORKFLOW_GUIDANCE = (
    "Use fit_heston_from_bundle(path) for the canonical one-shot workflow, "
    "or load_model_validation_bundle(path) -> prepare_heston_market_fit(bundle) "
    "-> fit_heston_market(prepared) when you need each step."
)


@dataclass(frozen=True, slots=True)
class HestonCalibrationConfig:
    """Optional overrides for the existing Heston multistart calibrator."""

    seeds: tuple[HestonParams, ...] | list[HestonParams] | None = None
    x0_params: HestonParams | None = None
    include_default_seed: bool = True
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    x_scale: Any = "jac"
    parameter_transform: HestonParameterTransform = "bounded"
    vega_floor: float | None = None
    price_floor: float | None = None
    spread_floor: float | None = None
    backend: Literal["gauss_legendre", "quad"] = "gauss_legendre"
    quad_cfg: Any | None = None
    reg: Any | None = None
    bounds: HestonCalibrationBounds | None = None
    use_analytic_jac: bool = True
    method: Literal["trf", "dogbox", "lm"] = "trf"
    max_nfev: int | None = None
    ftol: float | None = None
    xtol: float | None = None
    gtol: float | None = None


@dataclass(frozen=True, slots=True)
class HestonMarketFitResult:
    model_name: str
    status: str
    prepared: PreparedHestonMarketFit
    calibration_result: HestonMultistartResult | None
    best_params: HestonParams | None
    summary: dict[str, Any]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


class HestonMarketFitError(RuntimeError):
    """Raised when a Heston market-fit workflow fails with raise_on_failure."""


def fit_heston_market(
    prepared: PreparedHestonMarketFit,
    *,
    calibration_config: HestonCalibrationConfig | None = None,
    max_seeds: int | None = 8,
    raise_on_failure: bool = False,
    allow_blocked: bool = False,
) -> HestonMarketFitResult:
    """Fit Heston to a prepared market universe."""

    if not isinstance(prepared, PreparedHestonMarketFit):
        raise TypeError(
            "prepared must be a PreparedHestonMarketFit. "
            "Use prepare_heston_market_fit(bundle) before fit_heston_market(...). "
            f"{_HESTON_FIT_WORKFLOW_GUIDANCE}"
        )

    raise_on_failure = _validate_bool("raise_on_failure", raise_on_failure)
    allow_blocked = _validate_bool("allow_blocked", allow_blocked)
    if max_seeds is not None and int(max_seeds) <= 0:
        raise ValueError("max_seeds must be positive or None.")
    resolved_max_seeds = None if max_seeds is None else int(max_seeds)

    warnings = tuple(prepared.warnings)
    errors: tuple[str, ...] = ()

    if prepared.status == "empty":
        message = (
            "Heston market fit skipped because no selected quotes are available. "
            "Inspect prepared.rejected_quotes and prepared.stats.rejection_counts. "
            f"{_HESTON_FIT_WORKFLOW_GUIDANCE}"
        )
        warnings = _dedupe_strings((*warnings, message))
        errors = (message,)
        return _result(
            prepared,
            status="empty",
            calibration_result=None,
            warnings=warnings,
            errors=errors,
        )

    if prepared.status == "blocked" and not allow_blocked:
        message = (
            "Heston market fit blocked by quote preflight. "
            "Inspect prepared.preflight and prepared.warnings, or pass "
            "allow_blocked=True only for an explicit advanced rerun. "
            f"{_HESTON_FIT_WORKFLOW_GUIDANCE}"
        )
        warnings = _dedupe_strings((*warnings, message))
        errors = (message,)
        return _result(
            prepared,
            status="blocked",
            calibration_result=None,
            warnings=warnings,
            errors=errors,
        )

    if prepared.quote_set is None:
        message = (
            "Heston market fit failed because prepared.quote_set is not available "
            f"for prepared status {prepared.status!r}. "
            f"{_HESTON_FIT_WORKFLOW_GUIDANCE}"
        )
        errors = (_failure_message(prepared, message),)
        if raise_on_failure:
            raise HestonMarketFitError(errors[0])
        return _result(
            prepared,
            status="failed",
            calibration_result=None,
            warnings=warnings,
            errors=errors,
        )

    if prepared.status == "blocked" and allow_blocked:
        warnings = _dedupe_strings(
            (
                *warnings,
                "Heston calibration ran despite blocked quote preflight.",
            )
        )

    try:
        kwargs = _calibration_config_kwargs(calibration_config)
        if resolved_max_seeds is not None:
            kwargs["max_seeds"] = resolved_max_seeds
        calibration_result = calibrate_heston_multistart(
            prepared.quote_set,
            objective_type=prepared.objective_type,
            **kwargs,
        )
    except Exception as exc:
        message = _failure_message(
            prepared,
            f"{type(exc).__name__}: {exc}",
        )
        if raise_on_failure:
            raise HestonMarketFitError(message) from exc
        errors = (message,)
        return _result(
            prepared,
            status="failed",
            calibration_result=None,
            warnings=warnings,
            errors=errors,
        )

    return _result(
        prepared,
        status="ok",
        calibration_result=calibration_result,
        warnings=warnings,
        errors=errors,
    )


def fit_heston_from_bundle(
    path_or_bundle: str | Path | LoadedModelValidationBundle,
    *,
    objective_type: HestonObjectiveType | str = "price_rmse",
    min_expiry_days: float = 7.0,
    max_expiry_days: float | None = None,
    min_moneyness: float | None = 0.5,
    max_moneyness: float | None = 2.0,
    require_iv_for_seed: bool = True,
    require_vega_for_objective: bool | None = None,
    raise_on_block: bool = False,
    calibration_config: HestonCalibrationConfig | None = None,
    max_seeds: int | None = 8,
    raise_on_failure: bool = False,
    allow_blocked: bool = False,
) -> HestonMarketFitResult:
    """Load or accept a bundle, prepare Heston quotes, and fit Heston."""

    bundle = (
        path_or_bundle
        if isinstance(path_or_bundle, LoadedModelValidationBundle)
        else load_model_validation_bundle(path_or_bundle)
    )
    prepared = prepare_heston_market_fit(
        bundle,
        objective_type=objective_type,
        min_expiry_days=min_expiry_days,
        max_expiry_days=max_expiry_days,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
        require_iv_for_seed=require_iv_for_seed,
        require_vega_for_objective=require_vega_for_objective,
        raise_on_block=raise_on_block,
    )
    return fit_heston_market(
        prepared,
        calibration_config=calibration_config,
        max_seeds=max_seeds,
        raise_on_failure=raise_on_failure,
        allow_blocked=allow_blocked,
    )


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _calibration_config_kwargs(
    calibration_config: HestonCalibrationConfig | None,
) -> dict[str, Any]:
    if calibration_config is None:
        return {}
    if not isinstance(calibration_config, HestonCalibrationConfig):
        raise TypeError("calibration_config must be a HestonCalibrationConfig or None")

    kwargs = {
        field.name: getattr(calibration_config, field.name)
        for field in fields(HestonCalibrationConfig)
    }
    managed = {"quotes", "objective_type", "max_seeds"}
    overlap = managed.intersection(kwargs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"calibration_config cannot define managed fields: {names}")
    return kwargs


def _result(
    prepared: PreparedHestonMarketFit,
    *,
    status: str,
    calibration_result: HestonMultistartResult | None,
    warnings: tuple[str, ...],
    errors: tuple[str, ...],
) -> HestonMarketFitResult:
    best_params = _best_params(calibration_result)
    summary = _summary(
        prepared,
        status=status,
        calibration_result=calibration_result,
        best_params=best_params,
        warnings=warnings,
        errors=errors,
    )
    return HestonMarketFitResult(
        model_name=prepared.model_name,
        status=status,
        prepared=prepared,
        calibration_result=calibration_result,
        best_params=best_params,
        summary=summary,
        warnings=warnings,
        errors=errors,
    )


def _summary(
    prepared: PreparedHestonMarketFit,
    *,
    status: str,
    calibration_result: HestonMultistartResult | None,
    best_params: HestonParams | None,
    warnings: tuple[str, ...],
    errors: tuple[str, ...],
) -> dict[str, Any]:
    preflight = prepared.preflight
    summary: dict[str, Any] = {
        "objective_type": prepared.objective_type,
        "input_quote_count": int(prepared.stats.input_quote_count),
        "selected_quote_count": int(prepared.stats.selected_quote_count),
        "rejected_quote_count": int(prepared.stats.rejected_quote_count),
        "rejection_counts": dict(prepared.stats.rejection_counts),
        "expiry_count": int(prepared.stats.expiry_count),
        "preflight_status": prepared.status,
        "preflight_recommendation": (
            None if preflight is None else preflight.recommendation
        ),
        "calibration_status": status,
        "best_objective_value": _best_objective_value(calibration_result),
        "best_params": _params_to_dict(best_params),
        "warning_count": len(warnings),
        "error_count": len(errors),
    }
    if calibration_result is not None:
        summary.update(
            {
                "calibration_quote_count": int(calibration_result.quote_count),
                "calibration_success_count": int(calibration_result.success_count),
                "calibration_failure_count": int(calibration_result.failure_count),
            }
        )
    underlying = _selected_underlying(prepared.selected_quotes)
    if underlying is not None:
        summary["underlying"] = underlying
    spot = getattr(prepared.market_data, "spot", None)
    if spot is not None:
        summary["spot"] = float(spot)
    return summary


def _best_params(
    calibration_result: HestonMultistartResult | None,
) -> HestonParams | None:
    if calibration_result is None:
        return None
    return calibration_result.best_params


def _best_objective_value(
    calibration_result: HestonMultistartResult | None,
) -> float | None:
    if calibration_result is None:
        return None
    best_run = getattr(calibration_result, "best_run", None)
    cost = getattr(best_run, "cost", None)
    if cost is None:
        return None
    return float(cost)


def _params_to_dict(params: object | None) -> dict[str, float] | None:
    if params is None:
        return None
    names = ("kappa", "vbar", "eta", "rho", "v")
    if not all(hasattr(params, name) for name in names):
        return None
    return {name: float(getattr(params, name)) for name in names}


def _failure_message(prepared: PreparedHestonMarketFit, detail: str) -> str:
    context = {
        "selected_quote_count": int(prepared.stats.selected_quote_count),
        "objective_type": prepared.objective_type,
        "preflight_status": prepared.status,
        "preflight_recommendation": (
            None if prepared.preflight is None else prepared.preflight.recommendation
        ),
        "underlying": _selected_underlying(prepared.selected_quotes),
        "spot": getattr(prepared.market_data, "spot", None),
    }
    rendered = ", ".join(
        f"{key}={value!r}" for key, value in context.items() if value is not None
    )
    return (
        f"Heston market fit failed ({rendered}): {detail}. "
        f"{_HESTON_FIT_WORKFLOW_GUIDANCE}"
    )


def _selected_underlying(selected_quotes: pd.DataFrame) -> str | None:
    if "underlying" not in selected_quotes.columns or selected_quotes.empty:
        return None
    values = selected_quotes["underlying"].dropna().astype(str).unique()
    if len(values) == 0:
        return None
    if len(values) == 1:
        return str(values[0])
    return ",".join(sorted(str(value) for value in values))


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = [
    "HestonCalibrationConfig",
    "HestonMarketFitError",
    "HestonMarketFitResult",
    "fit_heston_from_bundle",
    "fit_heston_market",
]
