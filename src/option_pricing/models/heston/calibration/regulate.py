"""Regularization helpers for Heston calibration.

This module provides soft regularization residuals that can be appended to
the quote-pricing residual vector used by scipy.optimize.least_squares.

The key distinction is:

- bounds are hard feasibility constraints;
- regularization residuals are soft modeling preferences.

For example, a calibration may be allowed to violate the Feller condition, but
the user may optionally penalize severe violations to prefer cleaner variance
processes when the fit quality trade-off is acceptable.
"""

from __future__ import annotations

from math import inf, isfinite, sqrt

import numpy as np

from ..params import HestonParams
from .heston_types import HestonRegConfig

_N_HESTON_PARAMS = 5
_KAPPA_IDX = 0
_VBAR_IDX = 1
_ETA_IDX = 2
_RHO_IDX = 3
_V0_IDX = 4


def _validate_nonnegative_weight(name: str, value: float) -> None:
    """Raise if a regularization weight is invalid."""
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative. Got {value!r}.")


def _validate_positive_threshold(name: str, value: float) -> None:
    """Raise if a soft threshold is invalid."""
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive. Got {value!r}.")


def validate_heston_reg_config(cfg: HestonRegConfig) -> None:
    """Validate a Heston regularization configuration.

    This is intentionally stricter than the residual helpers because bad
    regularization settings can silently distort calibration.
    """
    _validate_nonnegative_weight(
        "feller_penalty_weight",
        cfg.feller_penalty_weight,
    )
    _validate_nonnegative_weight(
        "rho_boundary_weight",
        cfg.rho_boundary_weight,
    )
    _validate_nonnegative_weight(
        "variance_level_weight",
        cfg.variance_level_weight,
    )
    _validate_nonnegative_weight(
        "vol_of_vol_weight",
        cfg.vol_of_vol_weight,
    )

    _validate_positive_threshold("feller_ratio_target", cfg.feller_ratio_target)
    _validate_positive_threshold("vbar_soft_max", cfg.vbar_soft_max)
    _validate_positive_threshold("v0_soft_max", cfg.v0_soft_max)
    _validate_positive_threshold("eta_soft_max", cfg.eta_soft_max)
    _validate_positive_threshold("eps", cfg.eps)

    if not isfinite(cfg.rho_abs_soft_limit):
        raise ValueError(
            "rho_abs_soft_limit must be finite. " f"Got {cfg.rho_abs_soft_limit!r}."
        )

    if not 0.0 <= cfg.rho_abs_soft_limit < 1.0:
        raise ValueError(
            "rho_abs_soft_limit must satisfy 0 <= limit < 1. "
            f"Got {cfg.rho_abs_soft_limit!r}."
        )


def _get_param(params: HestonParams, name: str) -> float:
    """Read a Heston parameter from HestonParams or a mapping."""
    if isinstance(params, HestonParams):
        return float(getattr(params, name))
    return float(params[name])


def _positive_part(x: float) -> float:
    """Return max(0, x) as a small helper for hinge-style penalties."""
    return max(0.0, float(x))


def _soft_upper_violation(
    value: float,
    soft_max: float,
    *,
    scale: float | None = None,
    eps: float = 1e-12,
) -> float:
    """Return normalized violation above a soft upper threshold.

    Examples
    --------
    If value <= soft_max, the violation is zero.

    If value > soft_max, the violation is:

        (value - soft_max) / scale

    By default, scale is soft_max, so the result is dimensionless.

    This is a hinge penalty, not a hard bound. It does not prevent the
    optimizer from entering the region; it only adds a residual there.
    """
    value = float(value)
    soft_max = float(soft_max)

    if scale is None:
        scale = soft_max

    scale = max(float(scale), eps)

    return _positive_part(value - soft_max) / scale


def _soft_lower_violation(
    value: float,
    soft_min: float,
    *,
    scale: float | None = None,
    eps: float = 1e-12,
) -> float:
    """Return normalized violation below a soft lower threshold."""
    value = float(value)
    soft_min = float(soft_min)

    if scale is None:
        scale = abs(soft_min) if soft_min != 0.0 else 1.0

    scale = max(float(scale), eps)

    return _positive_part(soft_min - value) / scale


def _feller_ratio(params: HestonParams) -> float:
    """Return the Heston Feller ratio.

    The Feller condition is:

        2 * kappa * vbar >= eta**2

    so this returns:

        2 * kappa * vbar / eta**2

    A value >= 1.0 means the usual Feller condition is satisfied.

    Returns +inf when eta == 0 and the numerator is nonnegative.
    """
    kappa = _get_param(params, "kappa")
    vbar = _get_param(params, "vbar")
    eta = _get_param(params, "eta")

    numerator = 2.0 * kappa * vbar

    if eta == 0.0:
        return inf if numerator >= 0.0 else -inf

    return numerator / (eta * eta)


def _feller_violation(
    params: HestonParams,
    *,
    target: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """Return normalized violation of the Feller ratio target.

    The default target is 1.0, corresponding to:

        2 * kappa * vbar / eta**2 >= 1

    The violation is:

        max(0, target - ratio) / target

    This returns zero when the Feller ratio is above the target.
    """
    target = max(float(target), eps)
    ratio = _feller_ratio(params)

    if ratio == inf:
        return 0.0

    if not isfinite(ratio):
        return inf

    return _positive_part(target - ratio) / target


def _rho_boundary_violation(
    params: HestonParams,
    *,
    rho_abs_soft_limit: float = 0.95,
    eps: float = 1e-12,
) -> float:
    """Return normalized soft-boundary violation for rho.

    No penalty is applied while:

        abs(rho) <= rho_abs_soft_limit

    Above that, the violation ramps linearly to 1 as abs(rho) approaches 1.
    """
    if not 0.0 <= rho_abs_soft_limit < 1.0:
        raise ValueError(
            "rho_abs_soft_limit must satisfy 0 <= limit < 1. "
            f"Got {rho_abs_soft_limit!r}."
        )

    rho = _get_param(params, "rho")
    denom = max(1.0 - rho_abs_soft_limit, eps)

    return _positive_part(abs(rho) - rho_abs_soft_limit) / denom


def _variance_level_violations(
    params: HestonParams,
    *,
    vbar_soft_max: float = 1.0,
    v0_soft_max: float = 1.0,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Return soft upper violations for long-run and initial variance.

    These are variance thresholds, not volatility thresholds.

    For example:

        vbar_soft_max = 1.0

    corresponds to a 100% long-run volatility level.
    """
    vbar = _get_param(params, "vbar")
    v0 = _get_param(params, "v")

    vbar_violation = _soft_upper_violation(
        vbar,
        vbar_soft_max,
        eps=eps,
    )
    v0_violation = _soft_upper_violation(
        v0,
        v0_soft_max,
        eps=eps,
    )

    return vbar_violation, v0_violation


def _vol_of_vol_violation(
    params: HestonParams,
    *,
    eta_soft_max: float = 3.0,
    eps: float = 1e-12,
) -> float:
    """Return soft upper violation for Heston vol-of-vol eta."""
    eta = _get_param(params, "eta")

    return _soft_upper_violation(
        eta,
        eta_soft_max,
        eps=eps,
    )


def _empty_reg_jacobian() -> np.ndarray:
    return np.empty((0, _N_HESTON_PARAMS), dtype=float)


def _feller_violation_grad(
    params: HestonParams,
    *,
    target: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return constrained-parameter gradient of the Feller violation."""
    grad = np.zeros(_N_HESTON_PARAMS, dtype=float)
    target = max(float(target), eps)

    kappa = _get_param(params, "kappa")
    vbar = _get_param(params, "vbar")
    eta = _get_param(params, "eta")

    if eta <= 0.0:
        return grad

    eta_sq = eta * eta
    ratio = 2.0 * kappa * vbar / eta_sq
    if not isfinite(ratio) or ratio >= target:
        return grad

    grad[_KAPPA_IDX] = -(2.0 * vbar / eta_sq) / target
    grad[_VBAR_IDX] = -(2.0 * kappa / eta_sq) / target
    grad[_ETA_IDX] = (4.0 * kappa * vbar / (eta_sq * eta)) / target
    return grad


def _rho_boundary_violation_grad(
    params: HestonParams,
    *,
    rho_abs_soft_limit: float = 0.95,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return constrained-parameter gradient of the rho boundary violation."""
    grad = np.zeros(_N_HESTON_PARAMS, dtype=float)
    if not 0.0 <= rho_abs_soft_limit < 1.0:
        raise ValueError(
            "rho_abs_soft_limit must satisfy 0 <= limit < 1. "
            f"Got {rho_abs_soft_limit!r}."
        )

    rho = _get_param(params, "rho")
    if abs(rho) <= rho_abs_soft_limit:
        return grad

    denom = max(1.0 - rho_abs_soft_limit, eps)
    grad[_RHO_IDX] = (1.0 if rho > 0.0 else -1.0) / denom
    return grad


def _soft_upper_violation_grad(
    value: float,
    soft_max: float,
    *,
    scale: float | None = None,
    eps: float = 1e-12,
) -> float:
    """Return derivative of a normalized soft upper violation."""
    value = float(value)
    soft_max = float(soft_max)

    if value <= soft_max:
        return 0.0

    if scale is None:
        scale = soft_max

    return 1.0 / max(float(scale), eps)


def heston_regularization_jacobian(
    params: HestonParams,
    cfg: HestonRegConfig | None,
) -> np.ndarray:
    """Return regularization residual Jacobian with respect to Heston params.

    Rows are ordered to match ``heston_regularization_residuals``. Columns are
    ordered as ``[kappa, vbar, eta, rho, v]``.
    """
    if cfg is None:
        return _empty_reg_jacobian()

    validate_heston_reg_config(cfg)

    rows: list[np.ndarray] = []

    if cfg.feller_penalty_weight > 0.0:
        rows.append(
            sqrt(cfg.feller_penalty_weight)
            * _feller_violation_grad(
                params,
                target=cfg.feller_ratio_target,
                eps=cfg.eps,
            )
        )

    if cfg.rho_boundary_weight > 0.0:
        rows.append(
            sqrt(cfg.rho_boundary_weight)
            * _rho_boundary_violation_grad(
                params,
                rho_abs_soft_limit=cfg.rho_abs_soft_limit,
                eps=cfg.eps,
            )
        )

    if cfg.variance_level_weight > 0.0:
        sqrt_weight = sqrt(cfg.variance_level_weight)

        vbar_grad = np.zeros(_N_HESTON_PARAMS, dtype=float)
        vbar_grad[_VBAR_IDX] = _soft_upper_violation_grad(
            _get_param(params, "vbar"),
            cfg.vbar_soft_max,
            eps=cfg.eps,
        )
        rows.append(sqrt_weight * vbar_grad)

        v0_grad = np.zeros(_N_HESTON_PARAMS, dtype=float)
        v0_grad[_V0_IDX] = _soft_upper_violation_grad(
            _get_param(params, "v"),
            cfg.v0_soft_max,
            eps=cfg.eps,
        )
        rows.append(sqrt_weight * v0_grad)

    if cfg.vol_of_vol_weight > 0.0:
        eta_grad = np.zeros(_N_HESTON_PARAMS, dtype=float)
        eta_grad[_ETA_IDX] = _soft_upper_violation_grad(
            _get_param(params, "eta"),
            cfg.eta_soft_max,
            eps=cfg.eps,
        )
        rows.append(sqrt(cfg.vol_of_vol_weight) * eta_grad)

    if not rows:
        return _empty_reg_jacobian()

    return np.asarray(rows, dtype=float)


def heston_regularization_violations(
    params: HestonParams,
    cfg: HestonRegConfig | None,
) -> dict[str, float]:
    """Return unweighted, dimensionless regularization violations.

    This is useful for diagnostics and reporting. These values are not yet
    multiplied by sqrt(weight).

    The returned dictionary always contains the same keys.
    """
    if cfg is None:
        cfg = HestonRegConfig()

    validate_heston_reg_config(cfg)

    vbar_violation, v0_violation = _variance_level_violations(
        params,
        vbar_soft_max=cfg.vbar_soft_max,
        v0_soft_max=cfg.v0_soft_max,
        eps=cfg.eps,
    )

    return {
        "feller": _feller_violation(
            params,
            target=cfg.feller_ratio_target,
            eps=cfg.eps,
        ),
        "rho_boundary": _rho_boundary_violation(
            params,
            rho_abs_soft_limit=cfg.rho_abs_soft_limit,
            eps=cfg.eps,
        ),
        "vbar_level": vbar_violation,
        "v0_level": v0_violation,
        "vol_of_vol": _vol_of_vol_violation(
            params,
            eta_soft_max=cfg.eta_soft_max,
            eps=cfg.eps,
        ),
    }


def heston_regularization_residuals(
    params: HestonParams,
    cfg: HestonRegConfig | None,
) -> np.ndarray:
    """Return weighted regularization residuals for least-squares calibration.

    These residuals are intended to be appended to the quote residual vector.

    Example
    -------
    In the calibration objective:

        quote_residuals = ...
        reg_residuals = heston_regularization_residuals(params, cfg)
        return np.concatenate([quote_residuals, reg_residuals])

    The residual vector includes only active regularizers, i.e. regularizers
    whose weights are strictly positive.
    """
    if cfg is None:
        return np.empty(0, dtype=float)

    validate_heston_reg_config(cfg)

    violations = heston_regularization_violations(params, cfg)
    residuals: list[float] = []

    if cfg.feller_penalty_weight > 0.0:
        residuals.append(sqrt(cfg.feller_penalty_weight) * violations["feller"])

    if cfg.rho_boundary_weight > 0.0:
        residuals.append(sqrt(cfg.rho_boundary_weight) * violations["rho_boundary"])

    if cfg.variance_level_weight > 0.0:
        sqrt_weight = sqrt(cfg.variance_level_weight)
        residuals.append(sqrt_weight * violations["vbar_level"])
        residuals.append(sqrt_weight * violations["v0_level"])

    if cfg.vol_of_vol_weight > 0.0:
        residuals.append(sqrt(cfg.vol_of_vol_weight) * violations["vol_of_vol"])

    return np.asarray(residuals, dtype=float)


def feller_ratio(params: HestonParams) -> float:
    """Public diagnostic wrapper for the Heston Feller ratio."""
    return _feller_ratio(params)


def feller_satisfied(
    params: HestonParams,
    *,
    target: float = 1.0,
) -> bool:
    """Return whether the Feller ratio is at or above the target."""
    return _feller_ratio(params) >= target
