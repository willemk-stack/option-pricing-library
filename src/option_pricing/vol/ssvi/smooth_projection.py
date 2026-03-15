from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...market.curves import FlatCarryForwardCurve, FlatDiscountCurve, PricingContext
from ...typing import ArrayLike
from ..local_vol_gatheral import gatheral_local_var_diagnostics
from .models import (
    ESSVINodeSet,
    ESSVITermStructures,
    EtaTermStructure,
    PsiTermStructure,
    ThetaTermStructure,
)
from .surface import ESSVINodalSurface, ESSVISmoothedSurface
from .validation import (
    ESSVIValidationReport,
    validate_essvi_continuous,
    validate_essvi_nodes,
)


def _validation_expiries(expiries: np.ndarray, n: int) -> np.ndarray:
    exp = np.unique(np.asarray(expiries, dtype=np.float64))
    if exp.size:
        short = exp[0] * np.asarray([0.25, 0.5, 0.75], dtype=np.float64)
        exp = np.unique(np.concatenate([short, exp])).astype(np.float64)
    if exp.size == 1 or n <= exp.size:
        return exp
    dense = np.linspace(float(exp[0]), float(exp[-1]), int(n), dtype=np.float64)
    return np.unique(np.concatenate([exp, dense])).astype(np.float64)


def _unit_market() -> PricingContext:
    return PricingContext(
        spot=1.0,
        discount=FlatDiscountCurve(0.0),
        forward=FlatCarryForwardCurve(1.0, 0.0, 0.0),
    )


def _build_log_term(
    expiries: np.ndarray,
    values: np.ndarray,
) -> tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]:
    from ...numerics.interpolation import FritschCarlson

    x = np.log(np.asarray(expiries, dtype=np.float64))
    y = np.asarray(values, dtype=np.float64)

    if x.size == 1:
        y0 = float(y[0])
        return (
            lambda log_T: np.full_like(
                np.asarray(log_T, dtype=np.float64), y0, dtype=np.float64
            ),
            lambda log_T: np.zeros_like(
                np.asarray(log_T, dtype=np.float64), dtype=np.float64
            ),
        )

    value_fn, deriv_fn = FritschCarlson(x, y)
    return (
        lambda log_T: np.asarray(
            value_fn(np.asarray(log_T, dtype=np.float64)), dtype=np.float64
        ),
        lambda log_T: np.asarray(
            deriv_fn(np.asarray(log_T, dtype=np.float64)), dtype=np.float64
        ),
    )


def _build_term_with_origin_anchor(
    expiries: np.ndarray,
    values: np.ndarray,
) -> tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]:
    log_value, log_deriv = _build_log_term(expiries, values)
    first_expiry = float(np.asarray(expiries, dtype=np.float64)[0])
    first_value = float(np.asarray(values, dtype=np.float64)[0])
    short_slope = first_value / first_expiry

    def value_fn(T: ArrayLike) -> np.ndarray:
        T_arr = np.asarray(T, dtype=np.float64)
        out = np.empty_like(T_arr, dtype=np.float64)
        short_mask = T_arr < first_expiry
        if np.any(short_mask):
            out[short_mask] = short_slope * T_arr[short_mask]
        if np.any(~short_mask):
            out[~short_mask] = log_value(np.log(T_arr[~short_mask]))
        return out

    def deriv_fn(T: ArrayLike) -> np.ndarray:
        T_arr = np.asarray(T, dtype=np.float64)
        out = np.empty_like(T_arr, dtype=np.float64)
        short_mask = T_arr < first_expiry
        if np.any(short_mask):
            out[short_mask] = short_slope
        if np.any(~short_mask):
            long_T = T_arr[~short_mask]
            out[~short_mask] = log_deriv(np.log(long_T)) / long_T
        return out

    return value_fn, deriv_fn


@dataclass(frozen=True, slots=True)
class ESSVIProjectionConfig:
    validation_nt: int = 41
    validation_y_min: float = -2.5
    validation_y_max: float = 2.5
    validation_ny: int = 81
    dupire_nt: int = 21
    dupire_y_min: float = -1.5
    dupire_y_max: float = 1.5
    dupire_ny: int = 41
    eps_w: float = 1e-12
    eps_denom: float = 1e-12
    strict_validation: bool = False

    def __post_init__(self) -> None:
        if self.validation_nt <= 0 or self.dupire_nt <= 0:
            raise ValueError("validation_nt and dupire_nt must be > 0.")
        if self.validation_ny < 3 or self.dupire_ny < 3:
            raise ValueError("validation_ny and dupire_ny must be >= 3.")
        if self.validation_y_min >= self.validation_y_max:
            raise ValueError("validation_y_min must be < validation_y_max.")
        if self.dupire_y_min >= self.dupire_y_max:
            raise ValueError("dupire_y_min must be < dupire_y_max.")


@dataclass(frozen=True, slots=True)
class ESSVIProjectionDiagnostics:
    node_validation_ok: bool
    validation_expiries: np.ndarray
    validation_y_grid: np.ndarray
    validation: ESSVIValidationReport | None
    dupire_expiries: np.ndarray
    dupire_y_grid: np.ndarray
    dupire_invalid_count: int
    dupire_total_points: int
    message: str


@dataclass(frozen=True, slots=True)
class ESSVIProjectionResult:
    success: bool
    nodes: ESSVINodeSet
    params: ESSVITermStructures | None
    surface: ESSVISmoothedSurface | None
    fallback_surface: ESSVINodalSurface
    diag: ESSVIProjectionDiagnostics


def project_essvi_nodes(
    nodes: ESSVINodeSet,
    *,
    cfg: ESSVIProjectionConfig | None = None,
) -> ESSVIProjectionResult:
    cfg = ESSVIProjectionConfig() if cfg is None else cfg
    fallback_surface = ESSVINodalSurface(nodes=nodes)

    node_report = validate_essvi_nodes(nodes)
    validation_expiries = _validation_expiries(nodes.expiries, cfg.validation_nt)
    validation_y_grid = np.linspace(
        float(cfg.validation_y_min),
        float(cfg.validation_y_max),
        int(cfg.validation_ny),
        dtype=np.float64,
    )
    dupire_expiries = _validation_expiries(nodes.expiries, cfg.dupire_nt)
    dupire_y_grid = np.linspace(
        float(cfg.dupire_y_min),
        float(cfg.dupire_y_max),
        int(cfg.dupire_ny),
        dtype=np.float64,
    )

    try:
        if np.any(np.diff(nodes.theta) < 0.0):
            raise ValueError(
                "theta nodes must be nondecreasing for monotone projection."
            )
        if np.any(np.diff(nodes.g_plus) < 0.0):
            raise ValueError(
                "g_plus nodes must be nondecreasing for monotone projection."
            )
        if np.any(np.diff(nodes.g_minus) < 0.0):
            raise ValueError(
                "g_minus nodes must be nondecreasing for monotone projection."
            )

        theta_value, theta_first = _build_term_with_origin_anchor(
            nodes.expiries, nodes.theta
        )
        g_plus_value, g_plus_first = _build_term_with_origin_anchor(
            nodes.expiries, nodes.g_plus
        )
        g_minus_value, g_minus_first = _build_term_with_origin_anchor(
            nodes.expiries, nodes.g_minus
        )

        def psi_value(log_T):
            return 0.5 * (g_plus_value(log_T) + g_minus_value(log_T))

        def psi_first(log_T):
            return 0.5 * (g_plus_first(log_T) + g_minus_first(log_T))

        def eta_value(log_T):
            return 0.5 * (g_plus_value(log_T) - g_minus_value(log_T))

        def eta_first(log_T):
            return 0.5 * (g_plus_first(log_T) - g_minus_first(log_T))

        params = ESSVITermStructures(
            theta_term=ThetaTermStructure(
                value=theta_value,
                first_derivative=theta_first,
                input_variable="T",
            ),
            psi_term=PsiTermStructure(
                value=psi_value,
                first_derivative=psi_first,
                input_variable="T",
            ),
            eta_term=EtaTermStructure(
                value=eta_value,
                first_derivative=eta_first,
                input_variable="T",
            ),
            eps=nodes.eps,
        )
        surface = ESSVISmoothedSurface(params=params)
        validation = validate_essvi_continuous(
            params,
            _unit_market(),
            expiries=validation_expiries,
            y_grid=validation_y_grid,
            strict=False,
            tol=nodes.eps,
        )

        dupire_invalid_count = 0
        dupire_total_points = int(dupire_expiries.size * dupire_y_grid.size)
        for tau in dupire_expiries:
            y = np.asarray(dupire_y_grid, dtype=np.float64)
            w, w_y, w_yy, w_T = surface.w_and_derivs(y, float(tau))
            dupire_invalid_count += int(
                gatheral_local_var_diagnostics(
                    y=y,
                    w=w,
                    w_y=w_y,
                    w_yy=w_yy,
                    w_T=w_T,
                    eps_w=cfg.eps_w,
                    eps_denom=cfg.eps_denom,
                ).invalid_count
            )

        success = bool(validation.ok and dupire_invalid_count == 0)
        message = (
            "OK"
            if success
            else f"validation_ok={validation.ok}, dupire_invalid_count={dupire_invalid_count}"
        )
        diag = ESSVIProjectionDiagnostics(
            node_validation_ok=bool(node_report.ok),
            validation_expiries=validation_expiries,
            validation_y_grid=validation_y_grid,
            validation=validation,
            dupire_expiries=dupire_expiries,
            dupire_y_grid=dupire_y_grid,
            dupire_invalid_count=dupire_invalid_count,
            dupire_total_points=dupire_total_points,
            message=message,
        )
        if cfg.strict_validation and not success:
            raise ValueError(message)
        return ESSVIProjectionResult(
            success=success,
            nodes=nodes,
            params=params if success else None,
            surface=surface if success else None,
            fallback_surface=fallback_surface,
            diag=diag,
        )
    except Exception as exc:
        diag = ESSVIProjectionDiagnostics(
            node_validation_ok=bool(node_report.ok),
            validation_expiries=validation_expiries,
            validation_y_grid=validation_y_grid,
            validation=None,
            dupire_expiries=dupire_expiries,
            dupire_y_grid=dupire_y_grid,
            dupire_invalid_count=0,
            dupire_total_points=int(dupire_expiries.size * dupire_y_grid.size),
            message=str(exc),
        )
        if cfg.strict_validation:
            raise
        return ESSVIProjectionResult(
            success=False,
            nodes=nodes,
            params=None,
            surface=None,
            fallback_surface=fallback_surface,
            diag=diag,
        )
