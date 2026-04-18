"""
Heston Fourier inversion and probability integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from ...numerics.quadrature import (
    CompositeRule,
    QuadratureConfig,
    build_gauss_legendre_rule,
    integrate_composite_rule,
)
from ...typing import ArrayLike, FloatArray
from .charfunc import (
    _heston_affine_coeffs,
    _normalize_frequency_grid,
    _restore_frequency_shape,
)
from .params import HestonParams

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]
type Backend = Literal["gauss_legendre", "quad"]
type J = Literal[0, 1]


@dataclass(frozen=True, slots=True)
class HestonIntegralDiagnostics:
    backend: Backend
    j: J
    x: float
    tau: float
    total_integral: float
    probability: float
    panel_contribs: FloatArray | None = None
    panel_edges: FloatArray | None = None
    quad_error_estimate: float | None = None


def _pj_affine_factor(
    u: float | np.ndarray,
    tau: float,
    params: HestonParams,
    *,
    j: J,
) -> complex | ComplexArray:
    u_arr, scalar_input, original_shape = _normalize_frequency_grid(u)
    C, D = _heston_affine_coeffs(u_arr, tau, params, j=j)
    values = np.exp(C * params.vbar + D * params.v)
    return _restore_frequency_shape(
        values,
        scalar_input=scalar_input,
        original_shape=original_shape,
    )


def _integrand(
    u: ArrayLike,
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: J,
) -> float | RealArray:
    u_arr = np.asarray(u, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must be finite.")

    affine_factor = np.asarray(
        _pj_affine_factor(u_arr, tau, params, j=j),
        dtype=np.complex128,
    )

    if u_arr.ndim == 0 and x_arr.ndim == 0:
        value = np.real(np.exp(1j * u_arr * x_arr) * affine_factor / (1j * u_arr))
        return float(value)

    if u_arr.ndim == 0:
        values = np.real(np.exp(1j * u_arr * x_arr) * affine_factor / (1j * u_arr))
        return np.asarray(values, dtype=np.float64)

    if x_arr.ndim == 0:
        values = np.real(np.exp(1j * u_arr * x_arr) * affine_factor / (1j * u_arr))
        return np.asarray(values, dtype=np.float64)

    u_b = u_arr.reshape(u_arr.shape + (1,) * x_arr.ndim)  # TODO: Check corr functioning
    x_b = x_arr.reshape((1,) * u_arr.ndim + x_arr.shape)
    affine_b = affine_factor.reshape(affine_factor.shape + (1,) * x_arr.ndim)

    values = np.real(np.exp(1j * u_b * x_b) * affine_b / (1j * u_b))
    return np.asarray(values, dtype=np.float64)


def _integrand_scalar(
    u: float,
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
) -> float:
    return float(_integrand(u=u, x=x, tau=tau, params=params, j=j))


def _probability_from_integral(integral: float) -> float:
    return float(0.5 + integral / np.pi)


def _integrate_pj_quad(
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
) -> tuple[float, float]:
    integral, err_est = quad(
        _integrand_scalar,
        a=0.0,
        b=np.inf,
        args=(x, tau, params, j),
        complex_func=False,
    )
    return float(integral), float(err_est)


def _default_heston_quadrature_config() -> QuadratureConfig:
    return QuadratureConfig(
        u_max=150.0,
        n_panels=24,
        nodes_per_panel=16,
    )  # TODO: CHECK IF DEFAULTS ARE ACTUALLY VALID


def _build_heston_gauss_rule(cfg: QuadratureConfig) -> CompositeRule:
    return build_gauss_legendre_rule(cfg)


def _resolve_gauss_rule(
    *,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
) -> CompositeRule:
    if quad_cfg is not None and rule is not None:
        raise ValueError(
            "Pass either quad_cfg or rule, not both. "
            "If you already built a rule, it is the authoritative discretization."
        )

    if rule is not None:
        return rule

    cfg = quad_cfg or _default_heston_quadrature_config()
    return _build_heston_gauss_rule(cfg)


def _integrate_pj_fixed_rule(
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
    rule: CompositeRule,
) -> float:
    result = integrate_composite_rule(
        lambda u: _integrand(u=u, x=x, tau=tau, params=params, j=j),
        rule,
    )
    return float(result.total)


def _integrate_pj_fixed_rule_with_diagnostics(
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
    rule: CompositeRule,
) -> HestonIntegralDiagnostics:
    result = integrate_composite_rule(
        lambda u: _integrand(u=u, x=x, tau=tau, params=params, j=j),
        rule,
    )
    probability = _probability_from_integral(result.total)

    return HestonIntegralDiagnostics(
        backend="gauss_legendre",
        j=j,
        x=x,
        tau=tau,
        total_integral=float(result.total),
        probability=probability,
        panel_contribs=result.panel_contribs,
        panel_edges=rule.panel_edges,
        quad_error_estimate=None,
    )


def P_j_Scalar(
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
    backend: Backend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float:
    if backend == "quad":
        integral, _ = _integrate_pj_quad(x, tau, params, j)
        return _probability_from_integral(integral)

    if backend == "gauss_legendre":
        active_rule = _resolve_gauss_rule(quad_cfg=quad_cfg, rule=rule)
        integral = _integrate_pj_fixed_rule(x, tau, params, j, active_rule)
        return _probability_from_integral(integral)

    raise ValueError(f"Unknown backend: {backend}")


def P_j_with_diagnostics(
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
    backend: Backend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> HestonIntegralDiagnostics:
    if backend == "quad":
        integral, err_est = _integrate_pj_quad(x, tau, params, j)
        probability = _probability_from_integral(integral)
        return HestonIntegralDiagnostics(
            backend="quad",
            j=j,
            x=x,
            tau=tau,
            total_integral=float(integral),
            probability=probability,
            panel_contribs=None,
            panel_edges=None,
            quad_error_estimate=float(err_est),
        )

    if backend == "gauss_legendre":
        active_rule = _resolve_gauss_rule(quad_cfg=quad_cfg, rule=rule)
        return _integrate_pj_fixed_rule_with_diagnostics(
            x=x,
            tau=tau,
            params=params,
            j=j,
            rule=active_rule,
        )

    raise ValueError(f"Unknown backend: {backend}")
