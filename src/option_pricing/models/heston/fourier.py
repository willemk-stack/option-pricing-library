"""
Heston Fourier inversion and probability integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from ...numerics.quadrature import (
    CompositeRule,
    PanelSpacing,
    QuadratureConfig,
    build_gauss_legendre_rule,
)
from ...typing import ArrayLike, FloatArray
from .charfunc import (
    _cui_char_fn_and_param_grad,
    _heston_affine_coeffs,
    _normalize_frequency_grid,
    _restore_frequency_shape,
)
from .params import HestonParams

type QuadratureQuality = Literal["fast", "balanced", "robust", "diagnostics"]
type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type UInt32Array = NDArray[np.uint32]
type HestonBackend = Literal["gauss_legendre", "quad"]
type HestonProbabilityIndex = Literal[0, 1]
type Backend = HestonBackend
type J = HestonProbabilityIndex


# ---------------------------------------------------------------------------
# Diagnostics heuristics.
#
# These are intentionally module-level knobs to be tuned during Sprint 2
# without changing the public API. Might be a good idea to promote to a typed
# diagnostics settings object later.
# ---------------------------------------------------------------------------
HESTON_DIAG_PROBABILITY_TOL = 1.0e-10
HESTON_DIAG_TAIL_PANEL_COUNT = 3
HESTON_DIAG_TAIL_ABS_FRACTION_WARN = 5.0e-2
HESTON_DIAG_CANCELLATION_RATIO_WARN = 5.0e1
HESTON_DIAG_BAD_PANEL_COUNT_WARN = 5
HESTON_DIAG_BAD_PANEL_FRACTION_WARN = 1.0e-1
HESTON_DIAG_QUAD_REL_ERROR_WARN = 1.0e-6
HESTON_DIAG_NEAR_ORIGIN_PANEL_COUNT = 2
HESTON_DIAG_NEAR_ORIGIN_ABS_FRACTION_WARN = 6.0e-1
HESTON_DIAG_TAIL_LAST_OVER_PREV_WARN = 8.0e-1
HESTON_DIAG_OSCILLATION_SPIKE_FACTOR = 5.0
HESTON_DIAG_OSCILLATION_ABS_MASS_FLOOR = 1.0e-4
HESTON_DIAG_OSCILLATION_REL_MASS_FLOOR = 1.0e-3
_DIAG_EPS = 1.0e-14


class HestonPanelReason(IntFlag):
    NONE = 0
    NONFINITE_PANEL_CONTRIB = 1 << 0
    NONFINITE_INTEGRAND = 1 << 1
    UNDERRESOLVED_NEAR_ORIGIN = 1 << 2
    UNDERRESOLVED_TAIL = 1 << 3
    TAIL_TOO_LARGE = 1 << 4
    OSCILLATION_SPIKE = 1 << 5


class HestonIntegralWarning(IntFlag):
    NONE = 0
    NONFINITE_TOTAL = 1 << 0
    NONFINITE_PROBABILITY = 1 << 1
    PROBABILITY_OUT_OF_RANGE = 1 << 2
    LARGE_TAIL_FRACTION = 1 << 3
    EXCESSIVE_CANCELLATION = 1 << 4
    TOO_MANY_BAD_PANELS = 1 << 5
    QUAD_ERROR_LARGE = 1 << 6


@dataclass(frozen=True, slots=True)
class HestonIntegralDiagnostics:
    # configuration / identity
    backend: HestonBackend
    quad_cfg: QuadratureConfig | None
    j: HestonProbabilityIndex
    x: float
    tau: float

    # raw result
    total_integral: float
    probability: float

    # raw backend outputs
    panel_edges: FloatArray | None = None
    panel_contribs: FloatArray | None = None
    quad_error_estimate: float | None = None

    # panel-local diagnostics
    panel_invalid: BoolArray | None = None
    panel_reason: UInt32Array | None = None

    # whole-integral diagnostics
    warning_flags: HestonIntegralWarning = HestonIntegralWarning.NONE
    tail_abs_fraction: float | None = None
    cancellation_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class HestonIntegralBatchDiagnostics:
    """Batch diagnostics container for Heston probability integrals.

    Notes
    -----
    Leading shape matches ``x.shape``. Panel-local arrays append an extra
    trailing ``n_panels`` axis when the backend is fixed-rule Gauss-Legendre.
    For ``backend="quad"``, panel-local fields are ``None``.
    """

    # configuration / identity
    backend: HestonBackend
    quad_cfg: QuadratureConfig | None
    j: HestonProbabilityIndex
    x: RealArray  # shape batch_shape
    tau: float

    # raw result
    total_integral: RealArray  # shape batch_shape
    probability: RealArray  # shape batch_shape

    # raw backend outputs
    panel_edges: RealArray | None = None  # shape (n_panels + 1,)
    panel_contribs: RealArray | None = None  # shape batch_shape + (n_panels,)
    quad_error_estimate: RealArray | None = None  # shape batch_shape

    # panel-local diagnostics
    panel_invalid: BoolArray | None = None  # shape batch_shape + (n_panels,)
    panel_reason: UInt32Array | None = None  # shape batch_shape + (n_panels,)

    # whole-integral diagnostics
    warning_flags: UInt32Array | None = None  # shape batch_shape
    tail_abs_fraction: RealArray | None = None  # shape batch_shape
    cancellation_ratio: RealArray | None = None  # shape batch_shape


def _flag_u32(flag: IntFlag) -> np.uint32:
    return np.uint32(int(flag))


def _normalize_x_grid(
    x: ArrayLike,
) -> tuple[RealArray, bool, tuple[int, ...]]:
    x_arr = np.asarray(x, dtype=np.float64)
    scalar_input = x_arr.ndim == 0
    original_shape = x_arr.shape
    x_flat = np.asarray(x_arr.reshape(-1), dtype=np.float64)

    if not np.all(np.isfinite(x_flat)):
        raise ValueError("x must be finite.")

    return x_flat, scalar_input, original_shape


def _restore_x_shape(
    values: ArrayLike,
    *,
    scalar_input: bool,
    original_shape: tuple[int, ...],
) -> float | RealArray:
    values_arr = np.asarray(values, dtype=np.float64)
    if scalar_input:
        return float(values_arr.reshape(-1)[0])
    return np.asarray(values_arr.reshape(original_shape), dtype=np.float64)


def _pj_affine_factor(
    u: float | np.ndarray,
    tau: float,
    params: HestonParams,
    *,
    j: HestonProbabilityIndex,
) -> complex | ComplexArray:
    u_arr, scalar_input, original_shape = _normalize_frequency_grid(u)
    C, D = _heston_affine_coeffs(u_arr, tau, params, j=j)
    values = np.exp(C * params.vbar + D * params.v)
    return _restore_frequency_shape(
        values,
        scalar_input=scalar_input,
        original_shape=original_shape,
    )


def _pj_affine_factor_and_param_jac(
    u: float | np.ndarray,
    tau: float,
    params: HestonParams,
    *,
    j: HestonProbabilityIndex,
) -> tuple[complex | ComplexArray, ComplexArray]:
    u_arr, scalar_input, original_shape = _normalize_frequency_grid(u)

    if j == 0:
        gradient_frequency = u_arr
    elif j == 1:
        gradient_frequency = u_arr - 1j
    else:
        raise ValueError("j must be either 0 or 1.")

    # REVIEW: confirm the Phase 3 charfunc helper's x/phase convention before wiring this into production pricing.
    affine, d_affine = _cui_char_fn_and_param_grad(
        gradient_frequency,
        tau,
        params,
        x=0.0,
    )
    affine_arr = np.asarray(affine, dtype=np.complex128).reshape(-1)
    d_affine_arr = np.asarray(d_affine, dtype=np.complex128).reshape(-1, 5)

    if scalar_input:
        return complex(affine_arr[0]), np.asarray(d_affine_arr[0], dtype=np.complex128)

    return (
        np.asarray(affine_arr.reshape(original_shape), dtype=np.complex128),
        np.asarray(d_affine_arr.reshape(original_shape + (5,)), dtype=np.complex128),
    )


def _integrand(
    u: ArrayLike,
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
) -> float | RealArray:
    """Evaluate the real-valued Heston inversion integrand."""
    x_flat, x_scalar, x_shape = _normalize_x_grid(x)
    u_flat, u_scalar, u_shape = _normalize_frequency_grid(u)

    affine = np.asarray(
        _pj_affine_factor(u_flat, tau, params, j=j), dtype=np.complex128
    )

    values_2d = np.real(
        np.exp(1j * x_flat[:, None] * u_flat[None, :])
        * affine[None, :]
        / (1j * u_flat[None, :])
    )

    if x_scalar and u_scalar:
        return float(values_2d[0, 0])
    if x_scalar:
        return np.asarray(values_2d[0, :].reshape(u_shape), dtype=np.float64)
    if u_scalar:
        return np.asarray(values_2d[:, 0].reshape(x_shape), dtype=np.float64)

    return np.asarray(values_2d.reshape(x_shape + u_shape), dtype=np.float64)


def _integrand_and_param_jac(
    u: ArrayLike,
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
) -> tuple[float | RealArray, RealArray]:
    """Evaluate the real-valued Heston inversion integrand and parameter Jacobian."""
    x_flat, x_scalar, x_shape = _normalize_x_grid(x)
    u_flat, u_scalar, u_shape = _normalize_frequency_grid(u)

    affine, d_affine = _pj_affine_factor_and_param_jac(
        u_flat,
        tau,
        params,
        j=j,
    )
    affine_arr = np.asarray(affine, dtype=np.complex128).reshape(-1)
    d_affine_arr = np.asarray(d_affine, dtype=np.complex128).reshape(-1, 5)

    phase = np.exp(1j * x_flat[:, None] * u_flat[None, :])
    # REVIEW: formulas assume u != 0; fixed quadrature nodes should exclude zero or a separate limit should be implemented.
    denom = 1j * u_flat[None, :]

    values_2d = np.real(phase * affine_arr[None, :] / denom)

    jac_values_3d = np.real(
        phase[:, :, None] * d_affine_arr[None, :, :] / denom[:, :, None]
    )

    if x_scalar and u_scalar:
        return (
            float(values_2d[0, 0]),
            np.asarray(jac_values_3d[0, 0, :], dtype=np.float64),
        )
    if x_scalar:
        return (
            np.asarray(values_2d[0, :].reshape(u_shape), dtype=np.float64),
            np.asarray(
                jac_values_3d[0, :, :].reshape(u_shape + (5,)),
                dtype=np.float64,
            ),
        )
    if u_scalar:
        return (
            np.asarray(values_2d[:, 0].reshape(x_shape), dtype=np.float64),
            np.asarray(
                jac_values_3d[:, 0, :].reshape(x_shape + (5,)),
                dtype=np.float64,
            ),
        )

    return (
        np.asarray(values_2d.reshape(x_shape + u_shape), dtype=np.float64),
        np.asarray(
            jac_values_3d.reshape(x_shape + u_shape + (5,)),
            dtype=np.float64,
        ),
    )


def _integrand_scalar(
    u: float,
    x: float,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
) -> float:
    return float(_integrand(u=u, x=x, tau=tau, params=params, j=j))


def _probability_from_integral(integral: float | ArrayLike) -> float | RealArray:
    values = 0.5 + np.asarray(integral, dtype=np.float64) / np.pi
    if np.asarray(values).ndim == 0:
        return float(values)
    return np.asarray(values, dtype=np.float64)


def _default_heston_quadrature_config() -> QuadratureConfig:
    return QuadratureConfig(
        u_max=150.0,
        n_panels=24,
        nodes_per_panel=16,
    )


def _build_heston_gauss_rule(quad_cfg: QuadratureConfig) -> CompositeRule:
    """Build the cached fixed Gauss-Legendre rule used by the Heston backend."""
    return build_gauss_legendre_rule(quad_cfg)


def _resolve_gauss_rule_and_quad_cfg(
    *,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
) -> tuple[CompositeRule, QuadratureConfig | None]:
    if quad_cfg is not None and rule is not None:
        raise ValueError(
            "Pass either quad_cfg or rule, not both. "
            "If you already built a rule, it is the authoritative discretization."
        )

    if rule is not None:
        return rule, None

    active_quad_cfg = quad_cfg or _default_heston_quadrature_config()
    return _build_heston_gauss_rule(active_quad_cfg), active_quad_cfg


def _resolve_gauss_rule(
    *,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
) -> CompositeRule:
    active_rule, _ = _resolve_gauss_rule_and_quad_cfg(quad_cfg=quad_cfg, rule=rule)
    return active_rule


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple < 1:
        raise ValueError("multiple must be >= 1")
    return int(multiple * np.ceil(float(value) / float(multiple)))


def _bad_panel_count_threshold(n_panels: int) -> int:
    return max(
        int(HESTON_DIAG_BAD_PANEL_COUNT_WARN),
        int(np.ceil(HESTON_DIAG_BAD_PANEL_FRACTION_WARN * float(n_panels))),
    )


def _oscillation_mass_floor(abs_mass: ArrayLike) -> RealArray:
    abs_mass_arr = np.asarray(abs_mass, dtype=np.float64)
    return np.maximum(
        HESTON_DIAG_OSCILLATION_ABS_MASS_FLOOR,
        HESTON_DIAG_OSCILLATION_REL_MASS_FLOOR * abs_mass_arr,
    )


def recommend_heston_quadrature_config(
    *,
    x: float,
    tau: float,
    params: HestonParams,
    quality: QuadratureQuality = "balanced",
) -> QuadratureConfig:
    """Recommend a fixed Gauss-Legendre rule for Heston inversion."""
    if quality not in ("fast", "balanced", "robust", "diagnostics"):
        raise ValueError(
            "quality must be one of: 'fast', 'balanced', 'robust', 'diagnostics'"
        )

    x = float(x)
    tau = float(tau)

    if not np.isfinite(x):
        raise ValueError("x must be finite.")
    if not np.isfinite(tau):
        raise ValueError("tau must be finite.")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative.")

    # Branch baseline.
    base_u_max = 150.0
    base_n_panels = 24
    base_nodes = 16

    # At tau == 0 the pricer should usually short-circuit to intrinsic anyway.
    if tau == 0.0:
        return QuadratureConfig(
            u_max=base_u_max,
            n_panels=base_n_panels,
            nodes_per_panel=base_nodes,
        )

    abs_x = abs(x)
    eta = abs(float(params.eta))
    abs_rho = abs(float(params.rho))

    u_max = base_u_max
    n_panels = base_n_panels
    nodes_per_panel = base_nodes
    panel_spacing = PanelSpacing.UNIFORM
    cluster_strength = 2.0

    if tau < 0.05:
        u_max = max(u_max, 280.0)
        n_panels = max(n_panels, 44)
    elif tau < 0.15:
        u_max = max(u_max, 240.0)
        n_panels = max(n_panels, 36)
    elif tau < 0.50:
        u_max = max(u_max, 180.0)
        n_panels = max(n_panels, 28)

    if abs_x > 0.50:
        n_panels = max(n_panels, 32)
    if abs_x > 1.00:
        u_max = max(u_max, 220.0)
        n_panels = max(n_panels, 40)
    if abs_x > 1.50:
        u_max = max(u_max, 280.0)
        n_panels = max(n_panels, 48)

    if eta > 0.80:
        u_max = max(u_max, 220.0)
        n_panels = max(n_panels, 36)
    if eta > 1.00:
        u_max = max(u_max, 260.0)
        n_panels = max(n_panels, 44)
    if abs_rho > 0.70 and eta > 0.50:
        u_max = max(u_max, 240.0)
        n_panels = max(n_panels, 40)

    if eta < 1e-3:
        panel_spacing = PanelSpacing.CLUSTERED
        cluster_strength = 2.5
        n_panels = max(n_panels, 40)
        nodes_per_panel = max(nodes_per_panel, 24)
    elif eta < 0.10 and tau <= 1.0:
        panel_spacing = PanelSpacing.CLUSTERED
        cluster_strength = 2.0
        n_panels = max(n_panels, 28)

    if quality == "fast":
        u_max *= 0.85
        n_panels = max(16, int(np.ceil(0.75 * n_panels)))
        nodes_per_panel = 12 if nodes_per_panel <= 16 else nodes_per_panel
    elif quality == "robust":
        u_max *= 1.25
        n_panels = int(np.ceil(1.25 * n_panels))
        nodes_per_panel = max(nodes_per_panel, 24)
    elif quality == "diagnostics":
        u_max *= 1.50
        n_panels = int(np.ceil(1.50 * n_panels))
        nodes_per_panel = max(nodes_per_panel, 32)
        if panel_spacing == PanelSpacing.UNIFORM and (tau < 0.25 or eta < 0.10):
            panel_spacing = PanelSpacing.CLUSTERED
            cluster_strength = 2.0

    target_panel_width = 7.5 if quality == "fast" else 6.25
    n_panels = max(n_panels, int(np.ceil(u_max / target_panel_width)))

    n_panels = _round_up_to_multiple(n_panels, 4)
    nodes_per_panel = _round_up_to_multiple(nodes_per_panel, 4)

    cfg = QuadratureConfig(
        u_max=float(u_max),
        n_panels=int(n_panels),
        nodes_per_panel=int(nodes_per_panel),
        panel_spacing=panel_spacing,
        cluster_strength=float(cluster_strength),
    )
    cfg.validate()
    return cfg


def _integrate_pj_quad(
    x: float,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
) -> tuple[float, float]:
    """Integrate one Heston probability kernel with SciPy ``quad``."""
    integral, err_est = quad(
        _integrand_scalar,
        a=0.0,
        b=np.inf,
        args=(x, tau, params, j),
        complex_func=False,
    )
    return float(integral), float(err_est)


def _compute_fixed_rule_panel_reason(
    values_panel: ArrayLike,
    panel_contribs: ArrayLike,
) -> UInt32Array:
    """Return one bitmap per panel, flattened over any leading batch shape."""
    values_panel_arr = np.asarray(values_panel, dtype=np.float64)
    panel_contribs_arr = np.asarray(panel_contribs, dtype=np.float64)

    if values_panel_arr.ndim == 2:
        values_panel_flat = values_panel_arr.reshape(1, *values_panel_arr.shape)
    else:
        values_panel_flat = values_panel_arr.reshape(
            -1,
            values_panel_arr.shape[-2],
            values_panel_arr.shape[-1],
        )

    if panel_contribs_arr.ndim == 1:
        panel_contribs_flat = panel_contribs_arr.reshape(1, -1)
    else:
        panel_contribs_flat = panel_contribs_arr.reshape(
            -1, panel_contribs_arr.shape[-1]
        )

    if values_panel_flat.shape[:2] != panel_contribs_flat.shape:
        raise ValueError(
            "values_panel and panel_contribs shapes are inconsistent for diagnostics."
        )

    batch_size, n_panels, _ = values_panel_flat.shape
    panel_reason = np.zeros((batch_size, n_panels), dtype=np.uint32)

    panel_reason[np.any(~np.isfinite(values_panel_flat), axis=-1)] |= _flag_u32(
        HestonPanelReason.NONFINITE_INTEGRAND
    )
    panel_reason[~np.isfinite(panel_contribs_flat)] |= _flag_u32(
        HestonPanelReason.NONFINITE_PANEL_CONTRIB
    )

    abs_panel_contribs = np.abs(panel_contribs_flat)
    finite_abs_panel_contribs = np.where(
        np.isfinite(abs_panel_contribs), abs_panel_contribs, 0.0
    )
    abs_mass = np.sum(finite_abs_panel_contribs, axis=1)
    oscillation_mass_floor = _oscillation_mass_floor(abs_mass)

    near_n = min(HESTON_DIAG_NEAR_ORIGIN_PANEL_COUNT, n_panels)
    if near_n > 0:
        near_abs = np.sum(finite_abs_panel_contribs[:, :near_n], axis=1)
        near_fraction = np.zeros(batch_size, dtype=np.float64)
        nonzero_mass = abs_mass > 0.0
        near_fraction[nonzero_mass] = near_abs[nonzero_mass] / abs_mass[nonzero_mass]
        near_mask = near_fraction > HESTON_DIAG_NEAR_ORIGIN_ABS_FRACTION_WARN
        if np.any(near_mask):
            panel_reason[near_mask, :near_n] |= _flag_u32(
                HestonPanelReason.UNDERRESOLVED_NEAR_ORIGIN
            )

    tail_n = min(HESTON_DIAG_TAIL_PANEL_COUNT, n_panels)
    tail_fraction = np.zeros(batch_size, dtype=np.float64)
    if tail_n > 0:
        tail_abs = np.sum(finite_abs_panel_contribs[:, -tail_n:], axis=1)
        nonzero_mass = abs_mass > 0.0
        tail_fraction[nonzero_mass] = tail_abs[nonzero_mass] / abs_mass[nonzero_mass]
        tail_mask = tail_fraction > HESTON_DIAG_TAIL_ABS_FRACTION_WARN
        if np.any(tail_mask):
            panel_reason[tail_mask, -tail_n:] |= _flag_u32(
                HestonPanelReason.TAIL_TOO_LARGE
            )

    if n_panels >= 2 and tail_n > 0:
        last_abs = abs_panel_contribs[:, -1]
        prev_abs = abs_panel_contribs[:, -2]
        last_over_prev = np.full(batch_size, np.inf, dtype=np.float64)

        prev_nonzero = prev_abs > _DIAG_EPS
        last_over_prev[prev_nonzero] = last_abs[prev_nonzero] / prev_abs[prev_nonzero]
        last_over_prev[~prev_nonzero & (last_abs <= _DIAG_EPS)] = 0.0

        under_tail_mask = (tail_fraction > 0.5 * HESTON_DIAG_TAIL_ABS_FRACTION_WARN) & (
            last_over_prev > HESTON_DIAG_TAIL_LAST_OVER_PREV_WARN
        )
        if np.any(under_tail_mask):
            panel_reason[under_tail_mask, -tail_n:] |= _flag_u32(
                HestonPanelReason.UNDERRESOLVED_TAIL
            )

    if n_panels >= 3:
        for i in range(1, n_panels - 1):
            left = abs_panel_contribs[:, i - 1]
            right = abs_panel_contribs[:, i + 1]

            local_sum = np.zeros(batch_size, dtype=np.float64)
            local_count = np.zeros(batch_size, dtype=np.float64)

            left_finite = np.isfinite(left)
            right_finite = np.isfinite(right)

            local_sum[left_finite] += left[left_finite]
            local_sum[right_finite] += right[right_finite]
            local_count[left_finite] += 1.0
            local_count[right_finite] += 1.0

            local_ref = np.full(batch_size, _DIAG_EPS, dtype=np.float64)
            has_ref = local_count > 0.0
            local_ref[has_ref] = np.maximum(
                local_sum[has_ref] / local_count[has_ref],
                _DIAG_EPS,
            )

            spike_mask = (
                np.isfinite(abs_panel_contribs[:, i])
                & (
                    abs_panel_contribs[:, i]
                    > HESTON_DIAG_OSCILLATION_SPIKE_FACTOR * local_ref
                )
                & (abs_panel_contribs[:, i] >= oscillation_mass_floor)
            )
            if np.any(spike_mask):
                panel_reason[spike_mask, i] |= _flag_u32(
                    HestonPanelReason.OSCILLATION_SPIKE
                )

    return np.asarray(panel_reason.reshape(panel_contribs_arr.shape), dtype=np.uint32)


def _compute_global_warning_flags(
    total_integral: float | ArrayLike,
    probability: float | ArrayLike,
    *,
    panel_invalid: BoolArray | None = None,
    panel_contribs: ArrayLike | None = None,
    panel_reason: UInt32Array | None = None,
    quad_error_estimate: float | ArrayLike | None = None,
) -> tuple[UInt32Array, RealArray | None, RealArray | None]:
    """Return flattened warning flags and optional explanatory metrics."""
    total_flat = np.asarray(total_integral, dtype=np.float64).reshape(-1)
    probability_flat = np.asarray(probability, dtype=np.float64).reshape(-1)
    if total_flat.shape != probability_flat.shape:
        raise ValueError("total_integral and probability must have matching shapes.")

    flags = np.zeros(total_flat.shape, dtype=np.uint32)

    flags[~np.isfinite(total_flat)] |= _flag_u32(HestonIntegralWarning.NONFINITE_TOTAL)
    flags[~np.isfinite(probability_flat)] |= _flag_u32(
        HestonIntegralWarning.NONFINITE_PROBABILITY
    )

    prob_out_of_range = np.isfinite(probability_flat) & (
        (probability_flat < -HESTON_DIAG_PROBABILITY_TOL)
        | (probability_flat > 1.0 + HESTON_DIAG_PROBABILITY_TOL)
    )
    flags[prob_out_of_range] |= _flag_u32(
        HestonIntegralWarning.PROBABILITY_OUT_OF_RANGE
    )

    tail_abs_fraction: RealArray | None = None
    cancellation_ratio: RealArray | None = None

    if panel_contribs is not None:
        panel_contribs_arr = np.asarray(panel_contribs, dtype=np.float64)
        if panel_contribs_arr.ndim == 1:
            panel_contribs_flat = panel_contribs_arr.reshape(1, -1)
        else:
            panel_contribs_flat = panel_contribs_arr.reshape(
                -1, panel_contribs_arr.shape[-1]
            )

        if panel_contribs_flat.shape[0] != total_flat.shape[0]:
            raise ValueError(
                "panel_contribs batch shape must match total_integral/probability."
            )

        abs_panel_contribs = np.abs(panel_contribs_flat)
        finite_abs_panel_contribs = np.where(
            np.isfinite(abs_panel_contribs),
            abs_panel_contribs,
            0.0,
        )
        abs_mass = np.sum(finite_abs_panel_contribs, axis=1)

        tail_n = min(HESTON_DIAG_TAIL_PANEL_COUNT, panel_contribs_flat.shape[1])
        tail_abs_fraction = np.zeros(total_flat.shape, dtype=np.float64)
        if tail_n > 0:
            tail_abs = np.sum(finite_abs_panel_contribs[:, -tail_n:], axis=1)
            nonzero_mass = abs_mass > 0.0
            tail_abs_fraction[nonzero_mass] = (
                tail_abs[nonzero_mass] / abs_mass[nonzero_mass]
            )
            flags[tail_abs_fraction > HESTON_DIAG_TAIL_ABS_FRACTION_WARN] |= _flag_u32(
                HestonIntegralWarning.LARGE_TAIL_FRACTION
            )

        cancellation_ratio = np.zeros(total_flat.shape, dtype=np.float64)
        nonzero_mass = abs_mass > 0.0
        cancellation_ratio[nonzero_mass] = abs_mass[nonzero_mass] / np.maximum(
            np.abs(total_flat[nonzero_mass]), _DIAG_EPS
        )
        flags[cancellation_ratio > HESTON_DIAG_CANCELLATION_RATIO_WARN] |= _flag_u32(
            HestonIntegralWarning.EXCESSIVE_CANCELLATION
        )

        if panel_invalid is not None:
            panel_invalid_arr = np.asarray(panel_invalid, dtype=np.bool_)
            if panel_invalid_arr.ndim == 1:
                panel_invalid_flat = panel_invalid_arr.reshape(1, -1)
            else:
                panel_invalid_flat = panel_invalid_arr.reshape(
                    -1, panel_invalid_arr.shape[-1]
                )

            if panel_invalid_flat.shape != panel_contribs_flat.shape:
                raise ValueError(
                    "panel_invalid and panel_contribs must have matching panel shapes."
                )

            panel_reason_flat: UInt32Array | None = None
            if panel_reason is not None:
                panel_reason_arr = np.asarray(panel_reason, dtype=np.uint32)
                if panel_reason_arr.ndim == 1:
                    panel_reason_flat = panel_reason_arr.reshape(1, -1)
                else:
                    panel_reason_flat = panel_reason_arr.reshape(
                        -1, panel_reason_arr.shape[-1]
                    )

                if panel_reason_flat.shape != panel_contribs_flat.shape:
                    raise ValueError(
                        "panel_reason and panel_contribs must have matching panel "
                        "shapes."
                    )

            effective_bad_panels = np.asarray(panel_invalid_flat, dtype=np.bool_).copy()
            if panel_reason_flat is not None:
                near_origin_only = panel_reason_flat == _flag_u32(
                    HestonPanelReason.UNDERRESOLVED_NEAR_ORIGIN
                )
                effective_bad_panels &= ~near_origin_only

            bad_panel_counts = np.count_nonzero(effective_bad_panels, axis=1)
            threshold = _bad_panel_count_threshold(panel_contribs_flat.shape[1])
            flags[bad_panel_counts >= threshold] |= _flag_u32(
                HestonIntegralWarning.TOO_MANY_BAD_PANELS
            )

    if quad_error_estimate is not None:
        quad_error_flat = np.asarray(quad_error_estimate, dtype=np.float64).reshape(-1)
        if quad_error_flat.shape != total_flat.shape:
            raise ValueError(
                "quad_error_estimate must match total_integral/probability shape."
            )

        rel_error = quad_error_flat / np.maximum(1.0, np.abs(total_flat))
        flags[rel_error > HESTON_DIAG_QUAD_REL_ERROR_WARN] |= _flag_u32(
            HestonIntegralWarning.QUAD_ERROR_LARGE
        )

    return (
        np.asarray(flags, dtype=np.uint32),
        (
            None
            if tail_abs_fraction is None
            else np.asarray(tail_abs_fraction, dtype=np.float64)
        ),
        (
            None
            if cancellation_ratio is None
            else np.asarray(cancellation_ratio, dtype=np.float64)
        ),
    )


def _integrate_pj_fixed_rule(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    rule: CompositeRule,
) -> float | RealArray:
    """Integrate one Heston probability kernel on a reusable fixed rule."""
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)

    values_panel = np.asarray(
        _integrand(u=rule.u_panel, x=x_arr, tau=tau, params=params, j=j),
        dtype=np.float64,
    )

    omega = rule.omega_panel.reshape((1,) + rule.omega_panel.shape)
    weighted = values_panel * omega
    panel_contribs = np.sum(weighted, axis=-1)
    total = np.sum(panel_contribs, axis=-1)

    return _restore_x_shape(
        total,
        scalar_input=scalar_input,
        original_shape=original_shape,
    )


def _integrate_pj_fixed_rule_and_param_jac(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    rule: CompositeRule,
) -> tuple[float | RealArray, RealArray]:
    """Integrate one Heston probability kernel and parameter Jacobian."""
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)

    values_panel, jac_values_panel = _integrand_and_param_jac(
        u=rule.u_panel,
        x=x_arr,
        tau=tau,
        params=params,
        j=j,
    )
    values_panel_arr = np.asarray(values_panel, dtype=np.float64)
    jac_values_panel_arr = np.asarray(jac_values_panel, dtype=np.float64)

    omega = rule.omega_panel.reshape((1,) + rule.omega_panel.shape)
    weighted_values = values_panel_arr * omega
    weighted_jac = jac_values_panel_arr * omega[..., None]

    panel_contribs = np.sum(weighted_values, axis=-1)
    panel_jac_contribs = np.sum(weighted_jac, axis=-2)

    total = np.sum(panel_contribs, axis=-1)
    total_jac = np.sum(panel_jac_contribs, axis=-2)

    integral = _restore_x_shape(
        total,
        scalar_input=scalar_input,
        original_shape=original_shape,
    )
    total_jac_arr = np.asarray(total_jac, dtype=np.float64)
    if scalar_input:
        d_integral = np.asarray(total_jac_arr.reshape(-1, 5)[0], dtype=np.float64)
    else:
        d_integral = np.asarray(
            total_jac_arr.reshape(original_shape + (5,)),
            dtype=np.float64,
        )

    # REVIEW: diagnostics for panel-level Jacobian contributions are intentionally not added in this phase.
    return integral, d_integral


def _integrate_pj_fixed_rule_with_diagnostics(
    x: float,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    rule: CompositeRule,
    *,
    quad_cfg: QuadratureConfig | None,
) -> HestonIntegralDiagnostics:
    """Integrate one fixed-rule probability kernel and keep panel diagnostics."""
    values_panel = np.asarray(
        _integrand(u=rule.u_panel, x=x, tau=tau, params=params, j=j),
        dtype=np.float64,
    )
    panel_contribs = np.asarray(
        np.sum(rule.omega_panel * values_panel, axis=-1),
        dtype=np.float64,
    )
    total_integral = float(np.sum(panel_contribs))
    probability = float(_probability_from_integral(total_integral))

    panel_reason = _compute_fixed_rule_panel_reason(values_panel, panel_contribs)
    panel_invalid = np.asarray(panel_reason != 0, dtype=np.bool_)

    warning_flags_arr, tail_abs_fraction_arr, cancellation_ratio_arr = (
        _compute_global_warning_flags(
            total_integral,
            probability,
            panel_invalid=panel_invalid,
            panel_contribs=panel_contribs,
            panel_reason=panel_reason,
        )
    )

    tail_abs_fraction = (
        None if tail_abs_fraction_arr is None else float(tail_abs_fraction_arr[0])
    )
    cancellation_ratio = (
        None if cancellation_ratio_arr is None else float(cancellation_ratio_arr[0])
    )

    return HestonIntegralDiagnostics(
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
        j=j,
        x=x,
        tau=tau,
        total_integral=total_integral,
        probability=probability,
        panel_contribs=np.asarray(panel_contribs, dtype=np.float64),
        panel_edges=np.asarray(rule.panel_edges, dtype=np.float64),
        quad_error_estimate=None,
        panel_invalid=panel_invalid,
        panel_reason=np.asarray(panel_reason, dtype=np.uint32),
        warning_flags=HestonIntegralWarning(int(warning_flags_arr[0])),
        tail_abs_fraction=tail_abs_fraction,
        cancellation_ratio=cancellation_ratio,
    )


def _integrate_pj_fixed_rule_batch_with_diagnostics(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    rule: CompositeRule,
    *,
    quad_cfg: QuadratureConfig | None,
) -> HestonIntegralBatchDiagnostics:
    """Integrate a batch of fixed-rule probability kernels with diagnostics."""
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)

    if scalar_input:
        raise ValueError(
            "_integrate_pj_fixed_rule_batch_with_diagnostics requires array-like x, "
            "not a scalar."
        )

    values_panel = np.asarray(
        _integrand(u=rule.u_panel, x=x_arr, tau=tau, params=params, j=j),
        dtype=np.float64,
    )
    omega = rule.omega_panel.reshape((1,) + rule.omega_panel.shape)
    weighted = values_panel * omega

    panel_contribs_flat = np.asarray(np.sum(weighted, axis=-1), dtype=np.float64)
    total_integral = np.asarray(
        np.sum(panel_contribs_flat, axis=-1),
        dtype=np.float64,
    ).reshape(original_shape)
    probability = np.asarray(
        _probability_from_integral(total_integral),
        dtype=np.float64,
    ).reshape(original_shape)

    panel_reason_flat = _compute_fixed_rule_panel_reason(
        values_panel, panel_contribs_flat
    )
    panel_invalid_flat = np.asarray(panel_reason_flat != 0, dtype=np.bool_)

    warning_flags_flat, tail_abs_fraction_flat, cancellation_ratio_flat = (
        _compute_global_warning_flags(
            total_integral,
            probability,
            panel_invalid=panel_invalid_flat,
            panel_contribs=panel_contribs_flat,
            panel_reason=panel_reason_flat,
        )
    )

    n_panels = rule.u_panel.shape[0]

    return HestonIntegralBatchDiagnostics(
        backend="gauss_legendre",
        quad_cfg=quad_cfg,
        j=j,
        x=np.asarray(x_arr.reshape(original_shape), dtype=np.float64),
        tau=tau,
        total_integral=np.asarray(total_integral, dtype=np.float64),
        probability=probability,
        panel_contribs=np.asarray(
            panel_contribs_flat.reshape(original_shape + (n_panels,)),
            dtype=np.float64,
        ),
        panel_edges=np.asarray(rule.panel_edges, dtype=np.float64),
        quad_error_estimate=None,
        panel_invalid=np.asarray(
            panel_invalid_flat.reshape(original_shape + (n_panels,)),
            dtype=np.bool_,
        ),
        panel_reason=np.asarray(
            panel_reason_flat.reshape(original_shape + (n_panels,)),
            dtype=np.uint32,
        ),
        warning_flags=np.asarray(
            warning_flags_flat.reshape(original_shape),
            dtype=np.uint32,
        ),
        tail_abs_fraction=(
            None
            if tail_abs_fraction_flat is None
            else np.asarray(
                tail_abs_fraction_flat.reshape(original_shape), dtype=np.float64
            )
        ),
        cancellation_ratio=(
            None
            if cancellation_ratio_flat is None
            else np.asarray(
                cancellation_ratio_flat.reshape(original_shape), dtype=np.float64
            )
        ),
    )


@overload
def heston_probability(
    x: float | np.floating,
    tau: float,
    params: HestonParams,
    probability_index: HestonProbabilityIndex,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float: ...


@overload
def heston_probability(
    x: np.ndarray,
    tau: float,
    params: HestonParams,
    probability_index: HestonProbabilityIndex,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> RealArray: ...


def heston_probability(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    probability_index: HestonProbabilityIndex,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float | RealArray:
    """Evaluate a Heston probability integral."""
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)
    j = probability_index

    if backend == "quad":
        if quad_cfg is not None or rule is not None:
            raise ValueError("quad backend does not accept quad_cfg or rule.")
        values = np.asarray(
            [_integrate_pj_quad(float(xi), tau, params, j)[0] for xi in x_arr],
            dtype=np.float64,
        )
        probabilities = _probability_from_integral(values)
        return _restore_x_shape(
            probabilities,
            scalar_input=scalar_input,
            original_shape=original_shape,
        )

    if backend == "gauss_legendre":
        active_rule = _resolve_gauss_rule(quad_cfg=quad_cfg, rule=rule)
        integral = _integrate_pj_fixed_rule(x, tau, params, j, active_rule)
        return _restore_x_shape(
            _probability_from_integral(integral),
            scalar_input=scalar_input,
            original_shape=original_shape,
        )

    raise ValueError(f"Unknown backend: {backend}")


def heston_probability_and_param_jac(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    probability_index: HestonProbabilityIndex,
    *,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> tuple[float | RealArray, RealArray]:
    """Evaluate a Heston probability integral and constrained parameter Jacobian."""
    _, scalar_input, original_shape = _normalize_x_grid(x)
    j = probability_index

    if backend == "quad":
        raise NotImplementedError(
            "Analytic Heston probability Jacobian is currently implemented only "
            "for backend='gauss_legendre'."
        )

    if backend == "gauss_legendre":
        active_rule = _resolve_gauss_rule(quad_cfg=quad_cfg, rule=rule)
        integral, d_integral_dtheta = _integrate_pj_fixed_rule_and_param_jac(
            x=x,
            tau=tau,
            params=params,
            j=j,
            rule=active_rule,
        )

        probability = _restore_x_shape(
            _probability_from_integral(integral),
            scalar_input=scalar_input,
            original_shape=original_shape,
        )
        d_probability_arr = np.asarray(d_integral_dtheta, dtype=np.float64) / np.pi
        if scalar_input:
            d_probability_dtheta = np.asarray(
                d_probability_arr.reshape(-1, 5)[0],
                dtype=np.float64,
            )
        else:
            d_probability_dtheta = np.asarray(
                d_probability_arr.reshape(original_shape + (5,)),
                dtype=np.float64,
            )

        # REVIEW: this function exposes constrained-parameter derivatives only; unconstrained optimizer chain-rule belongs in the calibration objective phase.
        return probability, d_probability_dtheta

    raise ValueError(f"Unknown backend: {backend}")


def P_j(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float | RealArray:
    """Backward-compatible alias for :func:`heston_probability`."""
    return heston_probability(
        x=x,
        tau=tau,
        params=params,
        probability_index=j,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )


def heston_probability_with_diagnostics(
    x: float,
    tau: float,
    params: HestonParams,
    probability_index: HestonProbabilityIndex,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> HestonIntegralDiagnostics:
    """Evaluate one Heston probability and return scalar integration diagnostics."""
    j = probability_index
    if backend == "quad":
        if quad_cfg is not None or rule is not None:
            raise ValueError("quad backend does not accept quad_cfg or rule.")
        integral, err_est = _integrate_pj_quad(x, tau, params, j)
        probability = float(_probability_from_integral(integral))
        warning_flags_arr, _, _ = _compute_global_warning_flags(
            integral,
            probability,
            quad_error_estimate=err_est,
        )
        return HestonIntegralDiagnostics(
            backend="quad",
            quad_cfg=None,
            j=j,
            x=x,
            tau=tau,
            total_integral=float(integral),
            probability=probability,
            panel_contribs=None,
            panel_edges=None,
            quad_error_estimate=float(err_est),
            panel_invalid=None,
            panel_reason=None,
            warning_flags=HestonIntegralWarning(int(warning_flags_arr[0])),
            tail_abs_fraction=None,
            cancellation_ratio=None,
        )

    if backend == "gauss_legendre":
        active_rule, active_quad_cfg = _resolve_gauss_rule_and_quad_cfg(
            quad_cfg=quad_cfg,
            rule=rule,
        )
        return _integrate_pj_fixed_rule_with_diagnostics(
            x=x,
            tau=tau,
            params=params,
            j=j,
            rule=active_rule,
            quad_cfg=active_quad_cfg,
        )

    raise ValueError(f"Unknown backend: {backend}")


def P_j_with_diagnostics(
    x: float,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> HestonIntegralDiagnostics:
    """Backward-compatible alias for :func:`heston_probability_with_diagnostics`."""
    return heston_probability_with_diagnostics(
        x=x,
        tau=tau,
        params=params,
        probability_index=j,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )


def heston_probability_batch_with_diagnostics(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    probability_index: HestonProbabilityIndex,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> HestonIntegralBatchDiagnostics:
    """Evaluate a batch of Heston probabilities and return diagnostics."""
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)
    j = probability_index

    if scalar_input:
        raise ValueError(
            "heston_probability_batch_with_diagnostics requires array-like x, "
            "not a scalar. Use heston_probability_with_diagnostics for the "
            "scalar case."
        )

    if backend == "quad":
        if quad_cfg is not None or rule is not None:
            raise ValueError("quad backend does not accept quad_cfg or rule.")

        integrals = np.empty_like(x_arr, dtype=np.float64)
        errors = np.empty_like(x_arr, dtype=np.float64)

        for idx, xi in enumerate(x_arr):
            integral_i, err_i = _integrate_pj_quad(float(xi), tau, params, j)
            integrals[idx] = integral_i
            errors[idx] = err_i

        total_integral = np.asarray(integrals.reshape(original_shape), dtype=np.float64)
        probability = np.asarray(
            _probability_from_integral(total_integral),
            dtype=np.float64,
        ).reshape(original_shape)

        warning_flags_flat, _, _ = _compute_global_warning_flags(
            total_integral,
            probability,
            quad_error_estimate=np.asarray(
                errors.reshape(original_shape), dtype=np.float64
            ),
        )

        return HestonIntegralBatchDiagnostics(
            backend="quad",
            quad_cfg=None,
            j=j,
            x=np.asarray(x_arr.reshape(original_shape), dtype=np.float64),
            tau=tau,
            total_integral=total_integral,
            probability=probability,
            panel_contribs=None,
            panel_edges=None,
            quad_error_estimate=np.asarray(
                errors.reshape(original_shape), dtype=np.float64
            ),
            panel_invalid=None,
            panel_reason=None,
            warning_flags=np.asarray(
                warning_flags_flat.reshape(original_shape),
                dtype=np.uint32,
            ),
            tail_abs_fraction=None,
            cancellation_ratio=None,
        )

    if backend == "gauss_legendre":
        active_rule, active_quad_cfg = _resolve_gauss_rule_and_quad_cfg(
            quad_cfg=quad_cfg,
            rule=rule,
        )
        return _integrate_pj_fixed_rule_batch_with_diagnostics(
            x=x_arr.reshape(original_shape),
            tau=tau,
            params=params,
            j=j,
            rule=active_rule,
            quad_cfg=active_quad_cfg,
        )

    raise ValueError(f"Unknown backend: {backend}")


def P_j_batch_with_diagnostics(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: HestonProbabilityIndex,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> HestonIntegralBatchDiagnostics:
    """Backward-compatible alias for :func:`heston_probability_batch_with_diagnostics`."""
    return heston_probability_batch_with_diagnostics(
        x=x,
        tau=tau,
        params=params,
        probability_index=j,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )
