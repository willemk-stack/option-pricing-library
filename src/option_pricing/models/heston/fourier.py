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
    PanelSpacing,
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

type QuadratureQuality = Literal["fast", "balanced", "robust", "diagnostics"]
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
    )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple < 1:
        raise ValueError("multiple must be >= 1")
    return int(multiple * np.ceil(float(value) / float(multiple)))


def recommend_heston_quadrature_config(
    *,
    x: float,
    tau: float,
    params: HestonParams,
    quality: QuadratureQuality = "balanced",
) -> QuadratureConfig:
    """
    Heuristic config recommender for fixed-rule Heston Fourier inversion.

    Rationale
    ---------
    Start from the branch's current fixed baseline:
        u_max = 150, n_panels = 24, nodes_per_panel = 16

    Then tilt the rule in three ways:

    1. Increase u_max when truncation is more likely to bind.
       This is most common for short maturities, large |x|, and high vol-of-vol.

    2. Increase n_panels when oscillation or localization suggests smaller panel
       widths are needed. When u_max is increased, keep panel width from growing
       too much by also increasing n_panels.

    3. Increase nodes_per_panel and prefer clustered spacing when early panels are
       expected to dominate, especially in near-deterministic / near-Black regimes.

    Notes for reviewers
    -------------------
    These constants are deliberately simple and easy to tweak. They are not a
    theorem, and they are not copied from one paper. They are chosen to reflect:

    - fixed-rule reproducibility,
    - regime-aware truncation in the spirit of transform methods,
    - and the panel-diagnostics workflow already present in this branch.

    Review points:
    - tau thresholds control tail/truncation conservatism
    - |x| thresholds control oscillation-driven panel refinement
    - eta thresholds control near-origin clustering and tail extension
    - the target panel width preserves roughly the same local resolution as the
      current baseline when u_max is enlarged
    """
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

    # ------------------------------------------------------------------
    # 1) Truncation heuristic: short maturities and hard regimes often
    #    need a longer interval.
    # ------------------------------------------------------------------
    if tau < 0.05:
        u_max = max(u_max, 280.0)
        n_panels = max(n_panels, 44)
    elif tau < 0.15:
        u_max = max(u_max, 240.0)
        n_panels = max(n_panels, 36)
    elif tau < 0.50:
        u_max = max(u_max, 180.0)
        n_panels = max(n_panels, 28)

    # ------------------------------------------------------------------
    # 2) Oscillation heuristic: larger |x| means faster e^{iux} oscillation.
    #    That argues for more panels, and sometimes a larger interval too.
    # ------------------------------------------------------------------
    if abs_x > 0.50:
        n_panels = max(n_panels, 32)
    if abs_x > 1.00:
        u_max = max(u_max, 220.0)
        n_panels = max(n_panels, 40)
    if abs_x > 1.50:
        u_max = max(u_max, 280.0)
        n_panels = max(n_panels, 48)

    # ------------------------------------------------------------------
    # 3) Vol-of-vol / correlation heuristic:
    #    high eta broadens difficult regimes; strong rho with high eta tends
    #    to make the inversion less forgiving.
    # ------------------------------------------------------------------
    if eta > 0.80:
        u_max = max(u_max, 220.0)
        n_panels = max(n_panels, 36)
    if eta > 1.00:
        u_max = max(u_max, 260.0)
        n_panels = max(n_panels, 44)
    if abs_rho > 0.70 and eta > 0.50:
        u_max = max(u_max, 240.0)
        n_panels = max(n_panels, 40)

    # ------------------------------------------------------------------
    # 4) Near-deterministic / near-Black heuristic:
    #    these regimes often do not need more tail, but do need more local
    #    resolution near the origin. Prefer clustering and more local order.
    # ------------------------------------------------------------------
    if eta < 1e-3:
        panel_spacing = PanelSpacing.CLUSTERED
        cluster_strength = 2.5
        n_panels = max(n_panels, 40)
        nodes_per_panel = max(nodes_per_panel, 24)
    elif eta < 0.10 and tau <= 1.0:
        panel_spacing = PanelSpacing.CLUSTERED
        cluster_strength = 2.0
        n_panels = max(n_panels, 28)

    # ------------------------------------------------------------------
    # 5) Quality presets.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6) Preserve local panel width when u_max grows.
    #
    # The current baseline width is 150 / 24 = 6.25. Keep roughly that scale
    # for balanced/robust/diagnostics so that increasing u_max does not
    # accidentally *reduce* local resolution.
    # ------------------------------------------------------------------
    target_panel_width = 7.5 if quality == "fast" else 6.25
    n_panels = max(n_panels, int(np.ceil(u_max / target_panel_width)))

    # Round to tidy values for repeatability / caching / readability.
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
