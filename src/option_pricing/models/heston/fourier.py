"""
Heston Fourier inversion and probability integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

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
    """Scalar diagnostics for one Heston probability integral.

    Attributes
    ----------
    backend : {"gauss_legendre", "quad"}
        Integration backend used.
    j : {0, 1}
        Probability index.
    x : float
        Log-forward moneyness supplied to the integral.
    tau : float
        Time to expiry in years.
    total_integral : float
        Raw integral value before conversion to a probability.
    probability : float
        Probability obtained from ``0.5 + integral / pi``.
    panel_contribs : ndarray, optional
        Per-panel contributions for the fixed-rule backend.
    panel_edges : ndarray, optional
        Panel edges corresponding to ``panel_contribs``.
    quad_error_estimate : float, optional
        Error estimate returned by :func:`scipy.integrate.quad`.
    """

    backend: Backend
    j: J
    x: float
    tau: float
    total_integral: float
    probability: float
    panel_contribs: FloatArray | None = None  # shape (n_panels,)
    panel_edges: FloatArray | None = None  # shape (n_panels + 1,)
    quad_error_estimate: float | None = None


@dataclass(frozen=True, slots=True)
class HestonIntegralBatchDiagnostics:
    """Batch diagnostics container for fixed-rule probability integrals.

    Notes
    -----
    This shape is suitable for vectorized fixed-rule evaluation over a batch of
    ``x`` values. The public diagnostics entry point exposed today,
    :func:`P_j_with_diagnostics`, is still scalar-only.
    """

    backend: Backend
    j: J
    x: FloatArray  # shape batch_shape
    tau: float
    total_integral: FloatArray  # shape batch_shape
    probability: FloatArray  # shape batch_shape
    panel_contribs: FloatArray | None = None  # shape batch_shape + (n_panels,)
    panel_edges: FloatArray | None = None  # shape (n_panels + 1,)
    quad_error_estimate: FloatArray | None = None  # shape batch_shape, or None


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
    """Evaluate the real-valued Heston inversion integrand.

    Parameters
    ----------
    u : array-like
        Frequency point or grid.
    x : array-like
        Log-forward moneyness value(s), usually ``log(F / K)``.
    tau : float
        Time to expiry in years.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index.

    Returns
    -------
    float or ndarray
        Integrand values with output shape ``x.shape + u.shape`` after scalar
        dimensions are removed in the usual way.

    Notes
    -----
    This helper is vectorized jointly over ``x`` and ``u`` and provides the
    batched shape contract used by the fixed-rule Gauss-Legendre backend.
    """
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


def _integrand_scalar(
    u: float,
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
) -> float:
    return float(_integrand(u=u, x=x, tau=tau, params=params, j=j))


def _probability_from_integral(integral: float | ArrayLike) -> float | RealArray:
    values = 0.5 + np.asarray(integral, dtype=np.float64) / np.pi
    if np.asarray(values).ndim == 0:
        return float(values)
    return np.asarray(values, dtype=np.float64)


def _integrate_pj_quad(
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
) -> tuple[float, float]:
    """Integrate one Heston probability kernel with SciPy ``quad``.

    Parameters
    ----------
    x : float
        Scalar log-forward moneyness.
    tau : float
        Time to expiry in years.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index.

    Returns
    -------
    tuple of float
        Integral value and ``quad`` error estimate.

    Notes
    -----
    This path is intentionally scalar. When callers pass an array of ``x``
    values through :func:`P_j`, the ``quad`` backend evaluates this function in
    a Python loop and is therefore not vectorized across the batch.
    """
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
    """Recommend a fixed Gauss-Legendre rule for Heston inversion.

    Parameters
    ----------
    x : float
        Representative absolute log-forward moneyness. For slice pricing, a good
        choice is ``max(abs(log(F / K_i)))`` over the slice.
    tau : float
        Time to expiry in years. Must be finite and nonnegative.
    params : HestonParams
        Heston parameter set.
    quality : {"fast", "balanced", "robust", "diagnostics"}, default "balanced"
        Heuristic quality preset controlling interval length, panel count, and
        nodes per panel.

    Returns
    -------
    QuadratureConfig
        Recommended composite Gauss-Legendre configuration.

    Notes
    -----
    The recommender starts from the baseline rule

    ``u_max = 150, n_panels = 24, nodes_per_panel = 16``

    and then adjusts it using simple regime heuristics:

    - short maturities, large ``|x|``, and large ``eta`` extend the truncation
      interval and usually increase the panel count,
    - near-deterministic and near-Black regimes increase local resolution and
      may switch to clustered panel spacing,
    - robust and diagnostics modes preserve roughly the baseline local panel
      width when ``u_max`` grows.

    These constants are heuristic rather than theorem-driven. They are chosen to
    keep the fixed-rule path reproducible while remaining conservative in harder
    parameter regimes.
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
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: J,
    rule: CompositeRule,
) -> float | RealArray:
    """Integrate one Heston probability kernel on a reusable fixed rule.

    Parameters
    ----------
    x : array-like
        Scalar or batch of log-forward moneyness values.
    tau : float
        Time to expiry in years.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index.
    rule : CompositeRule
        Prebuilt composite Gauss-Legendre rule.

    Returns
    -------
    float or ndarray
        Integral value(s) with the same scalar/array shape as ``x``.

    Notes
    -----
    This helper is vectorized over ``x`` because
    :func:`integrate_composite_rule` accepts a leading batch shape.
    """
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)
    result = integrate_composite_rule(
        lambda u: _integrand(u=u, x=x_arr, tau=tau, params=params, j=j),
        rule,
    )
    return _restore_x_shape(
        result.total,
        scalar_input=scalar_input,
        original_shape=original_shape,
    )


def _integrate_pj_fixed_rule_with_diagnostics(
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
    rule: CompositeRule,
) -> HestonIntegralDiagnostics:
    """Integrate one fixed-rule probability kernel and keep panel diagnostics.

    Parameters
    ----------
    x : float
        Scalar log-forward moneyness.
    tau : float
        Time to expiry in years.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index.
    rule : CompositeRule
        Prebuilt composite Gauss-Legendre rule.

    Returns
    -------
    HestonIntegralDiagnostics
        Scalar diagnostics object for this integral.

    Notes
    -----
    This wrapper is intentionally scalar even though the underlying fixed-rule
    integration can batch over ``x``. A public batch diagnostics constructor has
    not been wired yet.
    """
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
        probability=float(probability),
        panel_contribs=np.asarray(result.panel_contribs, dtype=np.float64),
        panel_edges=rule.panel_edges,
        quad_error_estimate=None,
    )


def _integrate_pj_fixed_rule_batch_with_diagnostics(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: J,
    rule: CompositeRule,
) -> HestonIntegralBatchDiagnostics:
    """Integrate a batch of fixed-rule probability kernels with diagnostics.

    Parameters
    ----------
    x : array-like
        Non-scalar batch of log-forward moneyness values.
    tau : float
        Time to expiry in years.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index.
    rule : CompositeRule
        Prebuilt composite Gauss-Legendre rule.

    Returns
    -------
    HestonIntegralBatchDiagnostics
        Batch diagnostics whose leading shape matches ``x.shape``.

    Raises
    ------
    ValueError
        If ``x`` is scalar.

    Notes
    -----
    This helper is intended for the fixed-rule Gauss-Legendre path only. There
    is no analogous batched diagnostics implementation for the scalar SciPy
    ``quad`` backend.
    """
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)

    if scalar_input:
        raise ValueError(
            "_integrate_pj_fixed_rule_batch_with_diagnostics requires array-like x, "
            "not a scalar."
        )

    result = integrate_composite_rule(
        lambda u: _integrand(u=u, x=x_arr, tau=tau, params=params, j=j),
        rule,
    )

    total_integral = np.asarray(result.total, dtype=np.float64).reshape(original_shape)
    probability = np.asarray(
        _probability_from_integral(total_integral),
        dtype=np.float64,
    ).reshape(original_shape)

    panel_contribs = np.asarray(result.panel_contribs, dtype=np.float64)

    return HestonIntegralBatchDiagnostics(
        backend="gauss_legendre",
        j=j,
        x=np.asarray(x_arr.reshape(original_shape), dtype=np.float64),
        tau=tau,
        total_integral=total_integral,
        probability=probability,
        panel_contribs=panel_contribs.reshape(
            original_shape + (rule.u_panel.shape[0],)
        ),
        panel_edges=np.asarray(rule.panel_edges, dtype=np.float64),
        quad_error_estimate=None,
    )


@overload
def P_j(
    x: float | np.floating,
    tau: float,
    params: HestonParams,
    j: J,
    backend: Backend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float: ...


@overload
def P_j(
    x: np.ndarray,
    tau: float,
    params: HestonParams,
    j: J,
    backend: Backend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> RealArray: ...


def P_j(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: J,
    backend: Backend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> float | RealArray:
    """Evaluate a Heston probability integral ``P_j``.

    Parameters
    ----------
    x : float or array-like
        Log-forward moneyness value(s), usually ``log(F / K)``.
    tau : float
        Time to expiry in years. Must be finite and nonnegative.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index in the Heston inversion formula.
    backend : {"gauss_legendre", "quad"}, default "gauss_legendre"
        Integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for the Gauss-Legendre backend.
    rule : CompositeRule, optional
        Prebuilt fixed rule for the Gauss-Legendre backend.

    Returns
    -------
    float or ndarray
        Probability value(s) with the same scalar/array shape as ``x``.

    Raises
    ------
    ValueError
        If an unknown backend is requested, or if ``quad_cfg`` / ``rule`` are
        passed to the scalar ``quad`` backend.

    Notes
    -----
    The Gauss-Legendre backend is vectorized across array-valued ``x`` inputs.
    The ``quad`` backend accepts array input for API consistency, but it
    internally evaluates one scalar :func:`scipy.integrate.quad` call per
    element of ``x``. In other words, batch ``quad`` evaluation is convenient
    but not vectorized.
    """
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)

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


def P_j_with_diagnostics(
    x: float,
    tau: float,
    params: HestonParams,
    j: J,
    backend: Backend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> HestonIntegralDiagnostics:
    """Evaluate ``P_j`` and return scalar integration diagnostics.

    Parameters
    ----------
    x : float
        Scalar log-forward moneyness.
    tau : float
        Time to expiry in years.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index in the Heston inversion formula.
    backend : {"gauss_legendre", "quad"}, default "gauss_legendre"
        Integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for the Gauss-Legendre backend.
    rule : CompositeRule, optional
        Prebuilt fixed rule for the Gauss-Legendre backend.

    Returns
    -------
    HestonIntegralDiagnostics
        Scalar diagnostics object for the requested integral.

    Notes
    -----
    Diagnostics are currently exposed only for scalar ``x`` inputs. The fixed
    Gauss-Legendre machinery can batch over ``x`` internally, but a public batch
    diagnostics routine has not been wired through this API yet.
    """
    if backend == "quad":
        if quad_cfg is not None or rule is not None:
            raise ValueError("quad backend does not accept quad_cfg or rule.")
        integral, err_est = _integrate_pj_quad(x, tau, params, j)
        probability = _probability_from_integral(integral)
        return HestonIntegralDiagnostics(
            backend="quad",
            j=j,
            x=x,
            tau=tau,
            total_integral=float(integral),
            probability=float(probability),
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


def P_j_batch_with_diagnostics(
    x: ArrayLike,
    tau: float,
    params: HestonParams,
    j: J,
    backend: Backend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> HestonIntegralBatchDiagnostics:
    """Evaluate ``P_j`` on a batch of ``x`` values and return diagnostics.

    Parameters
    ----------
    x : array_like
        Array-like log-forward moneyness grid. Scalars are rejected; use
        :func:`P_j_with_diagnostics` for the scalar case.
    tau : float
        Time to expiry in years.
    params : HestonParams
        Heston parameter set.
    j : {0, 1}
        Probability index in the Heston inversion formula.
    backend : {"gauss_legendre", "quad"}, default "gauss_legendre"
        Integration backend.
    quad_cfg : QuadratureConfig, optional
        Fixed-rule configuration for the Gauss-Legendre backend.
    rule : CompositeRule, optional
        Prebuilt fixed rule for the Gauss-Legendre backend.

    Returns
    -------
    HestonIntegralBatchDiagnostics
        Batch diagnostics object with outputs reshaped to match ``x``.

    Raises
    ------
    ValueError
        If ``x`` is scalar, if ``backend`` is unknown, or if ``quad_cfg`` or
        ``rule`` are provided with the ``"quad"`` backend.

    Notes
    -----
    For ``backend="gauss_legendre"``, the integrand and fixed-rule quadrature
    are evaluated in batch over ``x``.

    For ``backend="quad"``, array input is supported as a convenience API only.
    The routine still loops over each element of ``x`` and calls the scalar
    adaptive integrator once per point, so this route is not vectorized.
    """
    x_arr, scalar_input, original_shape = _normalize_x_grid(x)

    if scalar_input:
        raise ValueError(
            "P_j_batch_with_diagnostics requires array-like x, not a scalar. "
            "Use P_j_with_diagnostics for the scalar case."
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

        total_integral = integrals.reshape(original_shape)
        probability = np.asarray(
            _probability_from_integral(total_integral),
            dtype=np.float64,
        ).reshape(original_shape)

        return HestonIntegralBatchDiagnostics(
            backend="quad",
            j=j,
            x=np.asarray(x_arr.reshape(original_shape), dtype=np.float64),
            tau=tau,
            total_integral=np.asarray(total_integral, dtype=np.float64),
            probability=probability,
            panel_contribs=None,
            panel_edges=None,
            quad_error_estimate=np.asarray(
                errors.reshape(original_shape), dtype=np.float64
            ),
        )

    if backend == "gauss_legendre":
        active_rule = _resolve_gauss_rule(quad_cfg=quad_cfg, rule=rule)
        return _integrate_pj_fixed_rule_batch_with_diagnostics(
            x=x_arr.reshape(original_shape),
            tau=tau,
            params=params,
            j=j,
            rule=active_rule,
        )

    raise ValueError(f"Unknown backend: {backend}")
