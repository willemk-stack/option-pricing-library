"""Andersen QE scheme for Heston simulation."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ....typing import FloatArray
from ..params import HestonParams
from .shared import (
    _drift_step,
    _initialize_paths,
    _resolve_log_drift_increments,
    _resolve_qe_shocks,
    _validate_initial_state,
    _validate_time_grid,
)
from .types import HestonSimulationResult

type BoolArray = NDArray[np.bool_]


def _as_float_array(array: object) -> FloatArray:
    return cast(FloatArray, np.asarray(array, dtype=np.float64))


def _as_bool_array(array: object) -> BoolArray:
    return cast(BoolArray, np.asarray(array, dtype=np.bool_))


def _m(
    v_t: FloatArray,
    params: HestonParams,
    dt: float,
) -> FloatArray:
    """Exact conditional mean of V_{t+dt} given V_t."""

    return _as_float_array(
        params.vbar + (v_t - params.vbar) * np.exp(-params.kappa * dt)
    )


def _s2(
    v_t: FloatArray,
    params: HestonParams,
    dt: float,
) -> FloatArray:
    """Exact conditional variance of V_{t+dt} given V_t."""

    one_minus_exp = -np.expm1(-params.kappa * dt)
    exp_neg = 1.0 - one_minus_exp

    term1 = v_t * params.eta**2 * exp_neg * one_minus_exp / params.kappa

    term2 = params.vbar * params.eta**2 * one_minus_exp**2 / (2.0 * params.kappa)

    return _as_float_array(term1 + term2)


def _psi(
    v_t: FloatArray,
    params: HestonParams,
    dt: float,
) -> FloatArray:
    """QE switching statistic psi = s2 / m^2."""

    m = _m(v_t=v_t, params=params, dt=dt)
    s2 = _s2(v_t=v_t, params=params, dt=dt)

    return _as_float_array(s2 / (m**2))


def _validate_psi_c(psi_c: float) -> None:
    if not 1.0 <= psi_c <= 2.0:
        raise ValueError(f"psi_c must be in [1, 2]. Got psi_c={psi_c}.")


def _qe_quadratic_ab(
    m: FloatArray,
    psi: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Compute QE quadratic branch coefficients.

    Returns
    -------
    a, b2
        The branch samples V_next = a * (sqrt(b2) + Z)^2.
    """

    if np.any(psi > 2.0):
        raise ValueError(
            "The quadratic branch requires psi <= 2. " f"Got max(psi) = {np.max(psi)}."
        )

    b2 = (2.0 / psi) - 1.0 + np.sqrt(2.0 / psi) * np.sqrt((2.0 / psi) - 1.0)

    a = m / (1.0 + b2)

    return _as_float_array(a), _as_float_array(b2)


def _qe_exponential_p_beta(
    m: FloatArray,
    psi: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Compute QE exponential branch coefficients.

    Returns
    -------
    p, beta
        Atom-at-zero probability and exponential rate.
    """

    if np.any(psi < 1.0):
        raise ValueError(
            "The exponential branch requires psi >= 1. "
            f"Got min(psi) = {np.min(psi)}."
        )

    p = (psi - 1.0) / (psi + 1.0)
    beta = (1.0 - p) / m

    return _as_float_array(p), _as_float_array(beta)


def _v_timestep_quadratic(
    v_t: FloatArray,
    params: HestonParams,
    z_v_j: FloatArray,
    dt: float,
) -> FloatArray:
    """
    Andersen QE quadratic variance branch.

    Parameters
    ----------
    z_v_j
        Standard normal shocks.
    """

    m = _m(v_t=v_t, params=params, dt=dt)
    psi = _psi(v_t=v_t, params=params, dt=dt)

    a, b2 = _qe_quadratic_ab(m=m, psi=psi)
    b = np.sqrt(b2)

    return _as_float_array(a * (b + z_v_j) ** 2)


def _v_timestep_exponential(
    v_t: FloatArray,
    params: HestonParams,
    u_v_j: FloatArray,
    dt: float,
) -> FloatArray:
    """
    Andersen QE exponential variance branch.

    Parameters
    ----------
    u_v_j
        Uniform(0, 1) shocks.

    Notes
    -----
    The positive log form is intentional:

        V_next = (1 / beta) * log((1 - p) / (1 - U))

    for U > p.
    """

    m = _m(v_t=v_t, params=params, dt=dt)
    psi = _psi(v_t=v_t, params=params, dt=dt)

    p, beta = _qe_exponential_p_beta(m=m, psi=psi)

    u_v_j = _as_float_array(u_v_j)
    u_v_j = _as_float_array(np.clip(u_v_j, 0.0, np.nextafter(1.0, 0.0)))

    return _as_float_array(
        np.where(
            u_v_j <= p,
            0.0,
            (1.0 / beta) * np.log((1.0 - p) / (1.0 - u_v_j)),
        )
    )


def _v_timestep_qe(
    v_t: FloatArray,
    params: HestonParams,
    z_v_j: FloatArray,
    u_v_j: FloatArray,
    dt: float,
    psi_c: float = 1.5,
) -> FloatArray:
    """
    Andersen QE variance timestep.

    Uses
    ----
    quadratic branch:
        psi <= psi_c

    exponential branch:
        psi > psi_c

    Parameters
    ----------
    z_v_j
        Standard normal shocks for the quadratic branch.

    u_v_j
        Uniform(0, 1) shocks for the exponential branch.

    Notes
    -----
    This follows the library-style sampling logic: shocks are generated upstream
    and consumed here.
    """

    _validate_psi_c(psi_c)

    psi = _psi(v_t=v_t, params=params, dt=dt)

    v_t_arr, z_v_j_arr, u_v_j_arr, psi_arr = np.broadcast_arrays(
        v_t,
        z_v_j,
        u_v_j,
        psi,
    )
    v_t = _as_float_array(v_t_arr)
    z_v_j = _as_float_array(z_v_j_arr)
    u_v_j = _as_float_array(u_v_j_arr)
    psi = _as_float_array(psi_arr)

    v_next = _as_float_array(np.empty_like(psi, dtype=np.float64))

    quadratic_mask = _as_bool_array(psi <= psi_c)
    exponential_mask = _as_bool_array(np.logical_not(quadratic_mask))

    if np.any(quadratic_mask):
        v_next[quadratic_mask] = _v_timestep_quadratic(
            v_t=v_t[quadratic_mask],
            params=params,
            z_v_j=z_v_j[quadratic_mask],
            dt=dt,
        )

    if np.any(exponential_mask):
        v_next[exponential_mask] = _v_timestep_exponential(
            v_t=v_t[exponential_mask],
            params=params,
            u_v_j=u_v_j[exponential_mask],
            dt=dt,
        )

    return _as_float_array(v_next)


def _qe_log_coefficients(
    params: HestonParams,
    dt: float,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """
    Andersen QE log-price coefficients.

    Returns
    -------
    K0, K1, K2, K3, K4
    """

    if params.eta <= 0.0:
        raise ValueError("params.eta must be positive.")

    if abs(params.rho) > 1.0:
        raise ValueError(f"rho must be in [-1, 1]. Got rho={params.rho}.")

    if gamma1 < 0.0 or gamma2 < 0.0:
        raise ValueError("gamma1 and gamma2 must be nonnegative.")

    rho = params.rho
    kappa = params.kappa
    theta = params.vbar
    eta = params.eta

    kappa_rho_over_eta = kappa * rho / eta

    k0 = -rho * kappa * theta * dt / eta

    k1 = gamma1 * dt * (kappa_rho_over_eta - 0.5) - rho / eta

    k2 = gamma2 * dt * (kappa_rho_over_eta - 0.5) + rho / eta

    k3 = gamma1 * dt * (1.0 - rho**2)
    k4 = gamma2 * dt * (1.0 - rho**2)

    return k0, k1, k2, k3, k4


def _integrated_variance_deterministic(
    v_t: FloatArray,
    params: HestonParams,
    dt: float,
) -> FloatArray:
    """Exact integrated variance over one step when eta = 0."""

    one_minus_exp = -np.expm1(-params.kappa * dt)
    return _as_float_array(
        params.vbar * dt + (v_t - params.vbar) * (one_minus_exp / params.kappa)
    )


def _x_timestep_qe_deterministic_variance(
    log_x_t: FloatArray,
    v_t: FloatArray,
    params: HestonParams,
    z_x_j: FloatArray,
    dt: float,
    log_drift_step: FloatArray | float = 0.0,
) -> FloatArray:
    """Exact log-price step for the deterministic-variance eta = 0 limit."""

    integrated_variance = _integrated_variance_deterministic(
        v_t=v_t,
        params=params,
        dt=dt,
    )
    return _as_float_array(
        log_x_t
        + log_drift_step
        - 0.5 * integrated_variance
        + np.sqrt(np.maximum(integrated_variance, 0.0)) * z_x_j
    )


def _qe_branch_params(
    v_t: FloatArray,
    params: HestonParams,
    dt: float,
    psi_c: float = 1.5,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """
    Return m, psi, and the quadratic-branch mask.

    The branch mask must use the same psi_c as the variance timestep.
    """

    _validate_psi_c(psi_c)

    m = _m(v_t=v_t, params=params, dt=dt)
    psi = _psi(v_t=v_t, params=params, dt=dt)

    quadratic_mask = _as_bool_array(psi <= psi_c)

    return m, psi, quadratic_mask


def _qe_martingale_k0_star(
    v_t: FloatArray,
    params: HestonParams,
    dt: float,
    psi_c: float = 1.5,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
) -> FloatArray:
    """
    Andersen QE martingale correction.

    Computes

        K0* = -log(M) - (K1 + 0.5 K3) V_t

    where

        M = E[exp(A V_{t+dt}) | V_t]
        A = K2 + 0.5 K4

    Regularity checks
    -----------------
    Quadratic branch:
        A < 1 / (2a)

    Exponential branch:
        A < beta

    These checks matter especially for rho > 0. The function raises instead of
    applying the martingale correction blindly when the exponential moment is
    not finite.
    """

    _, k1, k2, k3, k4 = _qe_log_coefficients(
        params=params,
        dt=dt,
        gamma1=gamma1,
        gamma2=gamma2,
    )

    a_exp = k2 + 0.5 * k4

    v_t = _as_float_array(v_t)

    m, psi, quadratic_mask = _qe_branch_params(
        v_t=v_t,
        params=params,
        dt=dt,
        psi_c=psi_c,
    )

    v_t_arr, m_arr, psi_arr, quadratic_mask_arr = np.broadcast_arrays(
        v_t,
        m,
        psi,
        quadratic_mask,
    )
    v_t = _as_float_array(v_t_arr)
    m = _as_float_array(m_arr)
    psi = _as_float_array(psi_arr)
    quadratic_mask = _as_bool_array(quadratic_mask_arr)

    log_m = _as_float_array(np.empty_like(v_t, dtype=np.float64))

    exponential_mask = _as_bool_array(np.logical_not(quadratic_mask))

    if np.any(quadratic_mask):
        m_q = m[quadratic_mask]
        psi_q = psi[quadratic_mask]

        a, b2 = _qe_quadratic_ab(m=m_q, psi=psi_q)

        # Need A < 1 / (2a), equivalently 1 - 2 A a > 0.
        denom = 1.0 - 2.0 * a_exp * a

        if np.any(denom <= 0.0):
            raise ValueError(
                "QE martingale correction is invalid in the quadratic branch: "
                "requires A < 1 / (2a). "
                f"rho={params.rho}, A={a_exp}, "
                f"min(1 - 2*A*a)={np.min(denom)}."
            )

        # If V = a * (b + Z)^2, then
        #
        # E[exp(A V)]
        #   = (1 - 2 A a)^(-1/2)
        #     * exp(A a b^2 / (1 - 2 A a)).
        log_m[quadratic_mask] = _as_float_array(
            -0.5 * np.log(denom) + (a_exp * a * b2) / denom
        )

    if np.any(exponential_mask):
        m_e = m[exponential_mask]
        psi_e = psi[exponential_mask]

        p, beta = _qe_exponential_p_beta(m=m_e, psi=psi_e)

        # Need A < beta.
        if np.any(a_exp >= beta):
            raise ValueError(
                "QE martingale correction is invalid in the exponential branch: "
                "requires A < beta. "
                f"rho={params.rho}, A={a_exp}, "
                f"min(beta - A)={np.min(beta - a_exp)}."
            )

        # If V has atom p at zero plus exponential tail with rate beta, then
        #
        # E[exp(A V)] = p + (1 - p) * beta / (beta - A).
        m_exp = _as_float_array(p + (1.0 - p) * beta / (beta - a_exp))

        if np.any(m_exp <= 0.0):
            raise ValueError(
                "QE martingale correction produced a non-positive exponential moment."
            )

        log_m[exponential_mask] = _as_float_array(np.log(m_exp))

    return _as_float_array(-log_m - (k1 + 0.5 * k3) * v_t)


def _x_timestep_qe(
    log_x_t: FloatArray,
    v_t: FloatArray,
    v_next: FloatArray,
    params: HestonParams,
    z_x_j: FloatArray,
    dt: float,
    log_drift_step: FloatArray | float = 0.0,
    psi_c: float = 1.5,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
    martingale_correction: bool = True,
) -> FloatArray:
    """
    Andersen QE log-price timestep.

    Parameters
    ----------
    z_x_j
        Standard normal shocks independent of the random numbers used to
        generate v_next.

    martingale_correction
        If True, replaces K0 by Andersen's K0* so that

            E[X_{t+dt} | X_t] = X_t

        when log_drift_step = 0.

        If log_drift_step is nonzero, the correction gives

            E[X_{t+dt} | X_t] = X_t * exp(log_drift_step).
    """

    k0, k1, k2, k3, k4 = _qe_log_coefficients(
        params=params,
        dt=dt,
        gamma1=gamma1,
        gamma2=gamma2,
    )

    if martingale_correction:
        k0_used = _qe_martingale_k0_star(
            v_t=v_t,
            params=params,
            dt=dt,
            psi_c=psi_c,
            gamma1=gamma1,
            gamma2=gamma2,
        )
    else:
        k0_used = _as_float_array(np.full_like(v_t, fill_value=k0, dtype=np.float64))

    variance_term = k3 * v_t + k4 * v_next
    variance_term = _as_float_array(np.maximum(variance_term, 0.0))

    return _as_float_array(
        log_x_t
        + log_drift_step
        + k0_used
        + k1 * v_t
        + k2 * v_next
        + np.sqrt(variance_term) * z_x_j
    )


def simulate_heston_qe_paths(
    *,
    params: HestonParams,
    x0: float,
    tau: float,
    n_steps: int,
    shocks: FloatArray,
    log_drift_increments: FloatArray | None = None,
    psi_c: float = 1.5,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
    martingale_correction: bool = True,
) -> HestonSimulationResult:
    """
    Simulate Heston paths with Andersen's QE scheme.

    Expected shocks shape
    ---------------------
    shocks.shape == (n_paths, n_steps, 3)

    with

        shocks[:, :, 0] = z_v, standard normal shocks for V quadratic branch
        shocks[:, :, 1] = u_v, uniform(0, 1) shocks for V exponential branch
        shocks[:, :, 2] = z_x, standard normal shocks for log X

    The z_x shocks should be independent of the shocks used for V.

    Parameters
    ----------
    log_drift_increments
        Optional deterministic log-drift increments. Use shape (n_steps,) for
        common increments or (n_paths, n_steps) for pathwise increments.

        With martingale_correction=True, conditional expectation becomes

            E[X_{t+dt} | X_t] = X_t * exp(log_drift_step).
    """

    dt = _validate_time_grid(
        tau=tau,
        n_steps=n_steps,
        allow_zero_tau=False,
    )
    x0, v0 = _validate_initial_state(x0=x0, v0=params.v)
    n_paths, z_v, u_v, z_x = _resolve_qe_shocks(
        shocks=shocks,
        n_steps=n_steps,
    )

    if np.any((u_v < 0.0) | (u_v >= 1.0)):
        raise ValueError("u_v shocks must be in [0, 1).")

    log_drift_increments_arr = _resolve_log_drift_increments(
        log_drift_increments=log_drift_increments,
        n_steps=n_steps,
        n_paths=n_paths,
        allow_pathwise=True,
    )
    spot_paths, var_paths, log_x_t, v_t = _initialize_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        x0=x0,
        v0=v0,
    )
    deterministic_variance = params.eta == 0.0

    for j in range(n_steps):
        log_drift_step = _drift_step(
            log_drift_increments=log_drift_increments_arr,
            step_index=j,
        )

        if deterministic_variance:
            v_next = _m(v_t=v_t, params=params, dt=dt)
            log_x_t = _x_timestep_qe_deterministic_variance(
                log_x_t=log_x_t,
                v_t=v_t,
                params=params,
                z_x_j=z_x[:, j],
                dt=dt,
                log_drift_step=log_drift_step,
            )
        else:
            v_next = _v_timestep_qe(
                v_t=v_t,
                params=params,
                z_v_j=z_v[:, j],
                u_v_j=u_v[:, j],
                dt=dt,
                psi_c=psi_c,
            )

            log_x_t = _x_timestep_qe(
                log_x_t=log_x_t,
                v_t=v_t,
                v_next=v_next,
                params=params,
                z_x_j=z_x[:, j],
                dt=dt,
                log_drift_step=log_drift_step,
                psi_c=psi_c,
                gamma1=gamma1,
                gamma2=gamma2,
                martingale_correction=martingale_correction,
            )

        v_t = v_next
        var_paths[:, j + 1] = v_t
        spot_paths[:, j + 1] = np.exp(log_x_t)

    return HestonSimulationResult(
        spot_paths=spot_paths,
        var_paths=var_paths,
        dt=dt,
    )
