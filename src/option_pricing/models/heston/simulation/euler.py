"""Full-truncation Euler scheme for Heston simulation."""

import numpy as np

from ....typing import FloatArray
from ..params import HestonParams
from .types import HestonSimulationResult


def _resolve_shocks(
    *,
    shocks: FloatArray,
    n_steps: int,
) -> tuple[int, FloatArray, FloatArray]:

    if shocks.ndim != 3:
        raise ValueError("shocks must have shape (n_paths, n_steps, 2)")

    n_paths_from_shocks, n_steps_from_shocks, n_factors = shocks.shape

    if n_factors != 2:
        raise ValueError("shocks must have final dimension 2")

    if n_steps_from_shocks != n_steps:
        raise ValueError(f"shocks has {n_steps_from_shocks} steps, expected {n_steps}")

    z_x = shocks[:, :, 0]  # shape == (n_paths, n_steps)
    z_v = shocks[:, :, 1]  # shape == (n_paths, n_steps)

    return n_paths_from_shocks, z_x, z_v


def _v_timestep(
    v_t: FloatArray,
    params: HestonParams,
    z_v_j: FloatArray,
    dt: float,
) -> FloatArray:
    kappa, theta, eta = params.kappa, params.vbar, params.eta

    v_pos = np.maximum(v_t, 0.0)

    return np.maximum(
        v_t + kappa * (theta - v_pos) * dt + eta * np.sqrt(v_pos * dt) * z_v_j,
        0.0,
    )


def _x_timestep(
    log_x_t: FloatArray,
    v_t: FloatArray,
    z_x_j: FloatArray,
    dt: float,
    log_drift_step: FloatArray,
) -> FloatArray:
    v_pos = np.maximum(v_t, 0.0)

    log_x_next = (
        log_x_t + log_drift_step - 0.5 * v_pos * dt + np.sqrt(v_pos * dt) * z_x_j
    )

    return log_x_next


def simulate_heston_euler_paths(
    *,
    params: HestonParams,
    x0: float,
    tau: float,
    n_steps: int,
    shocks: FloatArray,
    log_drift_increments: FloatArray | None = None,
) -> HestonSimulationResult:

    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and non-negative")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    dt = tau / n_steps

    x0 = float(x0)
    if not np.isfinite(x0) or x0 <= 0.0:
        raise ValueError("x0 must be finite and positive")

    v0 = float(params.v)
    if not np.isfinite(v0) or v0 < 0.0:
        raise ValueError("v0 must be finite and non-negative")

    z = np.asarray(shocks, dtype=np.float64)
    n_paths, z_x, z_v = _resolve_shocks(shocks=z, n_steps=n_steps)

    if log_drift_increments is None:
        drift = np.zeros(n_steps, dtype=np.float64)
    else:
        drift = np.asarray(log_drift_increments, dtype=np.float64)
        if drift.shape != (n_steps,):
            raise ValueError(
                f"log_drift_increments must have shape ({n_steps},); got {drift.shape}"
            )

    spot_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    var_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    spot_paths[:, 0] = x0  # shape(n_paths, n_steps + 1)
    var_paths[:, 0] = v0  # shape(n_paths, n_steps + 1)

    # Update object
    log_x_t: FloatArray = np.full(n_paths, np.log(x0), dtype=np.float64)  # (n_paths,)
    v_t: FloatArray = np.full(n_paths, v0, dtype=np.float64)  # (n_paths,)

    for j in range(n_steps):

        # v_step
        z_v_j = z_v[:, j]
        assert z_v_j.shape == (n_paths,)

        v_t_dt = _v_timestep(
            v_t=v_t,
            params=params,
            z_v_j=z_v_j,
            dt=dt,
        )

        # x_step
        z_x_j = z_x[:, j]
        assert z_v_j.shape == (n_paths,)

        log_x_t_dt = _x_timestep(
            log_x_t=log_x_t,
            v_t=v_t,
            z_x_j=z_x_j,
            dt=dt,
            log_drift_step=drift[j],
        )

        # Updating logic
        v_t = v_t_dt
        log_x_t = log_x_t_dt
        var_paths[:, j + 1] = v_t_dt
        spot_paths[:, j + 1] = np.exp(log_x_t_dt)

    result = HestonSimulationResult(
        spot_paths=spot_paths,
        var_paths=var_paths,
        dt=dt,
    )

    return result
