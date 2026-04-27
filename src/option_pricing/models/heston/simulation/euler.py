"""Full-truncation Euler scheme for Heston simulation."""

import numpy as np

from ....typing import FloatArray
from ..params import HestonParams
from .shared import (
    _initialize_paths,
    _resolve_log_drift_increments,
    _resolve_shocks,
    _validate_initial_state,
    _validate_time_grid,
)
from .types import HestonSimulationResult


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
    dt = _validate_time_grid(
        tau=tau,
        n_steps=n_steps,
        allow_zero_tau=True,
    )
    x0, v0 = _validate_initial_state(x0=x0, v0=params.v)

    n_paths, z_x, z_v = _resolve_shocks(shocks=shocks, n_steps=n_steps)
    drift = _resolve_log_drift_increments(
        log_drift_increments=log_drift_increments,
        n_steps=n_steps,
        n_paths=n_paths,
        allow_pathwise=False,
    )
    spot_paths, var_paths, log_x_t, v_t = _initialize_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        x0=x0,
        v0=v0,
    )

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
