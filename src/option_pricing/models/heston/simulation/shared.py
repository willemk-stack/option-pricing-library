"""Simulation helpers shared across Heston schemes."""

from __future__ import annotations

import numpy as np

from ....typing import FloatArray


def _validate_time_grid(
    *,
    tau: float,
    n_steps: int,
    allow_zero_tau: bool,
) -> float:
    tau_value = float(tau)
    tau_is_invalid = (
        not np.isfinite(tau_value)
        or tau_value < 0.0
        or (not allow_zero_tau and tau_value == 0.0)
    )

    if tau_is_invalid:
        if allow_zero_tau:
            raise ValueError("tau must be finite and non-negative")
        raise ValueError("tau must be finite and positive")

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    return tau_value / n_steps


def _validate_initial_state(
    *,
    x0: float,
    v0: float,
) -> tuple[float, float]:
    x0_value = float(x0)
    if not np.isfinite(x0_value) or x0_value <= 0.0:
        raise ValueError("x0 must be finite and positive")

    v0_value = float(v0)
    if not np.isfinite(v0_value) or v0_value < 0.0:
        raise ValueError("v0 must be finite and non-negative")

    return x0_value, v0_value


def _resolve_factor_shocks(
    *,
    shocks: FloatArray,
    n_steps: int,
    n_factors: int,
) -> tuple[int, tuple[FloatArray, ...]]:
    shock_array = np.asarray(shocks, dtype=np.float64)

    if shock_array.ndim != 3:
        raise ValueError(f"shocks must have shape (n_paths, n_steps, {n_factors})")

    (
        n_paths_from_shocks,
        n_steps_from_shocks,
        n_factors_from_shocks,
    ) = shock_array.shape

    if n_factors_from_shocks != n_factors:
        raise ValueError(f"shocks must have final dimension {n_factors}")

    if n_steps_from_shocks != n_steps:
        raise ValueError(f"shocks has {n_steps_from_shocks} steps, expected {n_steps}")

    factor_slices = tuple(
        shock_array[:, :, factor_index] for factor_index in range(n_factors)
    )
    return n_paths_from_shocks, factor_slices


def _resolve_shocks(
    *,
    shocks: FloatArray,
    n_steps: int,
) -> tuple[int, FloatArray, FloatArray]:
    n_paths, factor_slices = _resolve_factor_shocks(
        shocks=shocks,
        n_steps=n_steps,
        n_factors=2,
    )
    z_x, z_v = factor_slices
    return n_paths, z_x, z_v


def _resolve_qe_shocks(
    *,
    shocks: FloatArray,
    n_steps: int,
) -> tuple[int, FloatArray, FloatArray, FloatArray]:
    n_paths, factor_slices = _resolve_factor_shocks(
        shocks=shocks,
        n_steps=n_steps,
        n_factors=3,
    )
    z_v, u_v, z_x = factor_slices
    return n_paths, z_v, u_v, z_x


def _resolve_log_drift_increments(
    *,
    log_drift_increments: FloatArray | None,
    n_steps: int,
    n_paths: int,
    allow_pathwise: bool,
) -> FloatArray:
    if log_drift_increments is None:
        return np.zeros(n_steps, dtype=np.float64)

    drift = np.asarray(log_drift_increments, dtype=np.float64)

    if allow_pathwise:
        valid_shapes = {(n_steps,), (n_paths, n_steps)}
        error_message = (
            "log_drift_increments must have shape "
            f"({n_steps},) or ({n_paths}, {n_steps}); got {drift.shape}"
        )
    else:
        valid_shapes = {(n_steps,)}
        error_message = (
            f"log_drift_increments must have shape ({n_steps},); got {drift.shape}"
        )

    if drift.shape not in valid_shapes:
        raise ValueError(error_message)

    return drift


def _drift_step(
    *,
    log_drift_increments: FloatArray,
    step_index: int,
) -> FloatArray | float:
    if log_drift_increments.ndim == 1:
        return float(log_drift_increments[step_index])
    return log_drift_increments[:, step_index]


def _initialize_paths(
    *,
    n_paths: int,
    n_steps: int,
    x0: float,
    v0: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    spot_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    var_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    log_x_t = np.full(n_paths, np.log(x0), dtype=np.float64)
    v_t = np.full(n_paths, v0, dtype=np.float64)

    spot_paths[:, 0] = x0
    var_paths[:, 0] = v0

    return spot_paths, var_paths, log_x_t, v_t
