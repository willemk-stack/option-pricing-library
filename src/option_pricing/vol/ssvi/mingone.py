from __future__ import annotations

from typing import Literal

import numpy as np

from ..svi.transforms import sigmoid, softplus
from .models import DEFAULT_NUMERICAL_TOL, ESSVINodeSet, MingoneGlobalParams

type ButterflyMode = Literal["GJ"]


def _as_float_vector(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _require_gj(mode: ButterflyMode) -> None:
    if mode != "GJ":
        raise ValueError("Only butterfly_mode='GJ' is implemented in this refactor.")


def decode_rho_raw(raw: np.ndarray, *, rho_cap: float) -> np.ndarray:
    raw_arr = _as_float_vector("rho_raw", raw)
    if not np.isfinite(float(rho_cap)) or not (0.0 < rho_cap < 1.0):
        raise ValueError("rho_cap must lie in (0, 1).")
    return np.asarray(float(rho_cap) * np.tanh(raw_arr), dtype=np.float64)


def decode_theta1_raw(raw: float, *, eps: float = DEFAULT_NUMERICAL_TOL) -> float:
    if not np.isfinite(float(raw)):
        raise ValueError("theta1_raw must be finite.")
    return float(softplus(float(raw)) + float(eps))


def decode_a_raw(raw: np.ndarray, *, eps: float = DEFAULT_NUMERICAL_TOL) -> np.ndarray:
    raw_arr = _as_float_vector("a_raw", raw)
    return np.asarray(softplus(raw_arr) + float(eps), dtype=np.float64)


def decode_c_raw(raw: np.ndarray, *, eps: float = DEFAULT_NUMERICAL_TOL) -> np.ndarray:
    raw_arr = _as_float_vector("c_raw", raw)
    clipped = np.clip(sigmoid(raw_arr), float(eps), 1.0 - float(eps))
    return np.asarray(clipped, dtype=np.float64)


def compute_p_sequence(rho: np.ndarray) -> np.ndarray:
    rho_arr = _as_float_vector("rho", rho)
    if np.any(np.abs(rho_arr) >= 1.0):
        raise ValueError("|rho| must be < 1.")

    p = np.ones_like(rho_arr, dtype=np.float64)
    if rho_arr.size == 1:
        return p

    plus = (1.0 + rho_arr[:-1]) / (1.0 + rho_arr[1:])
    minus = (1.0 - rho_arr[:-1]) / (1.0 - rho_arr[1:])
    p[1:] = np.maximum(plus, minus)
    return np.asarray(p, dtype=np.float64)


def butterfly_upper_bound(
    theta_i: float,
    abs_rho_i: float,
    *,
    mode: ButterflyMode = "GJ",
) -> float:
    _require_gj(mode)
    if theta_i <= 0.0 or not np.isfinite(theta_i):
        raise ValueError("theta_i must be finite and > 0.")
    if not np.isfinite(abs_rho_i) or abs_rho_i < 0.0 or abs_rho_i >= 1.0:
        raise ValueError("abs_rho_i must lie in [0, 1).")

    lee_cap = 4.0 / (1.0 + abs_rho_i)
    gj_cap = np.sqrt((4.0 * theta_i) / (1.0 + abs_rho_i))
    return float(min(lee_cap, gj_cap))


def compute_f_sequence(
    theta: np.ndarray,
    rho: np.ndarray,
    *,
    mode: ButterflyMode = "GJ",
) -> np.ndarray:
    _require_gj(mode)
    theta_arr = _as_float_vector("theta", theta)
    rho_arr = _as_float_vector("rho", rho)
    if theta_arr.size != rho_arr.size:
        raise ValueError("theta and rho must have the same length.")
    return np.asarray(
        [
            butterfly_upper_bound(float(theta_i), abs(float(rho_i)), mode=mode)
            for theta_i, rho_i in zip(theta_arr, rho_arr, strict=True)
        ],
        dtype=np.float64,
    )


def compute_global_psi_caps(
    theta: np.ndarray,
    rho: np.ndarray,
    *,
    mode: ButterflyMode = "GJ",
) -> np.ndarray:
    theta_arr = _as_float_vector("theta", theta)
    rho_arr = _as_float_vector("rho", rho)
    if theta_arr.size != rho_arr.size:
        raise ValueError("theta and rho must have the same length.")

    p = compute_p_sequence(rho_arr)
    f = compute_f_sequence(theta_arr, rho_arr, mode=mode)

    cumprod = np.ones_like(p, dtype=np.float64)
    if p.size > 1:
        cumprod[1:] = np.cumprod(p[1:])

    scaled = f / cumprod
    suffix_min = np.minimum.accumulate(scaled[::-1])[::-1]
    return np.asarray(cumprod * suffix_min, dtype=np.float64)


def compute_Apsi_Cpsi(
    theta: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    *,
    mode: ButterflyMode = "GJ",
) -> tuple[np.ndarray, np.ndarray]:
    theta_arr = _as_float_vector("theta", theta)
    rho_arr = _as_float_vector("rho", rho)
    psi_arr = _as_float_vector("psi", psi)
    if not (theta_arr.size == rho_arr.size == psi_arr.size):
        raise ValueError("theta, rho, and psi must have the same length.")

    p = compute_p_sequence(rho_arr)
    global_caps = compute_global_psi_caps(theta_arr, rho_arr, mode=mode)

    A = np.zeros_like(theta_arr, dtype=np.float64)
    C = np.empty_like(theta_arr, dtype=np.float64)
    C[0] = float(global_caps[0])

    for i in range(1, theta_arr.size):
        A[i] = float(psi_arr[i - 1] * p[i])
        calendar_cap = float(psi_arr[i - 1] * theta_arr[i] / theta_arr[i - 1])
        C[i] = float(min(calendar_cap, global_caps[i]))

    return np.asarray(A, dtype=np.float64), np.asarray(C, dtype=np.float64)


def reconstruct_nodes_from_global_params(
    params: MingoneGlobalParams,
    *,
    mode: ButterflyMode = "GJ",
) -> ESSVINodeSet:
    _require_gj(mode)

    rho = decode_rho_raw(params.rho_raw, rho_cap=float(params.rho_cap))
    theta1 = decode_theta1_raw(params.theta1_raw, eps=params.eps)
    a = (
        np.asarray([], dtype=np.float64)
        if params.size == 1
        else decode_a_raw(params.a_raw, eps=params.eps)
    )
    c = decode_c_raw(params.c_raw, eps=params.eps)
    p = compute_p_sequence(rho)

    theta = np.empty(params.size, dtype=np.float64)
    theta[0] = float(theta1)
    for i in range(1, params.size):
        theta[i] = float(theta[i - 1] * p[i] + a[i - 1])

    global_caps = compute_global_psi_caps(theta, rho, mode=mode)
    psi = np.empty(params.size, dtype=np.float64)
    A = np.zeros(params.size, dtype=np.float64)
    C = np.empty(params.size, dtype=np.float64)

    C[0] = float(global_caps[0])
    if C[0] <= float(params.eps):
        raise ValueError("The first Mingone psi interval is empty.")
    psi[0] = float(A[0] + c[0] * (C[0] - A[0]))

    for i in range(1, params.size):
        A[i] = float(psi[i - 1] * p[i])
        calendar_cap = float(psi[i - 1] * theta[i] / theta[i - 1])
        C[i] = float(min(calendar_cap, global_caps[i]))
        if C[i] <= A[i] + float(params.eps):
            raise ValueError(
                "The Mingone admissible psi interval is empty at node "
                f"{i} (A_psi={A[i]:.8g}, C_psi={C[i]:.8g})."
            )
        psi[i] = float(A[i] + c[i] * (C[i] - A[i]))

    return ESSVINodeSet(
        expiries=params.expiries,
        theta=theta,
        psi=psi,
        rho=rho,
        eps=params.eps,
    )
