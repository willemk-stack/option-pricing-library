from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ...typing import ArrayLike
from .models import DEFAULT_NUMERICAL_TOL, ESSVINodeSet


class ESSVIInterpolatedState(NamedTuple):
    theta: np.ndarray
    psi: np.ndarray
    rho: np.ndarray
    eta: np.ndarray


def _as_float_query(name: str, value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and > 0.")
    return arr


def right_theta_slope(
    nodes: ESSVINodeSet,
    *,
    eps: float = DEFAULT_NUMERICAL_TOL,
) -> float:
    if nodes.expiries.size == 1:
        return float(max(nodes.theta[0] / nodes.expiries[0], float(eps)))

    raw = (nodes.theta[-1] - nodes.theta[-2]) / (
        nodes.expiries[-1] - nodes.expiries[-2]
    )
    return float(max(raw, float(eps)))


def interpolate_nodes(
    nodes: ESSVINodeSet,
    T: ArrayLike,
    *,
    eps: float | None = None,
) -> ESSVIInterpolatedState:
    T_arr = _as_float_query("T", T)
    flat_T = np.ravel(T_arr)

    theta = np.empty_like(flat_T, dtype=np.float64)
    psi = np.empty_like(flat_T, dtype=np.float64)
    rho = np.empty_like(flat_T, dtype=np.float64)

    chi_nodes = nodes.eta
    slope_right = right_theta_slope(nodes, eps=nodes.eps if eps is None else eps)

    for idx, tau in enumerate(flat_T):
        if tau < nodes.expiries[0]:
            lam = tau / nodes.expiries[0]
            theta[idx] = lam * nodes.theta[0]
            psi[idx] = lam * nodes.psi[0]
            rho[idx] = nodes.rho[0]
            continue

        if tau > nodes.expiries[-1]:
            theta[idx] = nodes.theta[-1] + slope_right * (tau - nodes.expiries[-1])
            psi[idx] = nodes.psi[-1]
            rho[idx] = nodes.rho[-1]
            continue

        exact = np.flatnonzero(np.isclose(nodes.expiries, tau, rtol=0.0, atol=1e-12))
        if exact.size:
            i = int(exact[0])
            theta[idx] = nodes.theta[i]
            psi[idx] = nodes.psi[i]
            rho[idx] = nodes.rho[i]
            continue

        j = int(np.searchsorted(nodes.expiries, tau, side="right"))
        i = j - 1
        lam = (tau - nodes.expiries[i]) / (nodes.expiries[j] - nodes.expiries[i])

        theta[idx] = (1.0 - lam) * nodes.theta[i] + lam * nodes.theta[j]
        psi[idx] = (1.0 - lam) * nodes.psi[i] + lam * nodes.psi[j]
        chi = (1.0 - lam) * chi_nodes[i] + lam * chi_nodes[j]
        rho[idx] = chi / psi[idx]

    eta = psi * rho
    shape = T_arr.shape
    return ESSVIInterpolatedState(
        theta=np.asarray(theta.reshape(shape), dtype=np.float64),
        psi=np.asarray(psi.reshape(shape), dtype=np.float64),
        rho=np.asarray(rho.reshape(shape), dtype=np.float64),
        eta=np.asarray(eta.reshape(shape), dtype=np.float64),
    )
