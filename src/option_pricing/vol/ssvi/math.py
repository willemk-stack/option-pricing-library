from __future__ import annotations

import numpy as np

from ...typing import ArrayLike
from .models import ESSVITermStructures


def _as_float_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _guard_nonzero(name: str, value: np.ndarray, eps: float) -> None:
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} is undefined because it is not finite.")
    if np.any(np.abs(value) <= eps):
        raise ValueError(f"{name} is undefined because abs({name}) <= {eps:g}.")


def _surface_state(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params.validate(T)
    y_arr = _as_float_array(y)
    theta = _as_float_array(params.theta(T))
    psi = _as_float_array(params.psi(T))
    eta = _as_float_array(params.eta(T))
    y_arr, theta, psi, eta = np.broadcast_arrays(y_arr, theta, psi, eta)
    return (
        np.asarray(y_arr, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
        np.asarray(psi, dtype=np.float64),
        np.asarray(eta, dtype=np.float64),
    )


def _surface_state_with_T_derivs(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> tuple[np.ndarray, ...]:
    y_arr, theta, psi, eta = _surface_state(y=y, params=params, T=T)
    dtheta = _as_float_array(params.dtheta_dT(T))
    dpsi = _as_float_array(params.dpsi_dT(T))
    deta = _as_float_array(params.deta_dT(T))
    y_arr, theta, psi, eta, dtheta, dpsi, deta = np.broadcast_arrays(
        y_arr, theta, psi, eta, dtheta, dpsi, deta
    )
    return tuple(
        np.asarray(arr, dtype=np.float64)
        for arr in (y_arr, theta, psi, eta, dtheta, dpsi, deta)
    )


def _surface_state_with_second_T_derivs(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> tuple[np.ndarray, ...]:
    y_arr, theta, psi, eta, dtheta, dpsi, deta = _surface_state_with_T_derivs(
        y=y,
        params=params,
        T=T,
    )
    d2theta = _as_float_array(params.d2theta_dT2(T))
    d2psi = _as_float_array(params.d2psi_dT2(T))
    d2eta = _as_float_array(params.d2eta_dT2(T))
    arrays = np.broadcast_arrays(
        y_arr,
        theta,
        psi,
        eta,
        dtheta,
        dpsi,
        deta,
        d2theta,
        d2psi,
        d2eta,
    )
    return tuple(np.asarray(arr, dtype=np.float64) for arr in arrays)


def _radicant_from_state(
    y: np.ndarray,
    theta: np.ndarray,
    psi: np.ndarray,
    eta: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    q = theta * theta + 2.0 * theta * eta * y + psi * psi * y * y
    if np.any(q < -eps):
        raise ValueError(
            "The eSSVI radicant became negative; check theta/psi/eta consistency."
        )
    q = np.where((q < 0.0) & (q >= -eps), 0.0, q)
    with np.errstate(invalid="raise"):
        return np.asarray(np.sqrt(q), dtype=np.float64)


def radicant_D(y: ArrayLike, params: ESSVITermStructures, T: ArrayLike) -> np.ndarray:
    y_arr, theta, psi, eta = _surface_state(y=y, params=params, T=T)
    return _radicant_from_state(y_arr, theta, psi, eta, eps=params.eps)


def essvi_total_variance(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> np.ndarray:
    y_arr, theta, psi, eta = _surface_state(y=y, params=params, T=T)
    D = _radicant_from_state(y_arr, theta, psi, eta, eps=params.eps)
    return np.asarray(0.5 * (theta + eta * y_arr + D), dtype=np.float64)


def essvi_total_variance_dk(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> np.ndarray:
    y_arr, theta, psi, eta = _surface_state(y=y, params=params, T=T)
    D = _radicant_from_state(y_arr, theta, psi, eta, eps=params.eps)
    _guard_nonzero("D", D, params.eps)
    A = theta * eta + psi * psi * y_arr
    return np.asarray(0.5 * (eta + A / D), dtype=np.float64)


def essvi_total_variance_dkk(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> np.ndarray:
    y_arr, theta, psi, eta = _surface_state(y=y, params=params, T=T)
    D = _radicant_from_state(y_arr, theta, psi, eta, eps=params.eps)
    _guard_nonzero("D", D, params.eps)
    A = theta * eta + psi * psi * y_arr
    return np.asarray(0.5 * (psi * psi / D - (A * A) / (D * D * D)), dtype=np.float64)


def radicant_dT(y: ArrayLike, params: ESSVITermStructures, T: ArrayLike) -> np.ndarray:
    y_arr, theta, psi, eta, dtheta, dpsi, deta = _surface_state_with_T_derivs(
        y=y,
        params=params,
        T=T,
    )
    D = _radicant_from_state(y_arr, theta, psi, eta, eps=params.eps)
    _guard_nonzero("D", D, params.eps)
    B = (
        theta * dtheta
        + (dtheta * eta + theta * deta) * y_arr
        + psi * dpsi * y_arr * y_arr
    )
    return np.asarray(B / D, dtype=np.float64)


def essvi_total_variance_dT(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> np.ndarray:
    y_arr, _, _, _, dtheta, _, deta = _surface_state_with_T_derivs(
        y=y,
        params=params,
        T=T,
    )
    dD = radicant_dT(y=y_arr, params=params, T=T)
    return np.asarray(0.5 * (dtheta + deta * y_arr + dD), dtype=np.float64)


def essvi_total_variance_dk_dT(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> np.ndarray:
    y_arr, theta, psi, eta, dtheta, dpsi, deta = _surface_state_with_T_derivs(
        y=y,
        params=params,
        T=T,
    )
    D = _radicant_from_state(y_arr, theta, psi, eta, eps=params.eps)
    _guard_nonzero("D", D, params.eps)

    A = theta * eta + psi * psi * y_arr
    A_T = dtheta * eta + theta * deta + 2.0 * psi * dpsi * y_arr
    B = (
        theta * dtheta
        + (dtheta * eta + theta * deta) * y_arr
        + psi * dpsi * y_arr * y_arr
    )
    return np.asarray(0.5 * (deta + A_T / D - (A * B) / (D * D * D)), dtype=np.float64)


def radicant_dTT(y: ArrayLike, params: ESSVITermStructures, T: ArrayLike) -> np.ndarray:
    (
        y_arr,
        theta,
        psi,
        eta,
        dtheta,
        dpsi,
        deta,
        d2theta,
        d2psi,
        d2eta,
    ) = _surface_state_with_second_T_derivs(y=y, params=params, T=T)
    D = _radicant_from_state(y_arr, theta, psi, eta, eps=params.eps)
    _guard_nonzero("D", D, params.eps)
    B = (
        theta * dtheta
        + (dtheta * eta + theta * deta) * y_arr
        + psi * dpsi * y_arr * y_arr
    )
    C = (
        dtheta * dtheta
        + theta * d2theta
        + (d2theta * eta + 2.0 * dtheta * deta + theta * d2eta) * y_arr
        + (dpsi * dpsi + psi * d2psi) * y_arr * y_arr
    )
    return np.asarray(C / D - (B * B) / (D * D * D), dtype=np.float64)


def essvi_total_variance_dTT(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> np.ndarray:
    y_arr, _, _, _, _, _, _, d2theta, _, d2eta = _surface_state_with_second_T_derivs(
        y=y,
        params=params,
        T=T,
    )
    d2D = radicant_dTT(y=y_arr, params=params, T=T)
    return np.asarray(0.5 * (d2theta + d2eta * y_arr + d2D), dtype=np.float64)


def essvi_w_and_derivs(
    y: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        essvi_total_variance(y=y, params=params, T=T),
        essvi_total_variance_dk(y=y, params=params, T=T),
        essvi_total_variance_dkk(y=y, params=params, T=T),
        essvi_total_variance_dT(y=y, params=params, T=T),
    )
