from __future__ import annotations

from typing import overload

import numpy as np
from scipy.stats import norm

from ...market.curves import PricingContext
from ...types import MarketData, OptionType
from ...typing import ArrayLike
from .models import ESSVITermStructures


def _as_float_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _guard_nonzero(name: str, value: np.ndarray, eps: float) -> None:
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} is undefined because it is not finite.")
    if np.any(np.abs(value) <= eps):
        raise ValueError(f"{name} is undefined because abs({name}) <= {eps:g}.")


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def _forward_df_from_market(
    market: MarketData | PricingContext,
    T: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    ctx = _to_ctx(market)
    T_arr = _as_float_array(T)
    flat_T = np.ravel(T_arr)
    forward = np.asarray([ctx.fwd(float(tau)) for tau in flat_T], dtype=np.float64)
    df = np.asarray([ctx.df(float(tau)) for tau in flat_T], dtype=np.float64)
    return forward.reshape(T_arr.shape), df.reshape(T_arr.shape)


def _black76_price_broadcast(
    *,
    kind: OptionType,
    forward: ArrayLike,
    strike: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
    df: ArrayLike,
    eps: float,
) -> np.ndarray:
    forward_arr = _as_float_array(forward)
    strike_arr = _as_float_array(strike)
    sigma_arr = _as_float_array(sigma)
    T_arr = _as_float_array(T)
    df_arr = _as_float_array(df)
    forward_arr, strike_arr, sigma_arr, T_arr, df_arr = np.broadcast_arrays(
        forward_arr,
        strike_arr,
        sigma_arr,
        T_arr,
        df_arr,
    )

    if np.any(~np.isfinite(forward_arr)) or np.any(forward_arr <= 0.0):
        raise ValueError("forward must be finite and > 0.")
    if np.any(~np.isfinite(strike_arr)) or np.any(strike_arr <= 0.0):
        raise ValueError("strike must be finite and > 0.")
    if np.any(~np.isfinite(sigma_arr)) or np.any(sigma_arr < 0.0):
        raise ValueError("sigma must be finite and >= 0.")
    if np.any(~np.isfinite(T_arr)) or np.any(T_arr < 0.0):
        raise ValueError("T must be finite and >= 0.")
    if np.any(~np.isfinite(df_arr)) or np.any(df_arr <= 0.0):
        raise ValueError("df must be finite and > 0.")

    call_intrinsic = df_arr * np.maximum(forward_arr - strike_arr, 0.0)
    put_intrinsic = df_arr * np.maximum(strike_arr - forward_arr, 0.0)
    intrinsic = call_intrinsic if kind == OptionType.CALL else put_intrinsic

    sqrt_T = np.sqrt(T_arr)
    sigma_sqrt_T = sigma_arr * sqrt_T
    mask = (T_arr > eps) & (sigma_sqrt_T > eps)
    if not np.any(mask):
        return np.asarray(intrinsic, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_fk = np.log(forward_arr / strike_arr)
        d1 = (log_fk + 0.5 * sigma_arr * sigma_arr * T_arr) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        call_price = df_arr * (forward_arr * norm.cdf(d1) - strike_arr * norm.cdf(d2))

    if kind == OptionType.CALL:
        price = call_price
    elif kind == OptionType.PUT:
        price = call_price - df_arr * (forward_arr - strike_arr)
    else:
        raise ValueError(f"Unsupported option kind: {kind}")
    return np.asarray(np.where(mask, price, intrinsic), dtype=np.float64)


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


@overload
def essvi_implied_price(
    *,
    kind: OptionType,
    strike: ArrayLike,
    forward: ArrayLike,
    df: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
    market: None = None,
) -> np.ndarray: ...


@overload
def essvi_implied_price(
    *,
    kind: OptionType,
    strike: ArrayLike,
    params: ESSVITermStructures,
    T: ArrayLike,
    market: MarketData | PricingContext,
    forward: None = None,
    df: None = None,
) -> np.ndarray: ...


def essvi_implied_price(
    *,
    kind: OptionType,
    strike: ArrayLike,
    forward: ArrayLike | None = None,
    df: ArrayLike | None = None,
    params: ESSVITermStructures,
    T: ArrayLike,
    market: MarketData | PricingContext | None = None,
) -> np.ndarray:
    strike_arr = _as_float_array(strike)
    T_arr = _as_float_array(T)
    if np.any(~np.isfinite(strike_arr)) or np.any(strike_arr <= 0.0):
        raise ValueError("strike must be finite and > 0.")

    if market is not None:
        if forward is not None or df is not None:
            raise ValueError(
                "Pass either market or forward/df to essvi_implied_price, not both."
            )
        forward_arr, df_arr = _forward_df_from_market(market=market, T=T_arr)
    else:
        if forward is None or df is None:
            raise ValueError(
                "Either market or both forward and df must be provided to "
                "essvi_implied_price."
            )
        forward_arr = _as_float_array(forward)
        df_arr = _as_float_array(df)

    if np.any(~np.isfinite(forward_arr)) or np.any(forward_arr <= 0.0):
        raise ValueError("forward must be finite and > 0.")

    y = np.asarray(np.log(strike_arr / forward_arr), dtype=np.float64)
    w = essvi_total_variance(y=y, params=params, T=T_arr)
    _, _, w_arr, T_broadcast = np.broadcast_arrays(strike_arr, forward_arr, w, T_arr)
    sigma = np.asarray(
        np.sqrt(np.maximum(w_arr / T_broadcast, np.float64(0.0))),
        dtype=np.float64,
    )
    return _black76_price_broadcast(
        kind=kind,
        forward=forward_arr,
        strike=strike_arr,
        sigma=sigma,
        T=T_arr,
        df=df_arr,
        eps=params.eps,
    )


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


# TODO: Compute BS market implied vegas for residual weighting
