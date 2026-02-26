"""Vectorized implied volatility computation across multiple strikes (slices)."""

from __future__ import annotations

import numpy as np

from ..models.black_scholes.bs import black76_call_price_vega_vec
from ..typing import ArrayLike, FloatArray
from .implied_vol_types import ImpliedVolSliceResult


def implied_vol_black76_slice(
    forward: float,
    strikes: ArrayLike,
    tau: float,
    df: float,
    prices: ArrayLike,
    is_call: bool | ArrayLike = True,
    initial_sigma: float | ArrayLike = 0.2,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
    max_iter: int = 100,
    tol: float = 1e-10,
    return_result: bool = False,
) -> FloatArray | tuple[FloatArray, ImpliedVolSliceResult]:
    F = float(forward)
    if F <= 0.0:
        raise ValueError("forward must be positive")
    tau = float(tau)
    if tau < 0.0:
        raise ValueError("tau must be >= 0")
    df = float(df)
    if df <= 0.0:
        raise ValueError("df must be positive")

    K = np.asarray(strikes, dtype=float)
    P = np.asarray(prices, dtype=float)
    if K.shape != P.shape:
        raise ValueError("strikes and prices must have the same shape")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be positive")

    # call/put flags (broadcastable)
    if is_call is None:
        is_call_arr = np.ones_like(K, dtype=bool)
    else:
        is_call_arr = np.asarray(is_call, dtype=bool)
        if is_call_arr.shape != K.shape:
            is_call_arr = np.broadcast_to(is_call_arr, K.shape).astype(bool)

    # Convert puts -> call targets via parity
    call_target = np.where(is_call_arr, P, P + df * (F - K))

    intrinsic = df * np.maximum(F - K, 0.0)
    upper = df * F

    status = np.zeros_like(K, dtype=np.int8)
    vol = np.full_like(K, np.nan, dtype=float)
    converged = np.zeros_like(K, dtype=bool)
    iterations = np.zeros_like(K, dtype=np.int32)
    f_at_root = np.full_like(K, np.nan, dtype=float)

    invalid = (
        ~np.isfinite(call_target)
        | (call_target < intrinsic - 1e-12)
        | (call_target > upper + 1e-12)
    )
    status[invalid] = 3  # INVALID_PRICE

    # tau==0: only intrinsic is admissible
    if tau == 0.0:
        ok0 = ~invalid & (np.abs(call_target - intrinsic) <= tol)
        vol[ok0] = sigma_lo
        converged[ok0] = True
        f_at_root[ok0] = call_target[ok0] - intrinsic[ok0]
        status[ok0] = 0
        if return_result:
            res = ImpliedVolSliceResult(
                vol=vol,
                converged=converged,
                iterations=iterations,
                f_at_root=f_at_root,
                bracket_lo=np.full_like(K, sigma_lo),
                bracket_hi=np.full_like(K, sigma_hi),
                status=status,
            )
            return vol, res
        return vol

    # Per-strike brackets
    a = np.full_like(K, sigma_lo, dtype=float)
    b = np.full_like(K, sigma_hi, dtype=float)

    # f(a), f(b)
    pa, _ = black76_call_price_vega_vec(forward=F, strikes=K, sigma=a, tau=tau, df=df)
    pb, _ = black76_call_price_vega_vec(forward=F, strikes=K, sigma=b, tau=tau, df=df)
    fa = pa - call_target
    fb = pb - call_target

    ok = ~invalid
    low_clip = ok & (fa > tol)  # root < sigma_lo
    high_clip = ok & (fb < -tol)  # root > sigma_hi

    if np.any(low_clip):
        vol[low_clip] = sigma_lo
        converged[low_clip] = True
        status[low_clip] = 1
        f_at_root[low_clip] = fa[low_clip]
        iterations[low_clip] = 0

    if np.any(high_clip):
        vol[high_clip] = sigma_hi
        converged[high_clip] = True
        status[high_clip] = 2
        f_at_root[high_clip] = fb[high_clip]
        iterations[high_clip] = 0

    active = ok & ~low_clip & ~high_clip & (fa <= tol) & (fb >= -tol)

    # Initial guess
    x0 = np.asarray(initial_sigma, dtype=float)
    if x0.shape != K.shape:
        x0 = np.broadcast_to(x0, K.shape).astype(float)
    x = np.clip(x0, a, b)

    tol_x = 1e-12
    vega_min = 1e-14

    for it in range(1, max_iter + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break

        # One kernel call per iteration for all active points
        price, vega = black76_call_price_vega_vec(
            forward=F, strikes=K[idx], sigma=x[idx], tau=tau, df=df
        )
        fx = price - call_target[idx]

        # 1) residual convergence (at current x)
        done = np.abs(fx) <= tol
        if np.any(done):
            j = idx[done]
            vol[j] = x[j]
            converged[j] = True
            status[j] = 0
            f_at_root[j] = fx[done]
            iterations[j] = it - 1
            active[j] = False

        idx = idx[~done]
        if idx.size == 0:
            continue

        fx = fx[~done]
        vega = vega[~done]

        # 2) bracket update using sign change between a and x
        fa_idx = fa[idx]
        left = (fa_idx * fx) < 0.0  # root in [a, x] => b := x
        if np.any(left):
            jl = idx[left]
            b[jl] = x[jl]
            fb[jl] = fx[left]
        if np.any(~left):
            jr = idx[~left]
            a[jr] = x[jr]
            fa[jr] = fx[~left]

        # 3) bracket-width convergence
        width_done = ((b[idx] - a[idx]) * 0.5) <= tol_x
        if np.any(width_done):
            j = idx[width_done]
            xm = 0.5 * (a[j] + b[j])
            pm, _ = black76_call_price_vega_vec(
                forward=F, strikes=K[j], sigma=xm, tau=tau, df=df
            )
            fm = pm - call_target[j]
            vol[j] = xm
            converged[j] = True
            status[j] = 0
            f_at_root[j] = fm
            iterations[j] = it
            active[j] = False

        idx = idx[~width_done]
        if idx.size == 0:
            continue

        fx = fx[~width_done]
        vega = vega[~width_done]

        # 4) compute next sigma (Newton if safe & inside bracket, else bisection)
        newton_ok = np.isfinite(vega) & (vega > vega_min) & np.isfinite(fx)
        x_curr = x[idx]
        cand = x_curr - fx / vega
        mid = 0.5 * (a[idx] + b[idx])

        inside = (cand > a[idx]) & (cand < b[idx])
        x_new = np.where(newton_ok & inside, cand, mid)

        # 5) step convergence (sigma stops moving)
        step_small = np.abs(x_new - x_curr) <= tol_x * np.maximum(1.0, np.abs(x_new))
        if np.any(step_small):
            j = idx[step_small]
            vol[j] = x_new[step_small]
            converged[j] = True
            status[j] = 0
            pn, _ = black76_call_price_vega_vec(
                forward=F, strikes=K[j], sigma=vol[j], tau=tau, df=df
            )
            f_at_root[j] = pn - call_target[j]
            iterations[j] = it
            active[j] = False

        # update x only for still-active points (aligned indexing)
        idx = idx[~step_small]
        if idx.size:
            x[idx] = np.clip(x_new[~step_small], a[idx], b[idx])

    # anything still active: no convergence
    still = active
    if np.any(still):
        status[still] = 4
        vol[still] = 0.5 * (a[still] + b[still])
        pr, _ = black76_call_price_vega_vec(
            forward=F, strikes=K[still], sigma=vol[still], tau=tau, df=df
        )
        f_at_root[still] = pr - call_target[still]
        iterations[still] = max_iter

    if return_result:
        res = ImpliedVolSliceResult(
            vol=vol,
            converged=converged,
            iterations=iterations,
            f_at_root=f_at_root,
            bracket_lo=a,
            bracket_hi=b,
            status=status,
        )
        return vol, res

    return vol
