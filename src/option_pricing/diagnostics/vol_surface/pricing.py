from __future__ import annotations

import numpy as np

from .contracts import Black76Module
from .sampling import _smile_grid, get_smile_at_T


def _default_black76() -> Black76Module:
    from option_pricing.models.black_scholes import bs as _bs

    return _bs


def surface_iv_from_strikes(
    surface,
    *,
    strikes,
    T,
    forward,
):
    from option_pricing.vol.surface_core import VolSurface

    K = np.asarray(strikes, dtype=float)
    tau = float(T)
    if isinstance(surface, VolSurface):
        return np.asarray(surface.iv(K, tau), dtype=float)

    F = float(forward(tau))
    if F <= 0.0:
        raise ValueError(f"forward(T) must be > 0, got {F} at T={tau}")
    y = np.log(K / F)
    return np.asarray(surface.iv(y, tau), dtype=float)


def build_surface_from_iv_function(
    *,
    expiries,
    x_grid,
    iv_fn,
    forward,
    VolSurface_cls=None,
):
    if VolSurface_cls is None:
        from option_pricing.vol.surface_core import VolSurface

        VolSurface_cls = VolSurface
    exp = np.asarray(list(expiries), dtype=float)
    if exp.ndim != 1 or exp.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or x_grid.size == 0:
        raise ValueError("x_grid must be a non-empty 1D array")
    rows = []
    for T in exp:
        for y in x_grid:
            iv = float(iv_fn(T, y))
            rows.append((T, y, iv))
    return VolSurface_cls.from_grid(rows, forward=forward)


def call_prices_from_smile(surface, *, T, forward, df, bs_model=None):
    if bs_model is None:
        bs_model = _default_black76()
    s = get_smile_at_T(surface, T)
    T = float(s.T)
    F = float(forward(T))
    dfT = float(df(T))
    y, w = _smile_grid(s)
    K = F * np.exp(y)
    iv = np.sqrt(np.maximum(w / T, 0.0))
    C = bs_model.black76_call_price_vec(forward=F, strikes=K, sigma=iv, tau=T, df=dfT)
    return K, iv, C


def call_prices_from_surface_on_strikes(
    surface,
    *,
    expiries,
    strikes,
    forward,
    df,
    bs_model=None,
):
    if bs_model is None:
        bs_model = _default_black76()
    Ts = np.asarray(list(expiries), dtype=float)
    K = np.asarray(strikes, dtype=float)
    if Ts.ndim != 1 or Ts.size == 0:
        raise ValueError("expiries must be a non-empty 1D sequence")
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty 1D array")
    if not np.all(np.diff(Ts) > 0):
        raise ValueError("expiries must be strictly increasing")
    if not np.all(np.diff(K) > 0):
        raise ValueError("strikes must be strictly increasing")
    if np.any(K <= 0):
        raise ValueError("strikes must be > 0")
    nT = int(Ts.size)
    nK = int(K.size)
    calls = np.empty((nT, nK), dtype=float)
    iv = np.empty((nT, nK), dtype=float)
    forwards = np.empty((nT,), dtype=float)
    dfs = np.empty((nT,), dtype=float)
    for i, T in enumerate(Ts):
        F = float(forward(T))
        dfT = float(df(T))
        iv[i, :] = surface_iv_from_strikes(
            surface,
            strikes=K,
            T=float(T),
            forward=forward,
        )
        calls[i, :] = bs_model.black76_call_price_vec(
            forward=F, strikes=K, sigma=iv[i, :], tau=T, df=dfT
        )
        forwards[i] = F
        dfs[i] = dfT
    return K, Ts, calls, iv, forwards
