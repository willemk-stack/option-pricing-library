from __future__ import annotations

import warnings

import numpy as np


def build_skewed_svi_localvol_surface(
    *,
    S0: float,
    r: float,
    q: float,
    sigma_atm: float,
    expiries: np.ndarray,
):
    """Build a LocalVolSurface from a moderately skewed SVI implied surface."""
    from option_pricing.types import MarketData
    from option_pricing.vol import LocalVolSurface, VolSurface
    from option_pricing.vol.svi import SVIParams, SVISmile

    ctx = MarketData(spot=S0, rate=r, dividend_yield=q).to_context()

    smiles: list[SVISmile] = []
    for Ti in expiries:
        Ti = float(Ti)
        b = 0.10 * Ti
        svi_sig = 0.30
        rho = -0.40
        m = 0.0

        w_atm = (sigma_atm * sigma_atm) * Ti
        a = max(1e-10, float(w_atm - b * svi_sig))

        smiles.append(
            SVISmile(T=Ti, params=SVIParams(a=a, b=b, rho=rho, m=m, sigma=svi_sig))
        )

    implied = VolSurface(
        expiries=np.asarray(expiries, dtype=float),
        smiles=tuple(smiles),
        forward=ctx.fwd,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lv = LocalVolSurface.from_implied(implied, forward=ctx.fwd, discount=ctx.df)

    return ctx, lv
