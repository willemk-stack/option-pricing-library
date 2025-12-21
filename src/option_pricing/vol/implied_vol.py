from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from option_pricing import PricingInputs, bs_greeks


def IV_solver(
    p: PricingInputs,
    mkt_price: float,
    root_method: Callable[..., float],  # bisection, bracketed_newton, etc.
    sigma0: float | None = None,
    sigma_lo: float | None = 1e-8,
    sigma_hi: float | None = 5.0,
) -> float:

    def Fn(sigma: float) -> float:
        px = replace(p, sigma=float(sigma))
        return bs_greeks(px)["price"] - mkt_price

    def dFn(sigma: float) -> float:
        px = replace(p, sigma=float(sigma))
        return bs_greeks(px)["vega"]

    # If the root_method ignores x0/dFn (like bisection), it should accept **_.
    return root_method(Fn, sigma_lo, sigma_hi, x0=(sigma0 or p.sigma), dFn=dFn)
