from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..numerics.finite_diff import finite_diff_greeks
from ..pricers.black_scholes import bs_greeks, bs_price
from ..types import PricingInputs

Method = Literal["analytic", "fd"]


@dataclass(frozen=True)
class SweepResult:
    x: np.ndarray
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "x": self.x,
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
        }


def sweep_x(
    base: PricingInputs,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    n: int = 100,
    method: Method = "analytic",
    grid: np.ndarray | None = None,
) -> SweepResult:
    """
    Sweep underlying spot and compute price + Greeks.

    - method="analytic": bs_greeks
    - method="fd": finite_diff_greeks from bs_price (scalar)
    """
    if grid is None:
        if n < 2:
            raise ValueError("n must be >= 2")

        K = float(base.spec.strike)
        if x_min is None:
            x_min = 0.5 * K
        if x_max is None:
            x_max = 1.5 * K
        if x_min >= x_max:
            raise ValueError("x_min must be < x_max")

        x = np.linspace(x_min, x_max, n, dtype=float)
    else:
        x = np.asarray(grid, dtype=float)

    price = np.empty_like(x)
    delta = np.empty_like(x)
    gamma = np.empty_like(x)
    vega = np.empty_like(x)
    theta = np.empty_like(x)

    for i, s in enumerate(x):
        p = replace(base, market=replace(base.market, spot=float(s)))
        g = (
            bs_greeks(p)
            if method == "analytic"
            else finite_diff_greeks(p, price_fn=bs_price)
        )

        price[i] = float(g["price"])
        delta[i] = float(g["delta"])
        gamma[i] = float(g["gamma"])
        vega[i] = float(g["vega"])
        theta[i] = float(g["theta"])

    return SweepResult(
        x=x, price=price, delta=delta, gamma=gamma, vega=vega, theta=theta
    )
