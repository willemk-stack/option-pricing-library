from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from ..numerics.finite_diff import finite_diff_greeks
from ..pricers.black_scholes import bs_call_greeks, bs_price_call
from ..types import MarketData, OptionSpec, OptionType, PricingInputs


def sweep_x(
    t: float = 0.0,
    K: float = 100.0,
    r: float = 0.01,
    sigma: float = 0.2,
    T: float = 1.0,
    x_min: float | None = None,
    x_max: float | None = None,
    n: int = 100,
    method: Literal["analytic", "fd"] = "analytic",
    q: float = 0.0,
) -> None:
    """
    Sweep underlying price (spot) and plot price + Greeks for a European CALL.

    method:
      - "analytic": uses bs_call_greeks(PricingInputs)
      - "fd":       uses finite_diff_greeks(PricingInputs, price_fn=bs_price_call)

    q: continuous dividend yield (default 0.0)
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "sweep_x requires matplotlib. Install it with: pip install matplotlib"
        ) from e

    if x_min is None:
        x_min = 0.5 * K
    if x_max is None:
        x_max = 1.5 * K

    x_grid = np.linspace(x_min, x_max, n)

    base = PricingInputs(
        spec=OptionSpec(kind=OptionType.CALL, strike=K, expiry=T),
        market=MarketData(spot=float(x_grid[0]), rate=r, dividend_yield=q),
        sigma=sigma,
        t=t,
    )

    prices_list: list[float] = []
    deltas_list: list[float] = []
    gammas_list: list[float] = []
    vegas_list: list[float] = []
    thetas_list: list[float] = []

    for x in x_grid:
        p = replace(base, market=replace(base.market, spot=float(x)))

        if method == "analytic":
            g = bs_call_greeks(p)
        else:  # "fd"
            g = finite_diff_greeks(p, price_fn=bs_price_call)

        prices_list.append(float(g["price"]))
        deltas_list.append(float(g["delta"]))
        gammas_list.append(float(g["gamma"]))
        vegas_list.append(float(g["vega"]))
        thetas_list.append(float(g["theta"]))

    prices = np.asarray(prices_list, dtype=float)
    deltas = np.asarray(deltas_list, dtype=float)
    gammas = np.asarray(gammas_list, dtype=float)
    vegas = np.asarray(vegas_list, dtype=float)
    thetas = np.asarray(thetas_list, dtype=float)

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 12))

    axs[0].plot(x_grid, prices)
    axs[0].set_ylabel("Price")

    axs[1].plot(x_grid, deltas)
    axs[1].set_ylabel("Delta")

    axs[2].plot(x_grid, gammas)
    axs[2].set_ylabel("Gamma")

    axs[3].plot(x_grid, vegas)
    axs[3].set_ylabel("Vega")

    axs[4].plot(x_grid, thetas)
    axs[4].set_ylabel("Theta")
    axs[4].set_xlabel("Spot (S)")

    plt.tight_layout()
    plt.show()
