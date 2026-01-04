from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

Method = Literal["analytic", "fd"]


@dataclass(frozen=True)
class SweepResult:
    x: np.ndarray
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray


def _default_bs_greeks():
    try:
        from option_pricing.pricers.black_scholes import bs_greeks  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import bs_greeks at option_pricing.pricers.black_scholes.bs_greeks. "
            "Pass greeks_fn=... explicitly."
        ) from e
    return bs_greeks


def _default_bs_price():
    try:
        from option_pricing.pricers.black_scholes import bs_price  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import bs_price at option_pricing.pricers.black_scholes.bs_price. "
            "Pass price_fn=... explicitly."
        ) from e
    return bs_price


def _default_fd_greeks():
    try:
        from option_pricing.numerics.finite_diff import (
            finite_diff_greeks,  # type: ignore
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import finite_diff_greeks at option_pricing.numerics.finite_diff.finite_diff_greeks. "
            "Pass fd_greeks_fn=... explicitly."
        ) from e
    return finite_diff_greeks


def sweep_spot_greeks(
    base: Any,
    spots: Sequence[float],
    *,
    method: Method = "analytic",
    greeks_fn: Callable[[Any], dict[str, float]] | None = None,
    price_fn: Callable[[Any], float] | None = None,
    fd_greeks_fn: Callable[..., dict[str, float]] | None = None,
) -> SweepResult:
    """Sweep spot and compute greeks.

    Assumes `base` is a dataclass-like object with nested `market` that has `spot`,
    or else has attribute `S`/`spot` directly. We update spot using dataclasses.replace
    when available.

    The returned greek dict is expected to contain: price, delta, gamma, vega, theta.
    """
    spots_arr = np.asarray(list(spots), dtype=float)

    greeks_fn = greeks_fn or _default_bs_greeks()
    price_fn = price_fn or _default_bs_price()
    fd_greeks_fn = fd_greeks_fn or _default_fd_greeks()

    price = np.empty_like(spots_arr)
    delta = np.empty_like(spots_arr)
    gamma = np.empty_like(spots_arr)
    vega = np.empty_like(spots_arr)
    theta = np.empty_like(spots_arr)

    for i, s in enumerate(spots_arr):
        p = base
        # best-effort spot update
        try:
            mkt = base.market
            p = replace(base, market=replace(mkt, spot=float(s)))
        except Exception:
            try:
                p = replace(base, spot=float(s))  # type: ignore
            except Exception:
                p.spot = float(s)

        g = greeks_fn(p) if method == "analytic" else fd_greeks_fn(p, price_fn=price_fn)

        price[i] = float(g["price"])
        delta[i] = float(g["delta"])
        gamma[i] = float(g["gamma"])
        vega[i] = float(g["vega"])
        theta[i] = float(g["theta"])

    return SweepResult(
        x=spots_arr, price=price, delta=delta, gamma=gamma, vega=vega, theta=theta
    )
