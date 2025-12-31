from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def _default_bs_price():
    try:
        from option_pricing.pricers.black_scholes import bs_price  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import bs_price at option_pricing.pricers.black_scholes.bs_price. "
            "Pass bs_price_fn=... explicitly."
        ) from e
    return bs_price


def _default_binom_price():
    from option_pricing import binom_price

    return binom_price


def binom_convergence_series(
    p: Any,
    n_steps: int | Sequence[int],
    *,
    method: str = "crr",
    bs_price_fn: Callable[[Any], float] | None = None,
    binom_price_fn: Callable[..., float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute binomial price convergence to a BS benchmark across n_steps."""
    bs_price_fn = bs_price_fn or _default_bs_price()
    binom_price_fn = binom_price_fn or _default_binom_price()

    # Allow convenience: pass an int to mean "max steps".
    # For modest max_n (<=500) we compute all steps 1..max_n (handy for tolerance search).
    # For larger, we sample a dense-but-manageable grid.
    if isinstance(n_steps, (int, np.integer)):
        max_n = int(n_steps)
        if max_n <= 0:
            raise ValueError("n_steps must be a positive integer")
        if max_n <= 500:
            n_steps_vals = np.arange(1, max_n + 1, dtype=int)
        else:
            # ~80 points, denser at small N
            n_steps_vals = np.unique(
                np.round(np.geomspace(1, max_n, num=80)).astype(int)
            )
            if n_steps_vals[-1] != max_n:
                n_steps_vals = np.append(n_steps_vals, max_n)
    else:
        n_steps_vals = np.asarray(list(n_steps), dtype=int)
        if n_steps_vals.size == 0:
            raise ValueError("n_steps must be non-empty")
        if np.any(n_steps_vals <= 0):
            raise ValueError("n_steps must be positive integers")
        # Sort + unique for stable plotting
        n_steps_vals = np.unique(n_steps_vals)

    binom_vals = np.array(
        [float(binom_price_fn(p, int(n), method=method)) for n in n_steps_vals],
        dtype=float,
    )
    bs_val = float(bs_price_fn(p))
    abs_err = np.abs(bs_val - binom_vals)

    return {
        "n_steps": n_steps_vals,
        "binom": binom_vals,
        "bs": np.asarray([bs_val], dtype=float),
        "abs_error": abs_err,
    }
