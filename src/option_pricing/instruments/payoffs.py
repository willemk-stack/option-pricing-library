from __future__ import annotations

from collections.abc import Callable

import numpy as np

from option_pricing.types import OptionType


def call_payoff(ST: np.ndarray, *, K: float) -> np.ndarray:
    return np.maximum(ST - K, 0.0)


def put_payoff(ST: np.ndarray, *, K: float) -> np.ndarray:
    return np.maximum(K - ST, 0.0)


def make_vanilla_payoff(
    kind: OptionType, *, K: float
) -> Callable[[np.ndarray], np.ndarray]:
    if kind == OptionType.CALL:

        def payoff(ST: np.ndarray) -> np.ndarray:
            return call_payoff(ST, K=K)

        return payoff

    if kind == OptionType.PUT:

        def payoff(ST: np.ndarray) -> np.ndarray:
            return put_payoff(ST, K=K)

        return payoff

    raise ValueError(f"Unsupported option kind: {kind}")
