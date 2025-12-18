from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from ..models.stochastic_processes import sim_gbm_terminal
from ..types import OptionType, PricingInputs

# ----------------------------
# Payoff helpers
# ----------------------------


def _call_payoff(ST: np.ndarray, *, K: float) -> np.ndarray:
    return np.maximum(ST - K, 0.0)


def _put_payoff(ST: np.ndarray, *, K: float) -> np.ndarray:
    return np.maximum(K - ST, 0.0)


# ----------------------------
# MC model (thin container + algorithm)
# ----------------------------


@dataclass(frozen=True, slots=True)
class McGBMModel:
    S0: float
    r: float
    q: float
    sigma: float
    tau: float
    n_paths: int
    rng: np.random.Generator = field(
        default_factory=np.random.default_rng,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.S0 <= 0.0:
            raise ValueError("S0 must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")

    def simulate_terminal(self) -> np.ndarray:
        # Risk-neutral drift with continuous dividend yield q: mu = r - q
        return sim_gbm_terminal(
            n_paths=self.n_paths,
            T=self.tau,
            mu=self.r - self.q,
            sigma=self.sigma,
            S0=self.S0,
            rng=self.rng,
        )

    def price_european(
        self, payoff: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[float, float]:
        ST = self.simulate_terminal()
        payoff_vals = payoff(ST)

        disc = float(np.exp(-self.r * self.tau))
        mean = float(payoff_vals.mean())

        if self.n_paths > 1:
            std = float(payoff_vals.std(ddof=1))
        else:
            std = 0.0

        price = disc * mean
        std_err = disc * std / float(np.sqrt(self.n_paths))
        return price, std_err


# ----------------------------
# RNG helper
# ----------------------------


def _make_rng(seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if rng is not None:
        return rng
    if seed is not None:
        return np.random.default_rng(int(seed))
    return np.random.default_rng()


# ----------------------------
# Public API (PricingInputs)
# ----------------------------


def mc_price(
    p: PricingInputs,
    *,
    n_paths: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Generic MC pricer that dispatches on p.spec.kind (CALL/PUT).
    Returns (price, standard_error).
    """
    model = McGBMModel(
        S0=p.S,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
        n_paths=int(n_paths),
        rng=_make_rng(seed, rng),
    )

    if p.spec.kind == OptionType.CALL:
        payoff = partial(_call_payoff, K=p.K)
    elif p.spec.kind == OptionType.PUT:
        payoff = partial(_put_payoff, K=p.K)
    else:
        raise ValueError(f"Unsupported option kind: {p.spec.kind}")

    return model.price_european(payoff)


def mc_price_call(
    p: PricingInputs,
    *,
    n_paths: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    model = McGBMModel(
        S0=p.S,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
        n_paths=int(n_paths),
        rng=_make_rng(seed, rng),
    )
    payoff = partial(_call_payoff, K=p.K)
    return model.price_european(payoff)


def mc_price_put(
    p: PricingInputs,
    *,
    n_paths: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    model = McGBMModel(
        S0=p.S,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
        n_paths=int(n_paths),
        rng=_make_rng(seed, rng),
    )
    payoff = partial(_put_payoff, K=p.K)
    return model.price_european(payoff)


# ----------------------------
# Backwards-compatible aliases
# ----------------------------


def mc_call_from_inputs(
    p: PricingInputs,
    n_paths: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    return mc_price_call(p, n_paths=int(n_paths), seed=seed, rng=rng)


def mc_put_from_inputs(
    p: PricingInputs,
    n_paths: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    return mc_price_put(p, n_paths=int(n_paths), seed=seed, rng=rng)
