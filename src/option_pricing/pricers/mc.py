from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from numpy.typing import NDArray

from ..models.stochastic_processes import sim_gbm_terminal
from ..types import PricingInputs
from ..vanilla import call_payoff, make_vanilla_payoff, put_payoff


def _apply_control_variate(X: np.ndarray, Y: np.ndarray, EY: float) -> np.ndarray:
    # Guard against degenerate controls
    var_y = float(np.var(Y, ddof=1)) if Y.size > 1 else 0.0
    if var_y <= 0.0:
        return X

    cov = float(np.cov(X, Y, ddof=1)[0, 1])
    b = cov / var_y
    return X - b * (Y - float(EY))


@dataclass(frozen=True, slots=True)
class ControlVariate:
    values: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    mean: float


@dataclass(frozen=True, slots=True)
class McGBMModel:
    S0: float
    r: float
    q: float
    sigma: float
    tau: float
    n_paths: int
    antithetic: bool = False
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
        if self.antithetic and (self.n_paths % 2 != 0):
            raise ValueError(
                "antithetic=True requires an even n_paths (paired samples)."
            )

    def simulate_terminal(self) -> np.ndarray:
        mu = self.r - self.q  # risk-neutral drift

        if not self.antithetic:
            return sim_gbm_terminal(
                n_paths=self.n_paths,
                T=self.tau,
                mu=mu,
                sigma=self.sigma,
                S0=self.S0,
                rng=self.rng,
            )

        # Antithetic: use Z and -Z pairs
        n_pairs = self.n_paths // 2
        Z = self.rng.standard_normal(n_pairs)

        drift = (mu - 0.5 * self.sigma**2) * self.tau
        vol = self.sigma * np.sqrt(self.tau)

        ST_pos = self.S0 * np.exp(drift + vol * Z)
        ST_neg = self.S0 * np.exp(drift - vol * Z)

        out = np.concatenate([ST_pos, ST_neg])
        return np.asarray(out, dtype=np.float64)

    def price_european(
        self,
        payoff: Callable[[np.ndarray], np.ndarray],
        *,
        control: ControlVariate | None = None,
    ) -> tuple[float, float]:
        ST = self.simulate_terminal()
        payoff_vals = payoff(ST)

        disc = float(np.exp(-self.r * self.tau))

        # ---------- plain MC ----------
        if not self.antithetic:
            X_eff = payoff_vals
            if control is not None:
                Y_vals = control.values(ST)
                X_eff = _apply_control_variate(X_eff, Y_vals, control.mean)

            mean = float(X_eff.mean())
            std = float(X_eff.std(ddof=1)) if self.n_paths > 1 else 0.0

            price = disc * mean
            std_err = disc * std / float(np.sqrt(self.n_paths))
            return price, std_err

        # ---------- antithetic MC ----------
        n_pairs = self.n_paths // 2
        Xp = 0.5 * (payoff_vals[:n_pairs] + payoff_vals[n_pairs:])

        if control is not None:
            Y_vals = control.values(ST)
            Yp = 0.5 * (Y_vals[:n_pairs] + Y_vals[n_pairs:])
            Xp = _apply_control_variate(Xp, Yp, control.mean)

        mean = float(Xp.mean())
        std = float(Xp.std(ddof=1)) if n_pairs > 1 else 0.0

        price = disc * mean
        std_err = disc * std / float(np.sqrt(n_pairs))
        return price, std_err


def _make_rng(seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if rng is not None:
        return rng
    if seed is not None:
        return np.random.default_rng(int(seed))
    return np.random.default_rng()


def mc_price(
    p: PricingInputs,
    *,
    n_paths: int,
    antithetic: bool = False,
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
        antithetic=bool(antithetic),
        rng=_make_rng(seed, rng),
    )

    payoff = make_vanilla_payoff(p.spec.kind, K=p.K)
    return model.price_european(payoff)


def mc_price_call(
    p: PricingInputs,
    *,
    n_paths: int,
    antithetic: bool = False,
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
        antithetic=bool(antithetic),
        rng=_make_rng(seed, rng),
    )
    payoff = partial(call_payoff, K=p.K)
    return model.price_european(payoff)


def mc_price_put(
    p: PricingInputs,
    *,
    n_paths: int,
    antithetic: bool = False,
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
        antithetic=bool(antithetic),
        rng=_make_rng(seed, rng),
    )
    payoff = partial(put_payoff, K=p.K)
    return model.price_european(payoff)
