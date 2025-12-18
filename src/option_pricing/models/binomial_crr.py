from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt


@dataclass(frozen=True, slots=True)
class BinomialModel:
    S0: float  # initial stock price
    u: float  # up factor
    d: float  # down factor
    r: float  # risk-free rate (cc, per unit time)
    q: float  # dividend yield (cc)
    dt: float  # time step
    n_steps: int

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if not (0.0 < self.d < self.u):
            raise ValueError("Need 0 < d < u")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

        # ensure risk-neutral prob is meaningful
        p = self.p_star
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"Risk-neutral probability out of bounds: p*={p:.6g}. "
                "Try increasing n_steps or check r/q/sigma."
            )

    @classmethod
    def from_crr(
        cls, *, S0: float, r: float, q: float, sigma: float, T: float, n_steps: int
    ) -> BinomialModel:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if T <= 0.0:
            raise ValueError("T must be positive")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")

        dt = T / n_steps
        u = exp(sigma * sqrt(dt))
        d = exp(-sigma * sqrt(dt))
        return cls(S0=S0, u=u, d=d, r=r, q=q, dt=dt, n_steps=n_steps)

    @property
    def T(self) -> float:
        return self.dt * self.n_steps

    @property
    def p_star(self) -> float:
        # Under continuous dividend yield q: E[S_{t+dt}/S_t] = exp((r-q)dt)
        growth = exp((self.r - self.q) * self.dt)
        return (growth - self.d) / (self.u - self.d)

    @property
    def disc_step(self) -> float:
        return exp(-self.r * self.dt)
