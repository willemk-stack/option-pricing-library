from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class DiscountCurve(Protocol):
    """Discount curve protocol.

    The library standardizes on **time-to-expiry** (``tau``) for internal pricing.

    Implementations must return the discount factor for a cashflow paid at
    ``tau`` years from now:

    ``df(tau) = P(0, tau)``.
    """

    def df(self, tau: float) -> float: ...

    def __call__(self, tau: float) -> float: ...


class ForwardCurve(Protocol):
    """Forward curve protocol.

    Implementations must return the *forward price* for delivery at maturity
    ``tau`` years from now:

    ``fwd(tau) = F(0, tau)``.

    Notes
    -----
    - For equity with continuous dividend yield q and flat r:
      ``F(0, tau) = S0 * exp((r-q)*tau)``.
    - Some legacy code in this repo uses a ``forward(T, t=0)`` signature.
      Implementations may provide that alias, but the preferred method is ``fwd``.
    """

    def fwd(self, tau: float) -> float: ...

    # Optional legacy alias (not required by the repo's core, but nice to have).
    def forward(self, T: float, t: float = 0.0) -> float: ...

    def __call__(self, tau: float) -> float: ...


@dataclass(frozen=True, slots=True)
class FlatDiscountCurve:
    """Flat (continuously-compounded) discount curve: ``df(tau)=exp(-r*tau)``."""

    r: float

    def df(self, tau: float) -> float:
        tau = float(tau)
        if tau < 0.0:
            raise ValueError("tau must be >= 0")
        return math.exp(-float(self.r) * tau)

    def __call__(self, tau: float) -> float:
        return self.df(tau)


@dataclass(frozen=True, slots=True)
class FlatCarryForwardCurve:
    """Forward curve with constant (continuous) carry.

    Uses the cost-of-carry relationship:

        ``F(0,tau) = S0 * exp((r - q) * tau)``

    Parameters
    ----------
    spot
        Spot price at valuation.
    r
        Continuously-compounded risk-free rate.
    q
        Continuously-compounded dividend yield / foreign rate / carry yield.
    """

    spot: float
    r: float
    q: float = 0.0

    def fwd(self, tau: float) -> float:
        tau = float(tau)
        if tau < 0.0:
            raise ValueError("tau must be >= 0")
        if float(self.spot) <= 0.0:
            raise ValueError("spot must be > 0")
        return float(self.spot) * math.exp((float(self.r) - float(self.q)) * tau)

    # Legacy alias for earlier code in this repo.
    def forward(self, T: float, t: float = 0.0) -> float:
        tau = float(T) - float(t)
        if tau < 0.0:
            raise ValueError("T must be >= t")
        return self.fwd(tau)

    def __call__(self, tau: float) -> float:
        return self.fwd(tau)


@dataclass(frozen=True, slots=True)
class PricingContext:
    """Container for the market term structures used by pricers.

    Attributes
    ----------
    spot
        Spot price at valuation.
    discount
        Discount curve providing ``df(tau)``.
    forward
        Forward curve providing ``fwd(tau)``.

    Notes
    -----
    The repo's *internal* convention is that pricers operate on **tau**
    (time-to-expiry). If you work with (t, T) in user code, convert to
    ``tau = T - t`` at the boundary.
    """

    spot: float
    discount: DiscountCurve
    forward: ForwardCurve

    def df(self, tau: float) -> float:
        return float(self.discount.df(float(tau)))

    def fwd(self, tau: float) -> float:
        return float(self.forward.fwd(float(tau)))

    def prepaid_forward(self, tau: float) -> float:
        """Prepaid forward ``Fp(tau) = df(tau) * fwd(tau)``."""

        tau = float(tau)
        return self.df(tau) * self.fwd(tau)


# -------------------------
# Small helpers
# -------------------------


def avg_rate_from_df(df: float, tau: float) -> float:
    """Compute the *average* continuously-compounded rate over [0, tau].

    For deterministic rates, ``df = exp(-∫ r dt)``. This returns

        ``r_avg = -log(df)/tau``.

    Used for models that still require a single number (e.g. CRR trees).
    """

    tau = float(tau)
    df = float(df)
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    if df <= 0.0:
        raise ValueError("df must be > 0")
    return -math.log(df) / tau


def avg_carry_from_forward(spot: float, forward: float, tau: float) -> float:
    """Compute the *average* cost-of-carry (r-q) over [0, tau].

    For deterministic carry, ``F = S0 * exp(∫ (r-q) dt)``. This returns

        ``b_avg = log(F/S0)/tau``.
    """

    tau = float(tau)
    spot = float(spot)
    forward = float(forward)
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    if spot <= 0.0:
        raise ValueError("spot must be > 0")
    if forward <= 0.0:
        raise ValueError("forward must be > 0")
    return math.log(forward / spot) / tau
