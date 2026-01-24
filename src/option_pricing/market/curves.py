from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ============================================================
# Curve Protocols
# ============================================================


class DiscountCurve(Protocol):
    """Discount curve protocol (curves-first).

    Internal convention is **time-to-expiry** (tau).

    Implementations must return the discount factor for a cashflow paid at tau:

        df(tau) = P(0, tau).
    """

    def df(self, tau: float) -> float: ...


class ForwardCurve(Protocol):
    """Forward curve protocol (curves-first).

    Implementations must return the *forward price* for delivery at maturity tau:

        fwd(tau) = F(0, tau).

    Notes
    -----
    - For equity with continuous dividend yield q and flat r:
        F(0, tau) = S0 * exp((r-q) * tau)
    - Some legacy code uses a forward(T, t=0) signature. That alias is modeled
      separately (see LegacyForwardCurve) to avoid forcing every implementation
      to carry legacy API.
    """

    def fwd(self, tau: float) -> float: ...


@runtime_checkable
class CallableDiscountCurve(DiscountCurve, Protocol):
    """Optional convenience: allow calling curve(tau) as curve.df(tau)."""

    def __call__(self, tau: float) -> float: ...


@runtime_checkable
class CallableForwardCurve(ForwardCurve, Protocol):
    """Optional convenience: allow calling curve(tau) as curve.fwd(tau)."""

    def __call__(self, tau: float) -> float: ...


class LegacyForwardCurve(ForwardCurve, Protocol):
    """Optional legacy alias: forward(T, t=0)."""

    def forward(self, T: float, t: float = 0.0) -> float: ...


# ============================================================
# Concrete Curves
# ============================================================


@dataclass(frozen=True, slots=True)
class FlatDiscountCurve:
    """Flat (continuously-compounded) discount curve: df(tau)=exp(-r*tau)."""

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

        F(0,tau) = S0 * exp((r - q) * tau)

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
        spot = float(self.spot)
        if spot <= 0.0:
            raise ValueError("spot must be > 0")
        return spot * math.exp((float(self.r) - float(self.q)) * tau)

    # Legacy alias for earlier code in this repo.
    def forward(self, T: float, t: float = 0.0) -> float:
        tau = float(T) - float(t)
        if tau < 0.0:
            raise ValueError("T must be >= t")
        return self.fwd(tau)

    def __call__(self, tau: float) -> float:
        return self.fwd(tau)


# ============================================================
# Pricing Context (curves-first market container)
# ============================================================


@dataclass(frozen=True, slots=True)
class PricingContext:
    """Container for market term structures used by pricers.

    Attributes
    ----------
    spot
        Spot price at valuation.
    discount
        Discount curve providing df(tau).
    forward
        Forward curve providing fwd(tau).

    Notes
    -----
    - Internal convention: pricers operate on tau (time-to-expiry).
    - This context is curves-first. Engines that require scalars (e.g. CRR trees,
      constant-coefficient BS PDE) can use r_avg/q_avg/b_avg methods.
    """

    spot: float
    discount: DiscountCurve
    forward: ForwardCurve

    def df(self, tau: float) -> float:
        tau = float(tau)
        return float(self.discount.df(tau))

    def fwd(self, tau: float) -> float:
        tau = float(tau)
        return float(self.forward.fwd(tau))

    def prepaid_forward(self, tau: float) -> float:
        """Prepaid forward: Fp(tau) = df(tau) * fwd(tau)."""
        tau = float(tau)
        return self.df(tau) * self.fwd(tau)

    # -------------------------
    # Effective (average) scalars
    # -------------------------

    def r_avg(self, tau: float) -> float:
        """Average continuously-compounded risk-free rate over [0, tau]."""
        tau = float(tau)
        if tau < 0.0:
            raise ValueError("tau must be >= 0")
        if tau == 0.0:
            return 0.0
        df = self.df(tau)
        if df <= 0.0:
            raise ValueError("discount.df(tau) must be > 0")
        return -math.log(df) / tau

    def b_avg(self, tau: float) -> float:
        """Average carry (r-q) over [0, tau], inferred from forward/spot."""
        tau = float(tau)
        if tau < 0.0:
            raise ValueError("tau must be >= 0")
        if tau == 0.0:
            return 0.0
        spot = float(self.spot)
        if spot <= 0.0:
            raise ValueError("spot must be > 0")
        fwd = self.fwd(tau)
        if fwd <= 0.0:
            raise ValueError("forward.fwd(tau) must be > 0")
        return math.log(fwd / spot) / tau

    def q_avg(self, tau: float) -> float:
        """Average dividend yield / foreign rate / carry yield over [0, tau]."""
        return self.r_avg(tau) - self.b_avg(tau)

    def df_q(self, tau: float) -> float:
        """Implied 'dividend discount factor' over [0, tau].

        Under deterministic carry, prepaid forward satisfies:
            Fp(tau) = S0 * exp(-∫ q dt)
        so:
            df_q(tau) = exp(-q_avg(tau)*tau) = Fp(tau)/S0
        """
        tau = float(tau)
        if tau < 0.0:
            raise ValueError("tau must be >= 0")
        if tau == 0.0:
            return 1.0
        spot = float(self.spot)
        if spot <= 0.0:
            raise ValueError("spot must be > 0")
        return self.prepaid_forward(tau) / spot


# ============================================================
# Small helpers (kept for reuse by other modules)
# ============================================================


def avg_rate_from_df(df: float, tau: float) -> float:
    """Average cont-comp rate over [0, tau] from a discount factor.

    For deterministic rates: df = exp(-∫ r dt) -> r_avg = -log(df)/tau
    """
    tau = float(tau)
    df = float(df)
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    if df <= 0.0:
        raise ValueError("df must be > 0")
    return -math.log(df) / tau


def avg_carry_from_forward(spot: float, forward: float, tau: float) -> float:
    """Average carry (r-q) over [0, tau] inferred from forward/spot.

    For deterministic carry: F = S0 * exp(∫ (r-q) dt) -> b_avg = log(F/S0)/tau
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
