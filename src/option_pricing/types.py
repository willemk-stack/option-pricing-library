import math
from dataclasses import dataclass
from enum import StrEnum

from .market.curves import FlatCarryForwardCurve, FlatDiscountCurve, PricingContext


class OptionType(StrEnum):
    """Option type enumeration.

    Attributes
    ----------
    CALL
        European or American call option.
    PUT
        European or American put option.
    """

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class MarketData:
    """Flat market data container.

    Primarily a convenience structure for common Black-Scholes and binomial tree scenarios.
    For richer market structures (term structures, stochastic rates), use
    :class:`~option_pricing.market.curves.PricingContext` instead.

    Attributes
    ----------
    spot
        Spot price at valuation time.
    rate
        Continuously-compounded domestic risk-free rate.
    dividend_yield
        Continuously-compounded dividend yield (default: 0.0).
    """

    spot: float
    rate: float
    dividend_yield: float = 0.0

    def df(self, T: float, t: float = 0.0) -> float:
        """Discount factor from time t to T.

        Parameters
        ----------
        T
            Maturity time.
        t
            Valuation time (default: 0.0).

        Returns
        -------
        float
            Discount factor P(0, T-t) = exp(-r * tau).
        """
        tau = float(T) - float(t)
        if tau < 0:
            raise ValueError("T must be >= t")
        return math.exp(-self.rate * tau)

    def forward(self, T: float, t: float = 0.0) -> float:
        """Forward price from time t for delivery at T.

        Uses the cost-of-carry model: F(t,T) = S(t) * exp((r - q) * (T - t)).

        Parameters
        ----------
        T
            Maturity time.
        t
            Valuation time (default: 0.0).

        Returns
        -------
        float
            Forward price.
        """
        tau = float(T) - float(t)
        if tau < 0:
            raise ValueError("T must be >= t")
        return self.spot * math.exp((self.rate - self.dividend_yield) * tau)

    def fwd(self, T: float, t: float = 0.0) -> float:
        """Alias for :meth:`forward`."""
        return self.forward(T, t)

    def to_context(self) -> PricingContext:
        """Convert to a :class:`~option_pricing.market.curves.PricingContext`.

        Returns
        -------
        PricingContext
            Curves-first market container with flat discount and forward curves.
        """
        discount = FlatDiscountCurve(self.rate)
        forward = FlatCarryForwardCurve(
            spot=self.spot, r=self.rate, q=self.dividend_yield
        )
        return PricingContext(spot=self.spot, discount=discount, forward=forward)


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Specification of a vanilla option.

    Attributes
    ----------
    kind
        Option type (call or put).
    strike
        Strike price.
    expiry
        Expiry time in years.

        In the legacy :class:`PricingInputs` workflow this is interpreted as the
        absolute expiry ``T`` and the remaining time is computed as
        ``tau = expiry - t``. With the default ``t=0``, this is numerically equal
        to time-to-expiry.
    """

    kind: OptionType
    strike: float
    expiry: float


@dataclass(frozen=True, slots=True)
class DigitalSpec(OptionSpec):
    """Specification of a digital (binary) option.

    Extends :class:`OptionSpec` with a fixed payout amount.

    Attributes
    ----------
    payout
        Fixed payoff amount at expiry (default: 1.0).
    """

    payout: float = 1.0


@dataclass(frozen=True, slots=True)
class PricingInputs[SpecT: OptionSpec]:
    """All inputs needed to price an option.

    Generic over the option specification (vanilla, digital, etc.).

    Attributes
    ----------
    spec
        Option specification (:class:`OptionSpec` or subclass).
    market
        Market data (:class:`MarketData`).
    sigma
        Implied volatility.
    t
        Current valuation time (default: 0.0).

    Notes
    -----
    ``PricingInputs`` follows the legacy flat-input convention where
    ``spec.expiry`` is the absolute expiry ``T`` and ``tau = T - t``.
    """

    spec: SpecT
    market: MarketData
    sigma: float
    t: float = 0.0

    @property
    def S(self) -> float:
        """Spot price from market data."""
        return self.market.spot

    @property
    def ctx(self) -> PricingContext:
        """Pricing context derived from market data."""
        return self.market.to_context()

    @property
    def K(self) -> float:
        """Strike price from option spec."""
        return self.spec.strike

    @property
    def T(self) -> float:
        """Absolute expiry time ``T`` from the option spec."""
        return self.spec.expiry

    @property
    def tau(self) -> float:
        """Remaining time to expiry ``tau = T - t``."""
        tau = float(self.T - self.t)
        if tau <= 0.0:
            raise ValueError("Need expiry > t")
        return tau

    @property
    def df(self) -> float:
        """Discount factor from current time to expiry."""
        return self.ctx.df(self.tau)

    @property
    def F(self) -> float:
        """Forward price for delivery at expiry."""
        return self.ctx.fwd(self.tau)


# Convenience alias
type PricingInputsDigital = PricingInputs[DigitalSpec]
