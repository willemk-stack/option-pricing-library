import math
from dataclasses import dataclass
from enum import Enum

from .market.curves import FlatCarryForwardCurve, FlatDiscountCurve, PricingContext


class OptionType(str, Enum):
    """Option contract type.

    An enumeration of plain-vanilla option types.

    Attributes
    ----------
    CALL : str
        Call option ("call").
    PUT : str
        Put option ("put").
    """

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market observables for equity vanilla options.

    This remains a *flat* convenience wrapper (spot, r, q) for tutorials/tests.
    Internally, pricing functions can convert this to a curves-first
    :class:`~option_pricing.market.curves.PricingContext` via :meth:`to_context`.
    """

    spot: float
    rate: float
    dividend_yield: float = 0.0

    def df(self, T: float, t: float = 0.0) -> float:
        """Risk-free discount factor ``P(t,T)`` under continuous compounding."""
        tau = float(T) - float(t)
        if tau < 0:
            raise ValueError("T must be >= t")
        return math.exp(-self.rate * tau)

    def forward(self, T: float, t: float = 0.0) -> float:
        """Forward price ``F(t,T)`` under a flat cost-of-carry model."""
        tau = float(T) - float(t)
        if tau < 0:
            raise ValueError("T must be >= t")
        return self.spot * math.exp((self.rate - self.dividend_yield) * tau)

    def to_context(self) -> PricingContext:
        """Convert flat quotes (spot, r, q) into a curves-first pricing context."""
        discount = FlatDiscountCurve(self.rate)
        forward = FlatCarryForwardCurve(
            spot=self.spot, r=self.rate, q=self.dividend_yield
        )
        return PricingContext(spot=self.spot, discount=discount, forward=forward)


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Specification of a plain-vanilla European option.

    Parameters
    ----------
    kind : OptionType
        Option type (call or put).
    strike : float
        Strike price of the option, typically denoted :math:`K`.
    expiry : float
        Option expiry time in the same time units as `t` in :class:`PricingInputs`
        (commonly years).

    Notes
    -----
    No explicit calendar/day count convention is enforced; it is the caller's
    responsibility to keep units consistent.
    """

    kind: OptionType
    strike: float
    expiry: float


@dataclass(frozen=True, slots=True)
class PricingInputs:
    """Pricing input bundle.

    This object glues together the option contract, the market observables, and the
    volatility parameter used by model pricers.

    Notes
    -----
    The public ``MarketData`` object remains flat (spot, r, q) for ergonomics, but
    internally the library prices off *curves-first* inputs. Access :attr:`ctx` to
    obtain a :class:`~option_pricing.market.curves.PricingContext`.
    """

    spec: OptionSpec
    market: MarketData
    sigma: float
    t: float = 0.0

    @property
    def S(self) -> float:
        return self.market.spot

    @property
    def ctx(self) -> PricingContext:
        """Curves-first view of the market inputs at the pricing time."""
        return self.market.to_context()

    @property
    def K(self) -> float:
        return self.spec.strike

    @property
    def r(self) -> float:
        return self.market.rate

    @property
    def q(self) -> float:
        return self.market.dividend_yield

    @property
    def T(self) -> float:
        return self.spec.expiry

    @property
    def tau(self) -> float:
        tau = float(self.T - self.t)
        if tau <= 0.0:
            raise ValueError("Need expiry > t")
        return tau

    @property
    def df(self) -> float:
        return self.ctx.df(self.tau)

    @property
    def F(self) -> float:
        return self.ctx.fwd(self.tau)
