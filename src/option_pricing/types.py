from dataclasses import dataclass
from enum import Enum


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
    """Market observables needed for option pricing.

    Parameters
    ----------
    spot : float
        Current spot price of the underlying, typically denoted :math:`S`.
    rate : float
        Continuously-compounded risk-free interest rate, typically denoted :math:`r`
        (annualized).
    dividend_yield : float, default 0.0
        Continuously-compounded dividend yield (or foreign risk-free rate for FX),
        typically denoted :math:`q` (annualized).

    Notes
    -----
    This container assumes continuous compounding for both `rate` and `dividend_yield`.
    """

    spot: float
    rate: float
    dividend_yield: float = 0.0


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
    """Inputs for an option pricing routine.

    This class bundles an option specification, market data, and model parameters
    (e.g., volatility) into a single immutable object. It also provides convenient
    aliases for commonly-used symbols (:math:`S, K, r, q, T`) and computes time-to-expiry.

    Parameters
    ----------
    spec : OptionSpec
        Option contract specification.
    market : MarketData
        Market observables (spot, rates, yields).
    sigma : float
        Volatility parameter (annualized). Interpretation depends on the pricing model
        but is typically Black-Scholes volatility.
    t : float, default 0.0
        Current valuation time in the same units as `spec.expiry`.

    Attributes
    ----------
    S : float
        Alias for ``market.spot``.
    K : float
        Alias for ``spec.strike``.
    r : float
        Alias for ``market.rate``.
    q : float
        Alias for ``market.dividend_yield``.
    T : float
        Alias for ``spec.expiry``.
    tau : float
        Time to expiry, computed as ``T - t``.

    Raises
    ------
    ValueError
        If ``T - t <= 0`` when accessing :attr:`tau`.

    Notes
    -----
    - This object is frozen (immutable) and uses slots for reduced memory overhead.
    - Time units are not enforced; ensure `t` and `T` share the same convention.
    """

    spec: OptionSpec
    market: MarketData
    sigma: float
    t: float = 0.0

    @property
    def S(self) -> float:
        return self.market.spot

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
        tau = self.T - self.t
        if tau <= 0.0:
            raise ValueError("Need expiry > t")
        return tau
