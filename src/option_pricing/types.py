import math
from dataclasses import dataclass
from enum import Enum

from .market.curves import FlatCarryForwardCurve, FlatDiscountCurve, PricingContext


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class MarketData:
    spot: float
    rate: float
    dividend_yield: float = 0.0

    def df(self, T: float, t: float = 0.0) -> float:
        tau = float(T) - float(t)
        if tau < 0:
            raise ValueError("T must be >= t")
        return math.exp(-self.rate * tau)

    def forward(self, T: float, t: float = 0.0) -> float:
        tau = float(T) - float(t)
        if tau < 0:
            raise ValueError("T must be >= t")
        return self.spot * math.exp((self.rate - self.dividend_yield) * tau)

    def fwd(self, T: float, t: float = 0.0) -> float:
        return self.forward(T, t)

    def to_context(self) -> PricingContext:
        discount = FlatDiscountCurve(self.rate)
        forward = FlatCarryForwardCurve(
            spot=self.spot, r=self.rate, q=self.dividend_yield
        )
        return PricingContext(spot=self.spot, discount=discount, forward=forward)


@dataclass(frozen=True, slots=True)
class OptionSpec:
    kind: OptionType
    strike: float
    expiry: float


@dataclass(frozen=True, slots=True)
class DigitalSpec(OptionSpec):
    payout: float = 1.0


@dataclass(frozen=True, slots=True)
class PricingInputs[SpecT: OptionSpec]:
    spec: SpecT
    market: MarketData
    sigma: float
    t: float = 0.0

    @property
    def S(self) -> float:
        return self.market.spot

    @property
    def ctx(self) -> PricingContext:
        return self.market.to_context()

    @property
    def K(self) -> float:
        return self.spec.strike

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


# Convenience alias
type PricingInputsDigital = PricingInputs[DigitalSpec]
