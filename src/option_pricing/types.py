from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class MarketData:
    spot: float
    rate: float
    dividend_yield: float = 0.0


@dataclass(frozen=True, slots=True)
class OptionSpec:
    kind: OptionType
    strike: float
    expiry: float


@dataclass(frozen=True, slots=True)
class PricingInputs:
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
