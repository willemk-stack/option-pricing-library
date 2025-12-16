from dataclasses import dataclass
from enum import Enum

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class MarketData:
    spot: float   # same as your S
    rate: float   # same as your r
    dividend_yield: float = 0.0

@dataclass(frozen=True, slots=True)
class OptionSpec:
    kind: OptionType
    strike: float
    expiry: float  # or datetime/date later
    
@dataclass(frozen=True, slots=True)
class PricingInputs:
    spec: OptionSpec
    market: MarketData
    sigma: float
    t: float = 0.0

    @property
    def S(self): return self.market.spot
    @property
    def K(self): return self.spec.strike
    @property
    def r(self): return self.market.rate
    @property
    def T(self): return self.spec.expiry
