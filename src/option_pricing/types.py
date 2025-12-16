from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PricingInputs:
    t: float      # valuation time
    S: float      # underlying price at time t
    K: float      # strike
    r: float      # risk-free rate (cont. comp.)
    sigma: float  # volatility
    T: float      # maturity
