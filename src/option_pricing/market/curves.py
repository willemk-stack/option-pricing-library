import math


def discount_factor(rate: float, tau: float) -> float:
    return math.exp(-rate * tau)


def forward(spot: float, r: float, q: float, tau: float) -> float:
    # F = S * e^{(r-q) tau}
    return spot * math.exp((r - q) * tau)
