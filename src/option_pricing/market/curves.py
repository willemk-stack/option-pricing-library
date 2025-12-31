from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class DiscountCurve(Protocol):
    def df(self, T: float) -> float: ...
    def __call__(self, T: float) -> float: ...


class ForwardCurve(Protocol):
    def forward(self, T: float, t: float = 0.0) -> float: ...
    def __call__(self, T: float) -> float: ...


@dataclass(frozen=True, slots=True)
class FlatDiscountCurve:
    r: float

    def df(self, T: float) -> float:
        T = float(T)
        if T < 0:
            raise ValueError("T must be >= 0")
        return math.exp(-self.r * T)

    def __call__(self, T: float) -> float:
        return self.df(T)


@dataclass(frozen=True, slots=True)
class FlatCarryForwardCurve:
    """
    Forward curve with constant (continuous) carry.

    F(t,T) = S_t * exp((r - q) * (T - t))

    - r: risk-free rate (continuous)
    - q: dividend yield / foreign rate / carry yield (continuous)
    """

    spot: float
    r: float
    q: float = 0.0

    def forward(self, T: float, t: float = 0.0) -> float:
        tau = float(T) - float(t)
        if tau < 0.0:
            raise ValueError("T must be >= t")
        if self.spot <= 0.0:
            raise ValueError("spot must be > 0")
        return self.spot * math.exp((self.r - self.q) * tau)

    def __call__(self, T: float) -> float:
        return self.forward(T)


@dataclass(frozen=True, slots=True)
class PricingContext:
    forward: Callable[[float], float]  # forward(T)->float
    df: Callable[[float], float]  # df(T)->float
