"""Vanilla (call/put) instruments and payoffs.

This module provides two small building blocks:

- :class:`VanillaPayoff` : a vectorized terminal payoff (call/put)
- :class:`VanillaOption` : an instrument wrapper bundling expiry + exercise + payoff
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np

from ..types import OptionType
from ..typing import FloatArray
from .base import ExerciseStyle


@overload
def call_payoff(ST: float, K: float) -> float: ...
@overload
def call_payoff(ST: FloatArray, K: float) -> FloatArray: ...
def call_payoff(ST: float | FloatArray, K: float) -> float | FloatArray:
    return np.maximum(ST - K, 0.0)


@overload
def put_payoff(ST: float, K: float) -> float: ...
@overload
def put_payoff(ST: FloatArray, K: float) -> FloatArray: ...
def put_payoff(ST: float | FloatArray, K: float) -> float | FloatArray:
    return np.maximum(K - ST, 0.0)


@dataclass(frozen=True, slots=True)
class VanillaPayoff:
    """Callable, vectorized call/put payoff."""

    kind: OptionType
    strike: float

    @overload
    def __call__(self, ST: float) -> float: ...
    @overload
    def __call__(self, ST: FloatArray) -> FloatArray: ...

    def __call__(self, ST: float | FloatArray) -> float | FloatArray:
        if self.kind == OptionType.CALL:
            out = call_payoff(ST, K=self.strike)
        elif self.kind == OptionType.PUT:
            out = put_payoff(ST, K=self.strike)
        else:
            raise ValueError(f"Unsupported option kind: {self.kind}")

        # Ensure scalar input returns a Python float (avoids np.float64 vs float typing issues)
        if np.ndim(out) == 0:
            return float(out)
        return out


@dataclass(frozen=True, slots=True)
class VanillaOption:
    """Vanilla option instrument.

    Notes
    -----
    ``expiry`` is interpreted as **time to expiry** (tau).
    """

    expiry: float
    strike: float
    kind: OptionType
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    from .base import TerminalPayoff

    @property
    def payoff(self) -> TerminalPayoff:
        return VanillaPayoff(kind=self.kind, strike=self.strike)

    @overload
    def intrinsic_value(self, spot: float) -> float: ...
    @overload
    def intrinsic_value(self, spot: FloatArray) -> FloatArray: ...

    def intrinsic_value(self, spot: float | FloatArray) -> float | FloatArray:
        return self.payoff(spot)
