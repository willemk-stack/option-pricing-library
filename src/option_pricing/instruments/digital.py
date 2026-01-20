from dataclasses import dataclass
from typing import overload

import numpy as np

from ..types import OptionType
from ..typing import FloatArray
from .base import ExerciseStyle, TerminalPayoff


@overload
def digital_call_payoff(ST: float, K: float, payout: float = 1.0) -> float: ...
@overload
def digital_call_payoff(
    ST: FloatArray, K: float, payout: float = 1.0
) -> FloatArray: ...
def digital_call_payoff(
    ST: float | FloatArray, K: float, payout: float = 1.0
) -> float | FloatArray:
    if np.isscalar(ST):
        return float(payout) if ST >= K else 0.0
    ST_arr = np.asarray(ST, dtype=float)
    return np.where(ST_arr >= K, float(payout), 0.0)


@overload
def digital_put_payoff(ST: float, K: float, payout: float = 1.0) -> float: ...
@overload
def digital_put_payoff(ST: FloatArray, K: float, payout: float = 1.0) -> FloatArray: ...
def digital_put_payoff(
    ST: float | FloatArray, K: float, payout: float = 1.0
) -> float | FloatArray:
    if np.isscalar(ST):
        return float(payout) if ST <= K else 0.0
    ST_arr = np.asarray(ST, dtype=float)
    return np.where(ST_arr <= K, float(payout), 0.0)


@dataclass(frozen=True, slots=True)
class DigitalPayoff:
    kind: OptionType
    strike: float
    payout: float = 1.0

    @overload
    def __call__(self, ST: float) -> float: ...
    @overload
    def __call__(self, ST: FloatArray) -> FloatArray: ...

    def __call__(self, ST: float | FloatArray) -> float | FloatArray:
        if self.kind == OptionType.CALL:
            out = digital_call_payoff(ST, K=self.strike, payout=self.payout)
        elif self.kind == OptionType.PUT:
            out = digital_put_payoff(ST, K=self.strike, payout=self.payout)
        else:
            raise ValueError(f"Unsupported option kind: {self.kind}")

        return float(out) if np.ndim(out) == 0 else out


@dataclass(slots=True, frozen=True)
class DigitalOption:
    expiry: float
    strike: float
    payout: float
    kind: OptionType
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    @property
    def payoff(self) -> TerminalPayoff:
        return DigitalPayoff(kind=self.kind, strike=self.strike, payout=self.payout)
