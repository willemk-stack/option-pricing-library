"""Lightweight instrument interfaces.

The library exposes both a "PricingInputs" API (flat, tutorial-friendly) and an
instrument-based API for pricers that operate on *terminal payoffs*.

Only a small subset is required by the pricers in this repository:
- an exercise style (European/American)
- a time-to-expiry (tau)
- a vectorizable payoff function of the terminal underlying price
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from option_pricing.typing import FloatArray


class ExerciseStyle(str, Enum):
    """Exercise style for vanilla options."""

    EUROPEAN = "european"
    AMERICAN = "american"


@runtime_checkable
class TerminalPayoff(Protocol):
    """Vectorizable terminal payoff callable.

    We model the payoff as a callable object (or function) that supports both
    scalar and vector evaluation.
    """

    @overload
    def __call__(self, ST: float) -> float:  # pragma: no cover
        ...

    @overload
    def __call__(self, ST: FloatArray) -> FloatArray:  # pragma: no cover
        ...

    def __call__(
        self, ST: NDArray[np.floating] | float
    ) -> FloatArray | float:  # pragma: no cover
        ...


@runtime_checkable
class TerminalInstrument(Protocol):
    @property
    def expiry(self) -> float: ...

    @property
    def exercise(self) -> ExerciseStyle: ...

    # payoff is a callable attribute (modelled as a read-only property)
    @property
    def payoff(self) -> TerminalPayoff: ...


# --- Backwards-compatible names (used by older notebooks / docs)
Instrument = TerminalInstrument


class PathPayoff(Protocol):
    """Path-dependent payoff callable (placeholder interface)."""

    def __call__(self, paths: FloatArray) -> FloatArray:  # pragma: no cover
        ...


@runtime_checkable
class PathInstrument(Protocol):
    @property
    def expiry(self) -> float: ...

    @property
    def exercise(self) -> ExerciseStyle: ...

    @property
    def payoff(self) -> PathPayoff: ...
