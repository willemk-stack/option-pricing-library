"""Protocols for Monte Carlo simulators."""

from typing import Protocol

from ..market.curves import PricingContext
from ..typing import FloatArray
from .config import MCConfig


class TerminalSimulator(Protocol):
    """Simulator that produces terminal underlying samples."""

    def simulate_terminal(
        self,
        *,
        ctx: PricingContext,
        tau: float,
        cfg: MCConfig,
    ) -> FloatArray: ...


class PathSimulator(Protocol):
    """Simulator that produces path samples with path dimension first."""

    def simulate_paths(
        self,
        *,
        ctx: PricingContext,
        tau: float,
        cfg: MCConfig,
    ) -> FloatArray: ...
