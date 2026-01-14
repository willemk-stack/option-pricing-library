"""option_pricing.instruments

Instrument definitions ("what is being priced").

This package is intentionally model- and pricer-agnostic. Instruments expose a
small interface that pricing engines can consume:

- time to expiry (``expiry``; in years or consistent units)
- exercise style (European / American)
- a payoff callable

``pricers`` then implement algorithms (BS/Black-76, trees, Monte Carlo, ...)
that take an instrument plus model/market inputs.
"""

from .base import (
    ExerciseStyle,
    PathInstrument,
    PathPayoff,
    TerminalInstrument,
    TerminalPayoff,
)
from .factory import from_option_spec, from_pricing_inputs
from .vanilla import VanillaOption, VanillaPayoff

__all__ = [
    "ExerciseStyle",
    "TerminalPayoff",
    "PathPayoff",
    "Instrument",
    "TerminalInstrument",
    "PathInstrument",
    "VanillaPayoff",
    "VanillaOption",
    "from_option_spec",
    "from_pricing_inputs",
]
