"""Backwards-compatible payoff helpers.

Historically the library exposed a small set of payoff functions in
``instruments.payoffs`` and pricers imported them directly.

With the move to an ``instruments/`` package, the canonical implementation of
vanilla payoffs lives in :mod:`option_pricing.instruments.vanilla`.

This module keeps the old names to avoid churn in pricers/notebooks.
"""

from __future__ import annotations

from ..types import OptionType
from .base import TerminalPayoff
from .vanilla import VanillaPayoff, call_payoff, put_payoff


def make_vanilla_payoff(kind: OptionType, *, K: float) -> TerminalPayoff:
    """Factory returning a vectorized vanilla payoff callable.

    The returned object is a :class:`~option_pricing.instruments.vanilla.VanillaPayoff`
    instance (which is still a callable), keeping the old API contract intact.
    """

    return VanillaPayoff(kind=kind, strike=float(K))


__all__ = [
    "call_payoff",
    "put_payoff",
    "VanillaPayoff",
    "make_vanilla_payoff",
]
