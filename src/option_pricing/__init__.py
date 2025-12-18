"""
option_pricing

Core option pricing library.

This package exposes the main user-facing functions at the top level, so you
can write, for example:

    from option_pricing import bs_call, mc_european_call
"""

# Re-export pricing entrypoints (nice public names)
from .pricers.black_scholes import bs_greeks, bs_price
from .pricers.mc import mc_price
from .pricers.tree import binom_price
from .types import MarketData, OptionSpec, OptionType, PricingInputs

__all__ = [
    # Types
    "OptionType",
    "OptionSpec",
    "MarketData",
    "PricingInputs",
    # Pricers
    "bs_price",
    "bs_greeks",
    "mc_price",
    "binom_price",
]
