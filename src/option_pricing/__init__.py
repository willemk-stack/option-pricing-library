"""
option_pricing

Core option pricing library.

This package exposes the main user-facing functions at the top level, so you
can write, for example:

    from option_pricing import bs_call, mc_european_call
"""

# Re-export core types
from .types import OptionSpec, OptionType, MarketData, PricingInputs

# Re-export pricing entrypoints (nice public names)
from .models.bs import bs_call_from_inputs as bs_price_call
# from .models.bs import bs_greeks  # uncomment if you have it

from .pricers.mc import mc_call_from_inputs as mc_price_call
from .pricers.mc import mc_put_from_inputs as mc_price_put
from .models.binomial_crr import binom_call_from_inputs as binom_price_call
from .models.binomial_crr import binom_put_from_inputs as binom_price_put

__all__ = [
    # Types
    "OptionType",
    "OptionSpec",
    "MarketData",
    "PricingInputs",

    # Pricers
    "mc_price_call",
    "mc_price_put",
    "binom_price_call",
    "binom_price_put",
    # "bs_greeks",
]