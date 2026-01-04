"""
option_pricing

Core option pricing library.

This package exposes the main user-facing functions at the top level, so you
can write, for example:

    from option_pricing import (
    MarketData, OptionSpec, OptionType, PricingInputs,
    bs_price, bs_greeks, mc_price, binom_price,
)

"""

# Re-export pricing entrypoints (nice public names)
from .config import ImpliedVolConfig, MCConfig, RandomConfig
from .numerics.root_finding import RootMethod
from .pricers.black_scholes import bs_greeks, bs_price
from .pricers.mc import mc_price
from .pricers.tree import binom_price
from .types import MarketData, OptionSpec, OptionType, PricingInputs
from .vol.implied_vol import implied_vol_bs, implied_vol_bs_result
from .vol.surface import Smile, VolSurface

# InvalidOptionPriceError (from option_pricing.exceptions)
__all__ = [
    # Types
    "OptionType",
    "OptionSpec",
    "MarketData",
    "PricingInputs",
    # Configs
    "MCConfig",
    "RandomConfig",
    "ImpliedVolConfig",
    "RootMethod",
    # Pricers
    "bs_price",
    "bs_greeks",
    "mc_price",
    "binom_price",
    # Implied vol
    "implied_vol_bs",
    "implied_vol_bs_result",
    # Vol objects
    "VolSurface",
    "Smile",
]
