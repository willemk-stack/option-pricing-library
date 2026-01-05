# Public API

## Flat convenience API

::: option_pricing.types
    options:
      members:
        - OptionType
        - MarketData
        - OptionSpec
        - PricingInputs

## Pricers (PricingInputs)

::: option_pricing.pricers.black_scholes
    options:
      members:
        - bs_price
        - bs_greeks

::: option_pricing.pricers.mc
    options:
      members:
        - mc_price

::: option_pricing.pricers.tree
    options:
      members:
        - binom_price
