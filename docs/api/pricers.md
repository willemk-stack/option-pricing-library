# Pricers

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

## Pricers (curves-first)

::: option_pricing.pricers.black_scholes
    options:
      members:
        - bs_price_from_ctx
        - bs_greeks_from_ctx

::: option_pricing.pricers.mc
    options:
      members:
        - mc_price_from_ctx

::: option_pricing.pricers.tree
    options:
      members:
        - binom_price_from_ctx