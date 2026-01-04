# API Reference

## Public entry points

::: option_pricing
    options:
      show_source: false
      members:
        # Flat convenience API
        - OptionType
        - MarketData
        - OptionSpec
        - PricingInputs

        # Curves-first market API
        - PricingContext
        - DiscountCurve
        - ForwardCurve
        - FlatDiscountCurve
        - FlatCarryForwardCurve

        # Pricers (PricingInputs)
        - bs_price
        - bs_greeks
        - mc_price
        - binom_price

        # Pricers (curves-first)
        - bs_price_from_ctx
        - bs_greeks_from_ctx
        - mc_price_from_ctx
        - binom_price_from_ctx

        # Implied volatility
        - implied_vol_bs
        - implied_vol_bs_result

        # Volatility objects
        - VolSurface
        - Smile

## Exceptions

::: option_pricing.exceptions
    options:
      show_source: false
      members:
        - InvalidOptionPriceError
        - RootFindingError
        - NoConvergenceError
        - NotBracketedError
        - NoBracketError
        - DerivativeTooSmallError
