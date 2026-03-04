# Curves-first API

The curves-first interface makes discounting and forwards explicit. All pricers in this
style accept a `PricingContext` plus `tau` (time to expiry).

## Context and curves

::: option_pricing.market.curves
    options:
      members:
        - PricingContext
        - DiscountCurve
        - ForwardCurve
        - FlatDiscountCurve
        - FlatCarryForwardCurve

## Notes

- `PricingContext.df(tau)` returns the discount factor to maturity `tau`.
- `PricingContext.fwd(tau)` returns the forward price for maturity `tau`.
- `MarketData.to_context()` is the bridge from the flat convenience API into this one.
