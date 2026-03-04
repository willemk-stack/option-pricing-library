# Pricers

The library exposes parallel entry points for three styles of usage:

- `PricingInputs` wrappers for compact workflows
- `*_from_ctx` functions for curves-first workflows
- instrument-based functions for reusable contracts and exercise handling

## Black-Scholes / Black-76 with `PricingInputs`

::: option_pricing.pricers.black_scholes
    options:
      members:
        - bs_price
        - bs_greeks

## Monte Carlo with `PricingInputs`

::: option_pricing.pricers.mc
    options:
      members:
        - mc_price

## Binomial CRR with `PricingInputs`

::: option_pricing.pricers.tree
    options:
      members:
        - binom_price

## Curves-first pricers

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

## Instrument-based pricers

::: option_pricing.pricers.black_scholes
    options:
      members:
        - bs_price_instrument
        - bs_price_instrument_from_ctx
        - bs_greeks_instrument
        - bs_greeks_instrument_from_ctx

::: option_pricing.pricers.mc
    options:
      members:
        - mc_price_instrument
        - mc_price_instrument_from_ctx

::: option_pricing.pricers.tree
    options:
      members:
        - binom_price_instrument
        - binom_price_instrument_from_ctx

## Notes

- `mc_price*` functions return `(price, std_error)`.
- Black-Scholes closed-form instrument pricers support European exercise only.
- Tree-based instrument pricers can use the instrument's exercise style, including American exercise where supported.
