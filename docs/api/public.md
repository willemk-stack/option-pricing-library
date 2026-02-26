# Public API

This page covers the main objects re-exported from `option_pricing`.
For pricing functions, see [Pricers](pricers.md).

## Core types

::: option_pricing.types
    options:
      members:
        - OptionType
        - MarketData
        - OptionSpec
        - PricingInputs

## Instrument layer

::: option_pricing.instruments.base
    options:
      members:
        - ExerciseStyle

::: option_pricing.instruments.vanilla
    options:
      members:
        - VanillaPayoff
        - VanillaOption

## Market context and curves

These names are re-exported from the package root:

- `PricingContext`
- `DiscountCurve`
- `ForwardCurve`
- `FlatDiscountCurve`
- `FlatCarryForwardCurve`

See the dedicated [Curves-first API](curves.md) page for the canonical class documentation.

## Configuration and solver selection

::: option_pricing.config
    options:
      members:
        - RandomConfig
        - MCConfig
        - ImpliedVolConfig

::: option_pricing.numerics.root_finding
    options:
      members:
        - RootMethod

## Volatility objects

These names are re-exported from the package root:

- `VolSurface`
- `Smile`

See the dedicated [Volatility](vol.md) page for the canonical class documentation.
