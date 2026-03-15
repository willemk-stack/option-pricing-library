# API

The `option_pricing` package exposes three complementary user-facing styles:

- **Flat convenience API**: `MarketData` + `OptionSpec` + `PricingInputs`
- **Curves-first API**: `PricingContext`, `DiscountCurve`, and `ForwardCurve`
- **Instrument-based API**: `VanillaOption` and `ExerciseStyle`

Use the pages below as the main entry points into the library surface.

## Recommended API path

- **Recommended API**: instrument-based workflow (`VanillaOption` + instrument pricers). This is the intended public entry point for most users.
- **Convenience API**: flat-input workflow (`PricingInputs`). Use this for compact tutorials and quick checks.
- **Advanced API**: curves-first workflow (`PricingContext`) and the volatility / PDE modules for term-structure or surface-heavy use cases.

## Overview

- [Public API](public.md) - top-level types, instruments, configs, and common objects re-exported from `option_pricing`
- [Curves-first API](curves.md) - `PricingContext` and curve objects
- [Pricers](pricers.md) - Black-Scholes, Monte Carlo, and Binomial entry points for all three API styles
- [Volatility](vol.md) - implied vol inversion, smiles, and surfaces
- [Exceptions](exceptions.md) - error types you may want to catch

## Quick snippets

### Recommended API (instrument-based)

```python
from option_pricing import (
    ExerciseStyle,
    MarketData,
    OptionType,
    VanillaOption,
    bs_price_instrument,
)

inst = VanillaOption(
    expiry=1.0,
    strike=100.0,
    kind=OptionType.CALL,
    exercise=ExerciseStyle.EUROPEAN,
)

market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
price = bs_price_instrument(inst, market=market, sigma=0.2)
```

### Convenience API (flat inputs)

```python
from option_pricing import MarketData, OptionSpec, PricingInputs, OptionType, bs_price

market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
p = PricingInputs(spec=spec, market=market, sigma=0.2)

price = bs_price(p)
```

### Advanced API (curves-first)

```python
from option_pricing import (
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    OptionType,
    PricingContext,
    bs_price_from_ctx,
)

ctx = PricingContext(
    spot=100.0,
    discount=FlatDiscountCurve(r=0.03),
    forward=FlatCarryForwardCurve(spot=100.0, r=0.03, q=0.01),
)

price = bs_price_from_ctx(
    kind=OptionType.CALL,
    strike=100.0,
    sigma=0.2,
    tau=1.0,
    ctx=ctx,
)
```

### Instrument-based API (alternate view)

See the recommended snippet above, or use the [Pricers](pricers.md) page for the full list of instrument entry points.

## Notes

- Times are expressed in years.
- `PricingInputs` uses absolute expiry `T` together with valuation time `t`, and exposes `tau` as `T - t`.
- With the default `t=0`, the numeric value of `expiry` happens to equal `tau` in the flat-input examples.
- `PricingContext` and instrument-based pricers work directly with `tau` (time to expiry).
