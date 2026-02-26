# API

The `option_pricing` package exposes three complementary user-facing styles:

- **Flat convenience API**: `MarketData` + `OptionSpec` + `PricingInputs`
- **Curves-first API**: `PricingContext`, `DiscountCurve`, and `ForwardCurve`
- **Instrument-based API**: `VanillaOption` and `ExerciseStyle`

Use the pages below as the main entry points into the library surface.

## Overview

- [Public API](public.md) - top-level types, instruments, configs, and common objects re-exported from `option_pricing`
- [Curves-first API](curves.md) - `PricingContext` and curve objects
- [Pricers](pricers.md) - Black-Scholes, Monte Carlo, and Binomial entry points for all three API styles
- [Volatility](vol.md) - implied vol inversion, smiles, and surfaces
- [Exceptions](exceptions.md) - error types you may want to catch

## Quick snippets

### Flat convenience API

```python
from option_pricing import MarketData, OptionSpec, PricingInputs, OptionType, bs_price

market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
p = PricingInputs(spec=spec, market=market, sigma=0.2)

price = bs_price(p)
```

### Curves-first API

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

### Instrument-based API

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

## Notes

- Times are expressed in years.
- `PricingInputs` uses absolute expiry `T` together with valuation time `t`, and exposes `tau` as `T - t`.
- `PricingContext` and instrument-based pricers work directly with `tau` (time to expiry).
