# Black-Scholes

This guide covers the analytic Black-Scholes / Black-76 pricing entry points for European vanilla options.

## Price a call or put with `PricingInputs`

```python
from option_pricing import (
    MarketData,
    OptionSpec,
    OptionType,
    PricingInputs,
    bs_price,
)

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.01)

call_inputs = PricingInputs(
    spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0),
    market=market,
    sigma=0.20,
    t=0.0,
)

put_inputs = PricingInputs(
    spec=OptionSpec(kind=OptionType.PUT, strike=100.0, expiry=1.0),
    market=market,
    sigma=0.20,
    t=0.0,
)

call_px = bs_price(call_inputs)
put_px = bs_price(put_inputs)
```

`bs_price` dispatches on `p.spec.kind`, so there is no separate function you need to call for puts.

## Analytic Greeks

```python
from option_pricing import bs_greeks

g = bs_greeks(call_inputs)
print(g["price"], g["delta"], g["gamma"], g["vega"], g["theta"])
```

## Curves-first pricing

```python
from option_pricing import OptionType, bs_price_from_ctx, bs_greeks_from_ctx

ctx = market.to_context()
tau = call_inputs.tau

price = bs_price_from_ctx(
    kind=OptionType.CALL,
    strike=100.0,
    sigma=0.20,
    tau=tau,
    ctx=ctx,
)

greeks = bs_greeks_from_ctx(
    kind=OptionType.CALL,
    strike=100.0,
    sigma=0.20,
    tau=tau,
    ctx=ctx,
)
```

## Instrument-based pricing

```python
from option_pricing import ExerciseStyle, VanillaOption, bs_price_instrument

inst = VanillaOption(
    expiry=1.0,
    strike=100.0,
    kind=OptionType.CALL,
    exercise=ExerciseStyle.EUROPEAN,
)

price = bs_price_instrument(inst, market=market, sigma=0.20)
```

## Sweep spot and inspect the Greeks profile

The diagnostics helper returns arrays that you can plot however you like:

```python
import numpy as np
from option_pricing.diagnostics.greeks.sweep import sweep_spot_greeks

spots = np.linspace(60.0, 140.0, 81)
out = sweep_spot_greeks(call_inputs, spots, method="analytic")

print(out.x.shape)
print(out.delta[:5])
```

## Notes

- These analytic formulas support European exercise only.
- The curves-first implementation prices in Black-76 form using `forward` and `discount` from `PricingContext`.
- For finite-difference Black-Scholes pricing, see [PDE pricing](pde_pricing.md).
