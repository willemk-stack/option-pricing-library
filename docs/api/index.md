# API

The `option_pricing` package exposes two ways to work:

- **Flat convenience API** (spot, r, q): great for quick pricing and greeks
- **Curves-first API**: explicit discount/forward curves for more realistic term structures

Use the pages below as entry points.

## Overview

- [Public API](public.md) — core dataclasses and the main entry points
- [Curves-first API](curves.md) — `PricingContext` and curve objects
- [Pricers](pricers.md) — Black–Scholes, Monte Carlo, Binomial
- [Volatility](vol.md) — implied vol, surfaces, smiles
- [Exceptions](exceptions.md) — error types you may want to catch

## Quick snippets

### Flat convenience API

```python
from option_pricing import MarketData, OptionSpec, PricingInputs, OptionType, bs_price

md = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
spec = OptionSpec(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
inp = PricingInputs(spec=spec, market=md, sigma=0.2)

price = bs_price(inp)
```

### Curves-first API

```python
from option_pricing import MarketData, OptionSpec, PricingInputs, OptionType, bs_price_from_ctx

md = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
spec = OptionSpec(option_type=OptionType.CALL, strike=100.0, expiry=1.0)
inp = PricingInputs(spec=spec, market=md, sigma=0.2)

ctx = inp.ctx
tau = inp.tau

price = bs_price_from_ctx(ctx=ctx, option=spec, sigma=inp.sigma, tau=tau)
```

## Notes

- Times are expressed in years.
- Internally, pricers work with `tau` (time-to-expiry). If you work with (t, T) in user code, convert using tau = T - t.