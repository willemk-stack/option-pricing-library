# Option Pricing Library

A small, usage-first Python library for pricing **European options** under common textbook models:

- **Black–Scholes (BS)** closed-form pricing (and analytic call Greeks)
- **CRR binomial tree** pricing (European)
- **Monte Carlo (GBM)** pricing with a reported **standard error**

The public API is intentionally small: you build a `PricingInputs` bundle, then call top-level pricers from `option_pricing`.

## Quick example

```python
from option_pricing import (
    MarketData,
    OptionSpec,
    OptionType,
    PricingInputs,
    bs_price_call,
    binom_price_call,
    mc_price_call,
)

market = MarketData(spot=100.0, rate=0.05)
opt = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)  # expiry is an absolute time
p = PricingInputs(spec=opt, market=market, sigma=0.20, t=0.0)

bs = bs_price_call(p)
mc, se = mc_price_call(p, n_paths=50_000, seed=0)
binom = binom_price_call(p, n_steps=400)

print(f"BS:    {bs:.4f}")
print(f"MC:    {mc:.4f}  (SE={se:.4f})")
print(f"CRR:   {binom:.4f}")
```

Notes:
- Times are **floats** (typically in **years**). `t` is “now”; `expiry` is an **absolute** time `T`. Time-to-maturity is `tau = T - t`.
- Rates are assumed to be **continuously compounded**.
- The provided top-level entry points focus on **calls**. You can price puts via the model modules (see the [API](api.md)).

## Next pages

- [Installation](installation.md)
- [Public API](api.md)
- [User guides](user_guides/quickstart.md)
