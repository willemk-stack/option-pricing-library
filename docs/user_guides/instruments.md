# Instruments

The library supports a newer instrument-based workflow in addition to the older `PricingInputs` API.
This is the recommended public entry point for most users.

The idea is simple:

- an **instrument** describes *what* is being priced
- a **pricer** describes *how* it is priced

For vanilla options, the main instrument type is `VanillaOption`.

## Why use instruments?

The instrument API becomes especially useful when:

- you want exercise style to be explicit
- you want to price the same contract with several methods
- you want to separate contracts from market inputs

## Create a vanilla option instrument

```python
from option_pricing import ExerciseStyle, OptionType, VanillaOption

inst = VanillaOption(
    expiry=1.0,  # time-to-expiry, not absolute clock time
    strike=100.0,
    kind=OptionType.CALL,
    exercise=ExerciseStyle.EUROPEAN,
)
```

Important difference from `PricingInputs`:

- `VanillaOption.expiry` means **time-to-expiry**
- `OptionSpec.expiry` in the legacy `PricingInputs` API means the absolute expiry `T`

## Price the same instrument with multiple pricers

```python
from option_pricing import (
    MarketData,
    bs_price_instrument,
    mc_price_instrument,
    binom_price_instrument,
)
from option_pricing.config import MCConfig, RandomConfig

market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.00)
sigma = 0.20

bs = bs_price_instrument(inst, market=market, sigma=sigma)
mc, se = mc_price_instrument(
    inst,
    market=market,
    sigma=sigma,
    cfg=MCConfig(n_paths=100_000, random=RandomConfig(seed=0)),
)
crr = binom_price_instrument(inst, market=market, sigma=sigma, n_steps=400)
```

## American exercise with the binomial tree

Black-Scholes and the terminal-only Monte Carlo pricer only support European exercise.
The CRR tree also supports American exercise.

```python
american_put = VanillaOption(
    expiry=1.0,
    strike=100.0,
    kind=OptionType.PUT,
    exercise=ExerciseStyle.AMERICAN,
)

price = binom_price_instrument(
    american_put,
    market=market,
    sigma=0.20,
    n_steps=400,
    method="tree",
)
```

For American exercise, `method="tree"` is required.

## Use instruments with curves-first inputs

```python
ctx = market.to_context()

bs_ctx = bs_price_instrument(inst, market=ctx, sigma=sigma)
mc_ctx, se_ctx = mc_price_instrument(inst, market=ctx, sigma=sigma)
crr_ctx = binom_price_instrument(inst, market=ctx, sigma=sigma, n_steps=400)
```

These wrappers accept either `MarketData` or `PricingContext`.

## Convert from legacy `PricingInputs`

If you already have a `PricingInputs` object, use the bridge helper:

```python
from option_pricing.instruments.factory import from_pricing_inputs

inst2 = from_pricing_inputs(p)
```

That converts `p.spec` together with `p.tau` into a `VanillaOption`.

## Related guides

- [Quickstart](quickstart.md)
- [Market APIs](market_api.md)
- [Binomial CRR](binomial_crr.md)
