# Quickstart

This guide shows the fastest path from inputs to prices.
It uses the convenience `PricingInputs` workflow because it is compact and easy to read.
For the recommended public API, start with the [Instruments](instruments.md) guide.

## 1) Build inputs

```python
from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.00)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

p = PricingInputs(
    spec=spec,
    market=market,
    sigma=0.20,
    t=0.0,
)
```

A few conventions matter right away:

- In `PricingInputs`, `OptionSpec.expiry` is the absolute expiry time `T`
- `t` is the valuation time
- `tau = T - t` is computed for you via `p.tau`
- with the default `t=0`, the numeric value of `expiry` happens to equal `tau`
- rates are continuously compounded

## 2) Price the same option three ways

```python
from option_pricing import bs_price, binom_price, mc_price
from option_pricing.config import MCConfig, RandomConfig

bs = bs_price(p)
binom = binom_price(p, n_steps=400)
mc, se = mc_price(
    p,
    cfg=MCConfig(n_paths=50_000, random=RandomConfig(seed=0)),
)

print(f"BS:    {bs:.6f}")
print(f"CRR:   {binom:.6f}")
print(f"MC:    {mc:.6f}  (SE={se:.6f})")
```

Typical behavior:

- Black-Scholes is the closed-form benchmark
- CRR approaches the Black-Scholes value as `n_steps` grows
- Monte Carlo returns both a price estimate and a standard error

## 3) Turn the Monte Carlo error into a rough confidence interval

```python
z = 1.96
ci_low = mc - z * se
ci_high = mc + z * se

print(ci_low, ci_high)
```

A common sanity check is whether the analytic Black-Scholes price falls inside this interval.

## 4) Price a put instead of a call

The pricing function does not change.
You only change the option specification:

```python
p_put = PricingInputs(
    spec=OptionSpec(kind=OptionType.PUT, strike=100.0, expiry=1.0),
    market=market,
    sigma=0.20,
    t=0.0,
)

put_bs = bs_price(p_put)
```

## Where to go next

- [Market APIs](market_api.md) for the difference between flat and curves-first inputs
- [Instruments](instruments.md) for the newer instrument-based workflow
- [Black-Scholes](black_scholes.md), [Monte Carlo](monte_carlo.md), and [Binomial CRR](binomial_crr.md) for method-specific details
