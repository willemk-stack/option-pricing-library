# Market APIs

The library supports two related ways to describe markets:

1. **Flat convenience API** using `MarketData` and `PricingInputs`
2. **Curves-first API** using `PricingContext`, `DiscountCurve`, and `ForwardCurve`

Both are valid. The choice is mostly about how explicit you want to be about term structure.

## Flat convenience API

This is the fastest path for examples, tests, and simple workflows.

```python
from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.02)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

p = PricingInputs(spec=spec, market=market, sigma=0.20, t=0.0)
print(bs_price(p))
```

`MarketData` also gives you small helpers:

```python
print(market.df(1.0))
print(market.fwd(1.0))
```

Important time convention:

- `MarketData.df(T, t=...)` and `MarketData.fwd(T, t=...)` work with absolute times
- `PricingInputs.tau` converts from `(t, T)` to time-to-expiry automatically

## Convert flat inputs into curves-first inputs

If you already have `MarketData`, the bridge is built in:

```python
ctx = market.to_context()
print(ctx.df(1.0))
print(ctx.fwd(1.0))
```

For flat rates and carry, the resulting `PricingContext` is exactly consistent with `MarketData`.

## Curves-first API

This is the better fit when you want the pricing code to consume discount and forward curves directly.

```python
from option_pricing import FlatDiscountCurve, FlatCarryForwardCurve, PricingContext

discount = FlatDiscountCurve(r=0.05)
forward = FlatCarryForwardCurve(spot=100.0, r=0.05, q=0.02)
ctx = PricingContext(spot=100.0, discount=discount, forward=forward)
```

With a `PricingContext`, you call the `*_from_ctx` pricers and pass `tau` directly:

```python
from option_pricing import OptionType, bs_price_from_ctx

price = bs_price_from_ctx(
    kind=OptionType.CALL,
    strike=100.0,
    sigma=0.20,
    tau=1.0,
    ctx=ctx,
)
```

## When to use which API

If you are following the recommended instrument-based workflow, you still choose between `MarketData` and `PricingContext` for the market inputs.

Use `MarketData` + `PricingInputs` when:

- you want the most compact examples
- rates and dividend yield are flat
- you are teaching, testing, or prototyping

Use `PricingContext` when:

- you already think in terms of discount factors and forwards
- you want to plug in term structures
- you want pricing code that does not depend on flat `(r, q)` assumptions at the interface level

## Matching results between the two APIs

For the same flat market assumptions, these should agree:

```python
from option_pricing import bs_price

price_inputs = bs_price(p)
price_ctx = bs_price_from_ctx(
    kind=p.spec.kind,
    strike=p.spec.strike,
    sigma=p.sigma,
    tau=p.tau,
    ctx=p.market.to_context(),
)
```

The same pattern also applies to:

- `bs_greeks_from_ctx`
- `mc_price_from_ctx`
- `binom_price_from_ctx`

## Related guides

- [Quickstart](quickstart.md)
- [Instruments](instruments.md)
- [Black-Scholes](black_scholes.md)
