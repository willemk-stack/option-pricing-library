# Market APIs

The library supports **two layers** of market inputs:

1. **Convenience API (flat markets)**  
   `MarketData` + `PricingInputs` — quickest to use for tutorials, tests, and simple workflows.

2. **Core curves-first API**  
   `PricingContext` + `DiscountCurve` + `ForwardCurve` — the preferred representation for
   term-structure-aware pricing.

Internally, pricers operate on **time-to-expiry** `tau` (years).

---

## Convenience API: MarketData + PricingInputs

```python
from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.02)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

p = PricingInputs(spec=spec, market=market, sigma=0.20)
print(bs_price(p))
```

`MarketData` also provides small helpers:

- `df(T, t=0)` for `P(t,T)`
- `fwd(T, t=0)` (alias for `forward`) for `F(t,T)`
- `to_context()` to convert into the curves-first representation.

---

## Curves-first API

### Protocols

A **discount curve** returns a discount factor:

```python
df = curve.df(tau)     # P(0, tau)
```

A **forward curve** returns a forward price:

```python
F = curve.fwd(tau)     # F(0, tau)
```

Any object implementing these methods is compatible with the library's
`DiscountCurve` / `ForwardCurve` protocols.

### PricingContext

A `PricingContext` bundles spot + curves:

```python
from option_pricing import FlatDiscountCurve, FlatCarryForwardCurve, PricingContext

discount = FlatDiscountCurve(r=0.05)
forward  = FlatCarryForwardCurve(spot=100.0, r=0.05, q=0.02)
ctx = PricingContext(spot=100.0, discount=discount, forward=forward)

df = ctx.df(1.0)
F  = ctx.fwd(1.0)
```

### Pricer entry points

For curves-first workflows, use the `*_from_ctx` functions:

- `bs_price_from_ctx`, `bs_greeks_from_ctx`
- `mc_price_from_ctx`
- `binom_price_from_ctx`

These accept `(kind, strike, sigma, tau, ctx, ...)`.

---

## Why two APIs?

- The flat API makes examples/tests small and readable.
- The curves-first API makes it easy to plug in **non-flat term structures**
  (piecewise curves, bootstrapped discount factors, dividend curves, etc.)
  without changing pricer logic.
