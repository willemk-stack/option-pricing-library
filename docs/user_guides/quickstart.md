# Quickstart

This guide shows the “happy path” for using the library.

## 1) Build inputs

```python
from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs

market = MarketData(spot=100.0, rate=0.05)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

p = PricingInputs(
    spec=spec,
    market=market,
    sigma=0.20,
    t=0.0,
)
```

Conventions:
- Times (`t`, `expiry`) are floats (commonly in **years**).
- `expiry` is an **absolute** time `T`; time-to-maturity is `tau = T - t`.
- Rates are **continuously compounded**.

## 2) Price a call three ways

```python
from option_pricing import bs_price, binom_price, mc_price

bs = bs_price(p)
binom = binom_price(p, n_steps=400)
mc, se = mc_price(p, n_paths=50_000, seed=0)

print(bs, binom, mc, se)
```

## 3) Use the MC standard error

For a single run, a rough 95% confidence interval is:

```python
z = 1.96
ci_low = mc - z * se
ci_high = mc + z * se
```

A common quick sanity check is: *does the BS price land inside this interval?* For large enough `n_paths`, it usually should.

## Next guides

- [Black–Scholes](black_scholes.md)
- [Monte Carlo](monte_carlo.md)
- [Binomial (CRR)](binomial_crr.md)
- [Diagnostics in notebooks](diagnostics.md)
