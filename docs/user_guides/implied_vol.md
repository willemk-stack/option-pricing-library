# Implied volatility

This guide shows how to invert a market option price into a Black-Scholes implied volatility.

The main entry points are:

- `implied_vol_bs(...)` for the volatility only
- `implied_vol_bs_result(...)` for the volatility plus solver diagnostics

## Minimal example

```python
from option_pricing import (
    MarketData,
    OptionSpec,
    OptionType,
    PricingInputs,
    bs_price,
    implied_vol_bs,
    implied_vol_bs_result,
)

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.00)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

sigma_true = 0.20
p = PricingInputs(spec=spec, market=market, sigma=sigma_true, t=0.0)
mkt_price = bs_price(p)

iv = implied_vol_bs(mkt_price, spec, market)
res = implied_vol_bs_result(mkt_price, spec, market)

print(iv)
print(res.vol, res.root_result.converged, res.root_result.iterations)
```

## Configuration with `ImpliedVolConfig`

```python
from option_pricing import ImpliedVolConfig, RootMethod

cfg = ImpliedVolConfig(
    root_method=RootMethod.BRACKETED_NEWTON,
    sigma_lo=1e-8,
    sigma_hi=5.0,
)

res = implied_vol_bs_result(mkt_price, spec, market, cfg=cfg)
```

Useful knobs:

- `root_method`
- `sigma_lo`, `sigma_hi`
- `bounds_eps`
- `numerics.abs_tol`, `numerics.rel_tol`, `numerics.max_iter`

## Warm starts with `sigma0`

If you are solving a strip of nearby quotes, an explicit seed can reduce iterations.

```python
res = implied_vol_bs_result(
    mkt_price,
    spec,
    market,
    cfg=cfg,
    sigma0=0.25,
)
```

## Input price validation

The solver checks European no-arbitrage bounds before root-finding starts.
If the price is outside the valid range, it raises `InvalidOptionPriceError`.

```python
from option_pricing.exceptions import InvalidOptionPriceError

try:
    iv = implied_vol_bs(mkt_price, spec, market, cfg=cfg)
except InvalidOptionPriceError:
    iv = None
```

## Use curves-first markets too

The `market` argument can be either `MarketData` or `PricingContext`.

```python
ctx = market.to_context()
iv = implied_vol_bs(mkt_price, spec, ctx, cfg=cfg)
```

## Result diagnostics

`implied_vol_bs_result` returns an `ImpliedVolResult` with:

- `vol`
- `root_result`
- `mkt_price`
- `bounds`
- `tau`

That makes it easier to log iteration counts, final residuals, and the active price bounds.

## Notes

- This guide is for Black-Scholes implied volatility only.
- The option spec is still given as `OptionSpec(kind, strike, expiry)` with absolute expiry `T` and valuation time `t`.
- For building whole smiles and surfaces from implied vols, see [Volatility surface](vol_surface.md) and [SVI](svi.md).
