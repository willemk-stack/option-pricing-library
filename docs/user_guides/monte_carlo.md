# Monte Carlo

The Monte Carlo pricer simulates terminal prices under risk-neutral GBM and returns:

- the discounted price estimate
- the estimated standard error of that estimate

## Basic usage

```python
from option_pricing import mc_price
from option_pricing.config import MCConfig, RandomConfig

cfg = MCConfig(
    n_paths=100_000,
    antithetic=False,
    random=RandomConfig(seed=0),
)

price, se = mc_price(p, cfg=cfg)
```

## Antithetic variates

You can reduce variance with paired `Z` and `-Z` samples:

```python
cfg = MCConfig(
    n_paths=100_000,
    antithetic=True,
    random=RandomConfig(seed=0),
)

price, se = mc_price(p, cfg=cfg)
```

When `antithetic=True`, `n_paths` must be even.

## Reproducibility

There are two common ways to control randomness.

### 1) Use `RandomConfig`

```python
from option_pricing.config import MCConfig, RandomConfig

cfg = MCConfig(n_paths=50_000, random=RandomConfig(seed=123, rng_type="pcg64"))
price, se = mc_price(p, cfg=cfg)
```

Supported NumPy-backed RNG types are:

- `"pcg64"`
- `"mt19937"`

`"sobol"` is reserved in the config type but is not implemented by NumPy in this pricer, so it raises `NotImplementedError` unless you provide your own explicit generator path in a future extension.

### 2) Pass an explicit generator

```python
import numpy as np
from option_pricing.config import MCConfig

rng = np.random.default_rng(123)
cfg = MCConfig(n_paths=50_000, rng=rng)

price1, se1 = mc_price(p, cfg=cfg)
price2, se2 = mc_price(p, cfg=cfg)  # same generator, now advanced
```

## Standard error scaling

A quick sanity check is that the standard error should shrink roughly like `1 / sqrt(N)`.

```python
from option_pricing.config import MCConfig, RandomConfig

_, se1 = mc_price(p, cfg=MCConfig(n_paths=25_000, random=RandomConfig(seed=0)))
_, se2 = mc_price(p, cfg=MCConfig(n_paths=100_000, random=RandomConfig(seed=0)))

print(se1 / se2)
```

A ratio near `2` is the usual rule-of-thumb expectation here.

## Curves-first and instrument-based workflows

The same engine is also available through:

- `mc_price_from_ctx(...)`
- `mc_price_instrument(...)`

Example:

```python
from option_pricing import mc_price_instrument

price, se = mc_price_instrument(inst, market=market, sigma=0.20, cfg=cfg)
```

## Notes

- This Monte Carlo implementation is for European, terminal-payoff products.
- It returns a standard error, not a confidence interval. You build that interval yourself from the returned `se`.
- For method comparisons against Black-Scholes, see [Diagnostics](diagnostics.md).
