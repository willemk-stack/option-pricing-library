# Monte Carlo

The Monte Carlo pricer simulates under risk-neutral GBM and returns a
`MonteCarloResult` object with:

- `price`: the discounted price estimate
- `stderr`: the estimated standard error of that estimate
- `n_paths` / `effective_n`: raw and effective sample counts
- optional metadata such as `antithetic` and path-step details

## Basic usage

```python
from option_pricing import mc_price
from option_pricing.monte_carlo import MCConfig, MonteCarloResult, RandomConfig

cfg = MCConfig(
    n_paths=100_000,
    antithetic=False,
    random=RandomConfig(seed=0),
)

result = mc_price(p, cfg=cfg)
assert isinstance(result, MonteCarloResult)

print(result.price, result.stderr)
```

## Antithetic variates

You can reduce variance with paired `Z` and `-Z` samples:

```python
cfg = MCConfig(
    n_paths=100_000,
    antithetic=True,
    random=RandomConfig(seed=0),
)

result = mc_price(p, cfg=cfg)
```

When `antithetic=True`, `n_paths` must be even.

## Reproducibility

There are two common ways to control randomness.

### 1) Use `RandomConfig`

```python
from option_pricing.monte_carlo import MCConfig, RandomConfig

cfg = MCConfig(n_paths=50_000, random=RandomConfig(seed=123, rng_type="pcg64"))
result = mc_price(p, cfg=cfg)
```

Supported NumPy-backed RNG types are:

- `"pcg64"`
- `"mt19937"`

`"sobol"` is reserved in the config type but is not implemented by NumPy in this pricer, so it raises `NotImplementedError` unless you provide your own explicit generator path in a future extension.

### 2) Pass an explicit generator

```python
import numpy as np
from option_pricing.monte_carlo import MCConfig

rng = np.random.default_rng(123)
cfg = MCConfig(n_paths=50_000, rng=rng)

result1 = mc_price(p, cfg=cfg)
result2 = mc_price(p, cfg=cfg)  # same generator, now advanced
```

## Standard error scaling

A quick sanity check is that the standard error should shrink roughly like `1 / sqrt(N)`.

```python
from option_pricing.monte_carlo import MCConfig, RandomConfig

se1 = mc_price(p, cfg=MCConfig(n_paths=25_000, random=RandomConfig(seed=0))).stderr
se2 = mc_price(p, cfg=MCConfig(n_paths=100_000, random=RandomConfig(seed=0))).stderr

print(se1 / se2)
```

A ratio near `2` is the usual rule-of-thumb expectation here.

## Curves-first and instrument-based workflows

The same engine is also available through:

- `mc_price_from_ctx(...)`
- `mc_price_instrument(...)`
- `mc_price_path_payoff_from_ctx(...)`

Example:

```python
from option_pricing import mc_price_instrument

result = mc_price_instrument(inst, market=market, sigma=0.20, cfg=cfg)
print(result.price, result.stderr)
```

For direct path payoffs, pass a callable that accepts `(paths, *, times=None)`:

```python
from option_pricing import mc_price_path_payoff_from_ctx

result = mc_price_path_payoff_from_ctx(
    ctx=ctx,
    payoff=lambda paths, *, times=None: paths[:, -1],
    sigma=0.20,
    tau=1.0,
    n_steps=32,
    cfg=cfg,
)
```

## Notes

- Terminal-payoff and direct path-payoff GBM workflows support European exercise.
- The returned `stderr` is a standard error, not a confidence interval. Build any interval from `result.price` and `result.stderr`.
- For method comparisons against Black-Scholes, see [Diagnostics](diagnostics.md).
