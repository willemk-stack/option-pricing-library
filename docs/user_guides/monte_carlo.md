# Monte Carlo (GBM)

The Monte Carlo pricer simulates terminal stock prices under risk-neutral GBM and estimates the discounted payoff.

## Price a call (returns price + SE)

```python
from option_pricing import mc_price
from option_pricing.config import MCConfig, RandomConfig

cfg = MCConfig(n_paths=100_000, random=RandomConfig(seed=0))
price, se = mc_price(p, cfg=cfg)
```

- `price` is the Monte Carlo estimate
- `se` is the (discounted) standard error of the estimator


## Calls and puts

`mc_price` dispatches on `p.spec.kind`. To price a put, set `kind=OptionType.PUT` in your `OptionSpec` and call `mc_price` the same way.

## Reproducibility

You can control randomness in two ways:

- provide `cfg.random.seed=...` (via `RandomConfig`, convenient)
- provide a `rng=np.random.Generator` (useful if you want to reuse a generator)

```python
import numpy as np
from option_pricing import mc_price
from option_pricing.config import MCConfig

rng = np.random.default_rng(123)
cfg = MCConfig(n_paths=50_000, rng=rng)
price1, se1 = mc_price(p, cfg=cfg)
price2, se2 = mc_price(p, cfg=cfg)  # advances the same RNG
```

## SE scaling quick check

A useful “math-signal” check is that standard error scales roughly like:

- `SE(N) ≈ c / sqrt(N)`

So if you 4× your paths, SE should drop about 2×.

```python
from option_pricing.config import MCConfig, RandomConfig

p1, se1 = mc_price(p, cfg=MCConfig(n_paths=25_000, random=RandomConfig(seed=0)))
p2, se2 = mc_price(p, cfg=MCConfig(n_paths=100_000, random=RandomConfig(seed=0)))
print(se1 / se2)  # should be around 2
```

## Price a put

The library exposes a Monte Carlo put function too:

```python
from option_pricing import mc_price, MCConfig, RandomConfig

cfg = MCConfig(n_paths=200_000, antithetic=True, random=RandomConfig(seed=123))
price, err = mc_price(p, cfg=cfg) # Make sure to configure OptionType.PUT correctly in PricingInputs
```

`mc_price_put` uses the same config-driven interface as `mc_price`.
