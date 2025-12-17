# Monte Carlo (GBM)

The Monte Carlo pricer simulates terminal stock prices under risk-neutral GBM and estimates the discounted payoff.

## Price a call (returns price + SE)

```python
from option_pricing import mc_price_call

price, se = mc_price_call(p, n_paths=100_000, seed=0)
```

- `price` is the Monte Carlo estimate
- `se` is the (discounted) standard error of the estimator

## Reproducibility

You can control randomness in two ways:

- provide `seed=...` (convenient)
- provide a `rng=np.random.Generator` (useful if you want to reuse a generator)

```python
import numpy as np
from option_pricing import mc_price_call

rng = np.random.default_rng(123)
price1, se1 = mc_price_call(p, n_paths=50_000, rng=rng)
price2, se2 = mc_price_call(p, n_paths=50_000, rng=rng)  # advances the same RNG
```

## SE scaling quick check

A useful “math-signal” check is that standard error scales roughly like:

- `SE(N) ≈ c / sqrt(N)`

So if you 4× your paths, SE should drop about 2×.

```python
p1, se1 = mc_price_call(p, n_paths=25_000, seed=0)
p2, se2 = mc_price_call(p, n_paths=100_000, seed=0)
print(se1 / se2)  # should be around 2
```

## Price a put

The library exposes a Monte Carlo put function too:

```python
from option_pricing import mc_price_put

put, se = mc_price_put(p, n_paths=100_000)
```

(If you want the same `seed` / `rng` controls for puts, consider mirroring the `mc_price_call` signature.)
