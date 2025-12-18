# Binomial (CRR)

The Cox–Ross–Rubinstein (CRR) binomial model approximates the lognormal diffusion with a recombining tree.

## Price a call / put

```python
from option_pricing import binom_price_call, binom_price_put

call = binom_price_call(p, n_steps=400)
put = binom_price_put(p, n_steps=400)
```

## Convergence intuition

As `n_steps` increases, CRR prices typically converge toward Black–Scholes for European options.

A simple experiment:

```python
from option_pricing import bs_price_call, binom_price_call

bs = bs_price_call(p)
for n in [25, 50, 100, 200, 400]:
    c = binom_price_call(p, n_steps=n)
    print(n, c - bs)
```

Tips:
- Very low `n_steps` can oscillate around BS.
- For “production-ish” accuracy, you often need a few hundred steps (depends on parameters).
