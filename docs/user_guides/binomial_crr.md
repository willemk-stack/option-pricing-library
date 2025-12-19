# Binomial (CRR)

The Cox–Ross–Rubinstein (CRR) binomial model approximates the lognormal diffusion with a recombining tree.

## Price a call / put

```python
from option_pricing import PricingInputs, OptionSpec, OptionType, binom_price

call = binom_price(p, n_steps=400)

p_put = PricingInputs(
    market=p.market,
    spec=OptionSpec(kind=OptionType.PUT, strike=p.K, expiry=p.spec.expiry),
    sigma=p.sigma,
    t=p.t,
)
put = binom_price(p_put, n_steps=400)
```

## Convergence intuition

As `n_steps` increases, CRR prices typically converge toward Black–Scholes for European options.

A simple experiment:

```python
from option_pricing import bs_price, binom_price

bs = bs_price(p)
for n in [25, 50, 100, 200, 400]:
    c = binom_price(p, n_steps=n)
    print(n, c - bs)
```

Tips:
- Very low `n_steps` can oscillate around BS.
- For “production-ish” accuracy, you often need a few hundred steps (depends on parameters).
