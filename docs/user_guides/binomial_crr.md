# Binomial CRR

The Cox-Ross-Rubinstein model approximates lognormal dynamics with a recombining tree.
It is useful both as a teaching tool and as a practical lattice pricer.

## Price a European option

```python
from option_pricing import binom_price

call = binom_price(p, n_steps=400)
```

The option kind comes from `p.spec.kind`, so a put is just another `PricingInputs` object with `kind=OptionType.PUT`.

## Compare tree prices to Black-Scholes

```python
from option_pricing import bs_price, binom_price

bs = bs_price(p)
for n in [25, 50, 100, 200, 400, 800]:
    crr = binom_price(p, n_steps=n)
    print(n, crr, crr - bs)
```

Typical behavior:

- small trees can oscillate around the Black-Scholes value
- increasing `n_steps` usually improves European pricing accuracy

## Tree versus closed-form binomial summation

For European vanilla options, the pricer supports both:

```python
px_tree = binom_price(p, n_steps=400, method="tree")
px_closed = binom_price(p, n_steps=400, method="closed_form")
```

`method="closed_form"` is only for European pricing.

## American exercise

The clearest way to express American exercise is with the instrument API:

```python
from option_pricing import (
    ExerciseStyle,
    OptionType,
    VanillaOption,
    binom_price_instrument,
)

american_put = VanillaOption(
    expiry=1.0,
    strike=100.0,
    kind=OptionType.PUT,
    exercise=ExerciseStyle.AMERICAN,
)

price = binom_price_instrument(
    american_put,
    market=market,
    sigma=0.20,
    n_steps=400,
    method="tree",
)
```

You can also use the legacy wrapper with `american=True`:

```python
price = binom_price(p, n_steps=400, american=True, method="tree")
```

## Curves-first workflow

```python
from option_pricing import OptionType, binom_price_from_ctx

ctx = market.to_context()
price = binom_price_from_ctx(
    kind=OptionType.CALL,
    strike=100.0,
    sigma=0.20,
    tau=p.tau,
    ctx=ctx,
    n_steps=400,
)
```

## Notes

- American exercise requires `method="tree"`.
- The tree model uses the same flat or implied-average market information as the other pricers.
- For a PDE alternative, see [PDE pricing](pde_pricing.md).
