# Black–Scholes

The Black–Scholes model provides a closed-form price for European options under lognormal dynamics (no dividends in the current implementation).

## Price a call

```python
from option_pricing import bs_price

price = bs_price(p)  # uses p.spec.kind (CALL/PUT)
```

## Price a put

You don’t need a separate import for puts. Set `kind=OptionType.PUT` in your `OptionSpec` and call `bs_price`:

```python
from option_pricing import PricingInputs, OptionType, OptionSpec, bs_price

p_put = PricingInputs(
    market=p.market,
    spec=OptionSpec(kind=OptionType.PUT, strike=p.K, expiry=p.spec.expiry),
    sigma=p.sigma,
    t=p.t,
)

put = bs_price(p_put)
```

## Greeks (analytic)

```python
from option_pricing import bs_greeks

g = bs_greeks(p)
print(g["delta"], g["gamma"], g["vega"], g["theta"])
```

## Plot sweeps (optional)

The BS module includes an optional helper that plots price + Greeks across spot levels:

```python
from option_pricing.models.bs import sweep_x

sweep_x(K=100, r=0.01, sigma=0.2, T=1.0, method="analytic")
```

This requires `matplotlib`.
