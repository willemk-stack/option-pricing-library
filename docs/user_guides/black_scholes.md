# Black–Scholes

The Black–Scholes model provides a closed-form price for European options under lognormal dynamics (no dividends in the current implementation).

## Price a call

```python
from option_pricing import bs_price_call

price = bs_price_call(p)
```

## Price a put (model module)

The put pricer isn’t currently re-exported at the top level. Import it directly from the model:

```python
from option_pricing.models.bs import bs_put_from_inputs

put = bs_put_from_inputs(p)
```

## Call Greeks (analytic)

```python
from option_pricing import bs_call_greeks

g = bs_call_greeks(p)
print(g["delta"], g["gamma"], g["vega"], g["theta"])
```

Returned keys:
- `price`, `delta`, `gamma`, `vega`, `theta`

Notes:
- `theta` is returned as ∂Price/∂t (calendar time, holding `T` fixed), per year.

## Plot sweeps (optional)

The BS module includes an optional helper that plots price + Greeks across spot levels:

```python
from option_pricing.models.bs import sweep_x

sweep_x(K=100, r=0.01, sigma=0.2, T=1.0, method="analytic")
```

This requires `matplotlib`.
