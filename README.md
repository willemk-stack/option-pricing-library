# option_pricing

Typed Python library for **European vanilla option pricing** (Black–Scholes, Monte Carlo under GBM, CRR binomial) + implied volatility helpers.

[![Tests](https://github.com/willemk-stack/option-pricing-library/actions/workflows/tests.yaml/badge.svg)](https://github.com/willemk-stack/option-pricing-library/actions/workflows/tests.yaml)
[![Docs](https://github.com/willemk-stack/option-pricing-library/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/willemk-stack/option-pricing-library/actions/workflows/deploy-docs.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

> **Note:** This `README.md` is **generated** from `README.template.md` and code snippets in `examples/`.
> Edit those sources, then run `python scripts/render_readme.py`.

## What’s included

- **Black–Scholes(-Merton)** closed-form pricing + analytic Greeks
- **Monte Carlo under GBM** price + standard error (optionally antithetic / control variates)
- **CRR binomial tree** pricer (European)
- **Implied volatility (BS inversion)** with robust bracketing/root-finding utilities
- **Volatility objects**: Smile, VolSurface (surface/smile containers + interpolation)
- Two market APIs: `MarketData` + `PricingInputs` (flat convenience) and `PricingContext` + curves (curves-first).

Docs: https://willemk-stack.github.io/option-pricing-library/  
API:  https://willemk-stack.github.io/option-pricing-library/api/

---

## Install

core:
```bash
pip install -e .
```

dev / notebooks:
```bash
pip install -e ".[dev]"
```

---

## Quick example (runnable)

```python
from option_pricing import (
    MarketData,
    OptionSpec,
    OptionType,
    PricingInputs,
    binom_price,
    bs_greeks,
    bs_price,
    mc_price,
)
from option_pricing.config import MCConfig, RandomConfig

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
p = PricingInputs(spec=spec, market=market, sigma=0.20, t=0.0)

print("BS:", bs_price(p))
print("Greeks:", bs_greeks(p))

cfg_mc = MCConfig(n_paths=200_000, antithetic=True, random=RandomConfig(seed=0))
price_mc, se = mc_price(p, cfg=cfg_mc)
print("MC:", price_mc, "(SE=", se, ")")

print("CRR:", binom_price(p, n_steps=500))
```

## Curves-first example (PricingContext)

```python   
from option_pricing import (
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    OptionType,
    PricingContext,
    binom_price_from_ctx,
    bs_greeks_from_ctx,
    bs_price_from_ctx,
    mc_price_from_ctx,
)
from option_pricing.config import MCConfig, RandomConfig

spot = 100.0
r = 0.05
q = 0.00
sigma = 0.20
tau = 1.0
K = 100.0

discount = FlatDiscountCurve(r)
forward = FlatCarryForwardCurve(spot=spot, r=r, q=q)
ctx = PricingContext(spot=spot, discount=discount, forward=forward)

print(
    "BS:",
    bs_price_from_ctx(
        kind=OptionType.CALL, strike=K, sigma=sigma, tau=tau, ctx=ctx
    ),
)
print(
    "Greeks:",
    bs_greeks_from_ctx(
        kind=OptionType.CALL, strike=K, sigma=sigma, tau=tau, ctx=ctx
    ),
)

cfg_mc = MCConfig(n_paths=200_000, antithetic=True, random=RandomConfig(seed=0))
price_mc, se = mc_price_from_ctx(
    kind=OptionType.CALL, strike=K, sigma=sigma, tau=tau, ctx=ctx, cfg=cfg_mc
)
print("MC:", price_mc, "(SE=", se, ")")

print(
    "CRR:",
    binom_price_from_ctx(
        kind=OptionType.CALL, strike=K, sigma=sigma, tau=tau, ctx=ctx, n_steps=500
    ),
)
```


---

### Implied volatility (BS inversion)

```python
from option_pricing import (
    ImpliedVolConfig,
    MarketData,
    OptionSpec,
    OptionType,
    RootMethod,
    implied_vol_bs_result,
)

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

cfg = ImpliedVolConfig(
    root_method=RootMethod.BRACKETED_NEWTON, sigma_lo=1e-8, sigma_hi=5.0
)

res = implied_vol_bs_result(mkt_price=10.0, spec=spec, market=market, cfg=cfg)

rr = res.root_result
print(f"IV: {res.vol:.6f}")
print(f"Converged: {rr.converged}  iters={rr.iterations}  method={rr.method}")
print(f"f(root)={rr.f_at_root:.3e}  bracket={rr.bracket}  bounds={res.bounds}")
```

---

## Notebooks / demos
- `demos/01_black_scholes_and_greeks.ipynb`
- `demos/02_monte_carlo_pricing_and_error.ipynb`
- `demos/03_binomial_convergence.ipynb`
- `demos/04_implied_volatility.ipynb`
- `demos/05_vol_surface_and_noarb.ipynb`

---

## Development
```bash
ruff check .
black --check .
pytest -q
mypy
```

---


## License
This project is licensed under the  Apache-2.0 — see the [LICENSE](./LICENSE) file for details.
