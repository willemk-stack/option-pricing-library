# option_pricing

Typed Python library for **European and American vanilla option pricing** — includes analytic, tree, and Monte Carlo methods — now with **instruments-based** and **legacy** APIs.

[![Tests](https://github.com/willemk-stack/option-pricing-library/actions/workflows/tests.yaml/badge.svg)](https://github.com/willemk-stack/option-pricing-library/actions/workflows/tests.yaml)
[![Docs](https://github.com/willemk-stack/option-pricing-library/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/willemk-stack/option-pricing-library/actions/workflows/deploy-docs.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

> **Note:** This file is **auto-generated** from `README.template.md` and snippets in `examples/`.  
> To edit, modify the template or example sources, then run:
>
> ```bash
> python scripts/render_readme.py
> ```

---

## Overview

The library provides a **typed**, **composable**, and **test-backed** toolkit for option pricing:

- **Black–Scholes(-Merton)** analytic formulas (price + Greeks)
- **Monte Carlo under GBM** (with optional antithetic / control variates)
- **CRR binomial tree** (European / American)
- **Implied volatility (BS inversion)** with robust bracketing solvers
- **Volatility structures**: `VolSmile`, `VolSurface` with interpolation and no-arbitrage checks
- **Market abstractions**:  
  - `MarketData`, `PricingInputs` for a flat, convenient API  
  - `PricingContext` for curves-first workflows
- **Instrument layer**: reusable definitions of contracts (e.g. `VanillaOption`) with structured payoffs

Docs: [📘 willemk-stack.github.io/option-pricing-library](https://willemk-stack.github.io/option-pricing-library)  
API Reference: [📘 /api](https://willemk-stack.github.io/option-pricing-library/api/)

---

## Installation

Core library:

```bash
pip install -e .
````

Development (tests, notebooks, linting):

```bash
pip install -e ".[dev]"
```

---

## Quick example (legacy API)

The original, convenient `PricingInputs` workflow still works:

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

---

## Instrument workflow (new)

Instruments cleanly separate *what you’re pricing* (the contract) from *how it’s priced* (the pricer and model).

```python
from option_pricing import (
    MarketData,
    OptionType,
    VanillaOption,
    ExerciseStyle,
    bs_price_instrument,
    mc_price_instrument,
    binom_price_instrument,
)

inst = VanillaOption(
    expiry=1.0,
    strike=100.0,
    kind=OptionType.CALL,
    exercise=ExerciseStyle.EUROPEAN,
)

market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
sigma = 0.2

# Analytic (BS)
bs_price_instrument(inst, market=market, sigma=sigma)

# Monte Carlo
mc_price_instrument(inst, market=market, sigma=sigma)

# Binomial tree (European/American)
binom_price_instrument(inst, market=market, sigma=sigma, n_steps=200)
```

Both APIs share the same pricing engines underneath; the `PricingInputs` versions simply wrap instruments internally.

---

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

## Module structure

| Layer              | Purpose                                                   | Example modules                        |
| ------------------ | --------------------------------------------------------- | -------------------------------------- |
| **`instruments/`** | Defines *what* is being priced (contracts + payoffs).     | `base.py`, `vanilla.py`, `factory.py`  |
| **`models/`**      | Defines *how* the underlying evolves (e.g., GBM, Heston). | `black_scholes.py`, `gbm.py`           |
| **`pricers/`**     | Numerical engines (analytic, tree, MC).                   | `black_scholes.py`, `tree.py`, `mc.py` |
| **`market/`**      | Market data (spot, rates, dividends).                     | `market_data.py`, `pricing_context.py` |
| **`vol/`**         | Volatility structures and interpolation.                  | `smile.py`, `surface.py`               |
| **`diagnostics/`** | No-arbitrage, convergence, stability checks.              | `noarb.py`, `monte_carlo_error.py`     |

---

## Notebooks / demos

| File                                           | Topic                        |
| ---------------------------------------------- | ---------------------------- |
| `demos/01_black_scholes_and_greeks.ipynb`      | Analytic pricing + Greeks    |
| `demos/02_monte_carlo_pricing_and_error.ipynb` | Monte Carlo pricing + SE     |
| `demos/03_binomial_convergence.ipynb`          | Tree convergence             |
| `demos/04_implied_volatility.ipynb`            | Implied volatility inversion |
| `demos/05_vol_surface_and_noarb.ipynb`         | Vol surfaces + no-arb checks |

---

## Roadmap

See the MkDocs roadmap: [docs/roadmap.md](./docs/roadmap.md)

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

Licensed under the **Apache-2.0** License. See [LICENSE](./LICENSE) for details.

```

---

### Summary of changes (TODO: REMOVE BEFORE MERGING)
✅ **New section** — “Instrument workflow (new)”  
✅ **Clarified structure** — added module overview table showing where `instruments` fits  
✅ **Retained template markers** (`from option_pricing import (
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

print("CRR:", binom_price(p, n_steps=500))`, `from option_pricing import (
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
)`, `from option_pricing import (
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
print(f"f(root)={rr.f_at_root:.3e}  bracket={rr.bracket}  bounds={res.bounds}")`) for automated rendering  
✅ **Consistent heading hierarchy and tone** with the rest of your docs  

This makes it clear to users that:
- the *legacy* API still works,
- the *instrument* API is now first-class,
- and the folder structure reflects the conceptual layers of your design.
