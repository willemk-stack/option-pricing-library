# option_pricing

Typed Python library for **vanilla option pricing, implied-volatility workflows, volatility surfaces, local-vol diagnostics, and finite-difference PDE pricing**.

It supports **analytic Black–Scholes(-Merton)** pricing, **CRR binomial trees** for European and American vanilla options, **Monte Carlo under GBM**, and more advanced **surface / local-vol / PDE** workflows — with both **instruments-based** and **flat-input** APIs.

[![Tests](https://github.com/willemk-stack/option-pricing-library/actions/workflows/tests.yaml/badge.svg)](https://github.com/willemk-stack/option-pricing-library/actions/workflows/tests.yaml)
[![Codecov](https://codecov.io/gh/willemk-stack/option-pricing-library/branch/main/graph/badge.svg)](https://codecov.io/gh/willemk-stack/option-pricing-library)
[![Docs](https://github.com/willemk-stack/option-pricing-library/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/willemk-stack/option-pricing-library/actions/workflows/deploy-docs.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

> **Note:** This file is **auto-generated** from `README.template.md` and snippets in `examples/`.  
> To edit, modify the template or example sources, then run:
>
> ```bash
> python scripts/render_readme.py
> ```

---

## Why this repo

This project is designed as a **typed, test-backed quant library** rather than a notebook-only collection of pricing code.

Core strengths:

- **Multiple pricing engines** for vanilla options: Black–Scholes, Monte Carlo, and CRR binomial trees
- **Volatility tooling**: implied-vol inversion, smiles, surfaces, no-arbitrage checks, SVI fitting and repair
- **Advanced numerics**: local-vol extraction, diagnostics, and finite-difference PDE pricing
- **Validation-first approach**: analytic baselines, convergence checks, stress tests, and CI-executed notebooks
- **Layered API design**: simple flat-input workflows, instrument-based workflows, and curves-first pricing contexts

Docs: [📘 willemk-stack.github.io/option-pricing-library](https://willemk-stack.github.io/option-pricing-library)  
API Reference: [📘 /api](https://willemk-stack.github.io/option-pricing-library/api/)

---

## Best places to start

### Flagship capstones

#### Capstone 1 — Vol surfaces, no-arbitrage diagnostics, and SVI

**quotes / smiles → surface construction → arbitrage diagnostics → SVI fitting / repair**

Start here:

- **User guides:** `docs/user_guides/vol_surface.md`, `docs/user_guides/svi.md`, `docs/user_guides/svi_repair.md`
- **Representative tests:** `tests/test_surface_svi_and_localvol.py`, `tests/test_svi.py`, `tests/test_svi_repair.py`, `tests/test_arbitrage.py`

#### Capstone 2 — Local Vol + PDE Pricing + Diagnostics

**surface quotes → implied surface → local-vol diagnostics → PDE pricing → convergence / repricing checks**

Why it stands out:

- **Local volatility is treated as a numerical engineering problem**, not just a formula.
- **Diagnostics are first-class outputs**: invalid masks, denominator failures, unstable regions.
- **The PDE stack is validated**, not just implemented: analytic baselines, convergence checks, and discontinuous-payoff remedies.
- **The workflow is end-to-end**: surface construction, local-vol extraction, pricing, and validation are connected in one pipeline.

Validated by:

- constant-vol recovery tests for Dupire local volatility
- vanilla PDE checks against Black–Scholes baselines
- digital-option convergence and remedy tests
- QuantLib comparison tests for local-vol digital pricing

Start here:

- **Showcase demo:** `demos/06_vol_surfaces_localvol_pde.ipynb`
- **PDE-focused demo:** `demos/05_pde_pricing_and_diagnostics.ipynb`
- **User guides:** `docs/user_guides/local_vol.md`, `docs/user_guides/pde_pricing.md`
- **Representative tests:**
  - `tests/test_dupire_constant_vol.py`
  - `tests/test_localvol_pde_vanilla_vs_bs.py`
  - `tests/test_localvol_digital_vs_quantlib.py`
  - `tests/test_localvol_digital_convergence_sweep.py`
  - `tests/test_convergence_remedies_digital.py`

**What this demonstrates:** end-to-end surface construction, local-vol extraction, PDE pricing, and numerical validation in a single workflow.

---

## Installation

Core library:

```bash
pip install -e .
```

Development (tests, notebooks, linting, docs):

```bash
pip install -e ".[dev,docs]"
```

Python requirement:

- **Python 3.12+**

---

## API styles

The repo supports three complementary ways to work:

- **Flat-input API** for quick experiments and tutorial-style usage (`PricingInputs`)
- **Instrument-based API** to separate contracts from pricing engines (`VanillaOption`, `ExerciseStyle`)
- **Curves-first API** for discount / forward curve workflows (`PricingContext`)

### Recommended API path

- **Recommended API**: instrument-based workflow (`VanillaOption` + instrument pricers). This is the intended public entry point for most users and keeps contracts separate from pricing methods.
- **Convenience API**: flat-input workflow (`PricingInputs`). Use this for compact tutorials, tests, and quick checks.
- **Advanced API**: curves-first + surface / PDE workflows (`PricingContext`, vol, diagnostics). Use this when you need term structures, surfaces, or local-vol / PDE pipelines.

---

## Quick example (convenience API)

The convenient `PricingInputs` workflow is a good starting point for quick pricing checks and tutorials:

```python
{{ QUICKSTART }}
```

---

## Instrument workflow

Instruments cleanly separate *what you’re pricing* (the contract) from *how it’s priced* (the pricer and model).

```python
from option_pricing import (
    ExerciseStyle,
    MarketData,
    OptionType,
    VanillaOption,
    binom_price_instrument,
    bs_price_instrument,
    mc_price_instrument,
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

# Binomial tree (European / American)
binom_price_instrument(inst, market=market, sigma=sigma, n_steps=200)
```

Both APIs share the same pricing engines underneath; the flat-input versions simply wrap instruments internally.

---

## Curves-first example (`PricingContext`)

```python
{{ CURVES_FIRST }}
```

---

## Implied volatility example

```python
{{ IMPLIED_VOL }}
```

---

## What is implemented

### Pricing engines

- **Black–Scholes(-Merton)** price and Greeks
- **CRR binomial tree** for European and American vanilla options
- **Monte Carlo under GBM** with optional variance-reduction features
- **Finite-difference PDE pricing** for selected advanced workflows

### Volatility and diagnostics

- **BS implied-volatility inversion** with bracketing-based solvers
- **Smile** and **VolSurface** objects with interpolation support
- **Static no-arbitrage diagnostics** for surfaces
- **SVI fitting and repair** workflows
- **Local-vol extraction and diagnostics** from surfaces
- **Convergence and model-validation utilities**

---

## Project layout

| Layer | Purpose |
| --- | --- |
| **`instruments/`** | Contracts, payoffs, and exercise-style abstractions |
| **`market/`** | Spot, rates, dividends, curves, and pricing contexts |
| **`pricers/`** | Public pricing entry points for analytic, tree, Monte Carlo, and PDE workflows |
| **`models/`** | Model-specific internals such as Black–Scholes and local-vol components |
| **`vol/`** | Implied vol, smiles, surfaces, SVI, and local-vol extraction |
| **`numerics/`** | Root-finding, finite differences, tridiagonal solvers, and PDE building blocks |
| **`diagnostics/`** | Arbitrage checks, convergence studies, repricing, and validation helpers |
| **`viz/`** | Plotting helpers for surfaces, diagnostics, and reports |

---

## Demos and notebooks

| File | Topic |
| --- | --- |
| `demos/01_black_scholes_and_greeks.ipynb` | Analytic pricing + Greeks |
| `demos/02_monte_carlo_pricing_and_error.ipynb` | Monte Carlo pricing + standard errors |
| `demos/03_binomial_convergence.ipynb` | CRR tree convergence |
| `demos/04_implied_volatility.ipynb` | Implied-volatility inversion |
| `demos/05_pde_pricing_and_diagnostics.ipynb` | PDE pricing, stability, and convergence diagnostics |
| `demos/06_vol_surfaces_localvol_pde.ipynb` | End-to-end surface → local vol → PDE showcase |

---

## Validation and development

Development checks:

```bash
ruff check .
black --check .
pytest -q
mypy
```

The repo also includes:

- GitHub Actions for tests and docs
- README freshness checks
- CI notebook execution via `nbmake`

---

## Roadmap

See the MkDocs roadmap: [docs/roadmap.md](./docs/roadmap.md)

---

## License

Licensed under the **Apache-2.0** License. See [LICENSE](./LICENSE) for details.
