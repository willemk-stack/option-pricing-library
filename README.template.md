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
{{ QUICKSTART }}
```

## Curves-first example (PricingContext)

```python
{{ CURVES_FIRST }}
```

---

### Implied volatility (BS inversion)

```python
{{ IMPLIED_VOL }}
```

---

## Notebooks / demos
- `demos/01_black_scholes_and_greeks.ipynb`
- `demos/02_monte_carlo_pricing_and_error.ipynb`
- `demos/03_binomial_convergence.ipynb`
- `demos/04_implied_volatility.ipynb`
- `demos/05_vol_surface_and_noarb.ipynb`

---

## Roadmap

- Published roadmap (MkDocs): `docs/roadmap.md`
- Root pointer: `ROADMAP.md`

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
