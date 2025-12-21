# option_pricing

A small, typed Python library for pricing **European** vanilla options with:

- **Black–Scholes (closed-form)** (calls + puts)
- **Monte Carlo under GBM** (price + standard error)
- **CRR binomial tree** (calls + puts; converges to BS as steps ↑)

It’s set up as a normal `src/` Python package and includes pytest “math-signal” tests (parity, bounds, monotonicity, MC SE scaling, binomial→BS convergence).

---

## Quickstart

```python
from option_pricing import (
    MarketData,
    OptionSpec,
    OptionType,
    PricingInputs,
    bs_price,
    bs_greeks,
    mc_price,
    binom_price,
)

p = PricingInputs(
    market=MarketData(spot=100.0, rate=0.05),
    spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0),  # expiry T (years)
    sigma=0.20,
    t=0.0,
)

# Prices
bs = bs_price(p)
mc, se = mc_price(p, n_paths=100_000, seed=0)
bt = binom_price(p, n_steps=400)

print("Black–Scholes:", bs)
print("Monte Carlo:  ", mc, "(SE=", se, ")")
print("Binomial:     ", bt)

# Greeks (analytic, Black–Scholes)
g = bs_greeks(p)  # dict: price, delta, gamma, vega, theta
print("Delta:", g["delta"])
```

---

## Installation

### Editable install (recommended for development)

```bash
python -m pip install -U pip
pip install -e .
```

### Optional extras

Plotting is **optional**:

```bash
pip install -e ".[plot]"
```

Notebooks / diagnostics extras:

```bash
pip install -e ".[notebooks]"
```

---

## API

The supported “public” API is exposed from the package root:

```python
from option_pricing import (
  # Types
  OptionType,
  OptionSpec,
  MarketData,
  PricingInputs,

  # Pricers
  bs_price,
  bs_greeks,
  mc_price,
  binom_price,

  # Advanced MC building blocks
  ControlVariate,
  McGBMModel,
)
```

Anything else should be treated as internal and may change without notice (e.g. `option_pricing.models`, `option_pricing.pricers`, etc.).

---

## Development

### Lint / format / type-check

```bash
pre-commit install
pre-commit run --all-files
```

### Tests

```bash
pytest -q
```

### Docs

There is a `docs/` folder with:
- `docs/index.md`
- `docs/installation.md`
- `docs/api.md`
- `docs/user_guides/*`


---

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
