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
    bs_price_call,
    mc_price_call,
    binom_price_call,
)

p = PricingInputs(
    market=MarketData(spot=100.0, rate=0.05),
    spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0),  # T (years)
    sigma=0.20,
    t=0.0,
)

bs = bs_price_call(p)
mc, se = mc_price_call(p, n_paths=100_000, seed=0)
bt = binom_price_call(p, n_steps=400)

print("BS:", bs)
print("MC:", mc, "SE:", se)
print("Binomial:", bt)
```

### Puts + Greeks

```python
from option_pricing import mc_price_put, binom_price_put, bs_call_greeks

put_mc, put_se = mc_price_put(p, n_paths=200_000, seed=1)
put_bt = binom_price_put(p, n_steps=400)

greeks = bs_call_greeks(p)  # dict: price, delta, gamma, vega, theta
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
  OptionType, OptionSpec, MarketData, PricingInputs,
  # Pricing entrypoints
  bs_price_call, bs_call_greeks,
  mc_price_call, mc_price_put,
  binom_price_call, binom_price_put,
)
```

If you need internals, import from submodules under `option_pricing.models`, `option_pricing.pricers`, etc.

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

(You can host these later with MkDocs or GitHub Pages if you want.)

---

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.