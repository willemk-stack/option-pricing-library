# Diagnostics

The `option_pricing.diagnostics` modules are meant for notebook-style analysis.
They are especially useful when you want to answer questions like:

- *Is my Monte Carlo estimator behaving as expected?*
- *How sensitive are the Greeks across spot?*
- *Is my volatility surface clean enough to use?*
- *Which SVI repair recipe worked best?*

Some helpers depend on `pandas` and `matplotlib`, so the notebook/dev extras are useful.

```bash
pip install -e ".[dev]"
```

## 1) Compare Monte Carlo against Black-Scholes

```python
from dataclasses import replace

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs
from option_pricing.diagnostics.mc_vs_bs.tables import compare_table, convergence_table

base = PricingInputs(
    spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0),
    market=MarketData(spot=100.0, rate=0.05),
    sigma=0.2,
    t=0.0,
)

cases = [
    ("ATM", base),
    ("ITM", replace(base, market=replace(base.market, spot=120.0))),
    ("High vol", replace(base, sigma=0.8)),
]

cmp_df = compare_table(cases, n_paths=50_000, seed=0)
conv_df = convergence_table(base, n_paths_list=[1_000, 5_000, 10_000, 50_000], seed=0)
```

## 2) Sweep spot and inspect Greeks

```python
import numpy as np
from option_pricing.diagnostics.greeks.sweep import sweep_spot_greeks

spots = np.linspace(60.0, 140.0, 81)
out = sweep_spot_greeks(base, spots, method="analytic")

print(out.delta[:5])
print(out.gamma[:5])
```

## 3) Run surface diagnostics on a volatility surface

```python
from option_pricing.diagnostics.vol_surface import run_surface_diagnostics

report = run_surface_diagnostics(
    surface,
    forward=forward,
    df=df,
    include_svi=True,
    include_domain=True,
)

print(report.meta)
print(report.tables["noarb_smiles"].head())
print(report.to_json()[:200])
```

The returned object bundles notebook-friendly `pandas.DataFrame` tables plus optional arrays.

## 4) Try several SVI build recipes with fallback

```python
from option_pricing.diagnostics.vol_surface.recipes import (
    build_svi_surface_with_fallback,
    default_svi_repair_candidates,
)

candidates = default_svi_repair_candidates()
surface_svi, label, attempts = build_svi_surface_with_fallback(
    rows,
    forward=forward,
    candidates=candidates,
    fallback_surface=surface_grid,
)

print(label)
print(attempts)
```

This is handy when you want a robust notebook workflow that tries repair-enabled SVI fits first and then falls back to a simple grid surface.

## Notes

- These helpers are mostly orchestration and reporting utilities.
- They are excellent for demos and notebooks, but they are not required to use the core pricing APIs.
- For the core surface objects themselves, see [Volatility surface](vol_surface.md), [SVI](svi.md), and [Local volatility](local_vol.md).
