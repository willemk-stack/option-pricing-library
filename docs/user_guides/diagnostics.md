# Diagnostics (notebooks)

The `option_pricing.diagnostics` modules are meant for notebook-style sanity checks:

- MC vs BS across a grid of parameter regimes
- Convergence tables (error and SE vs number of paths)
- Optional plotting helpers

These utilities typically require extra dependencies:

```bash
pip install -e ".[notebooks]"
```

## Compare MC vs BS across cases

A common workflow is:

1) define a “base” parameter set
2) generate a list of modified cases
3) build a `pandas.DataFrame` comparing MC and BS

```python
from dataclasses import replace

import numpy as np

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs
from option_pricing.diagnostics.mc_vs_bs import compare_table

base = PricingInputs(
    spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0),
    market=MarketData(spot=100.0, rate=0.05),
    sigma=0.2,
    t=0.0,
)

# Example case edits (PricingInputs is nested: market/spec)
cases = [
    ("ATM", base),
    ("ITM (spot=120)", replace(base, market=replace(base.market, spot=120.0))),
    ("High vol (80%)", replace(base, sigma=0.8)),
]

df = compare_table(cases, n_paths=50_000, seed=0)
display(df.head())
```

## Convergence table (SE scaling)

```python
from option_pricing.diagnostics.mc_vs_bs import convergence_table

df_conv = convergence_table(base, n_paths_list=[1_000, 5_000, 10_000, 50_000], seed=0)
display(df_conv)
```

A quick “math-signal” check is that SE drops roughly like `1/sqrt(N)`.

## Plotting helpers

- `option_pricing.plotting.core` contains generic histogram/curve utilities.
- `option_pricing.diagnostics.mc_vs_bs_plots` contains plotting helpers for the comparison tables.

These require `matplotlib`.
