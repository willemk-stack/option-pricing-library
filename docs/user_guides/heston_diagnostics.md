# Heston diagnostics

This guide covers the notebook-facing Heston diagnostics workflow under
`option_pricing.diagnostics.heston`.

It is a downstream review layer over the existing Fourier engine. The intended
flow is:

1. run one report with `run_heston_pricing_diagnostics(...)`
2. inspect the packaged `meta`, `tables`, and `arrays`
3. review convergence, smoothness, continuity, and visible failure modes with
   the frozen plot set

[Open the notebook](https://github.com/willemk-stack/option-pricing-library/blob/main/demos/11_heston_diagnostics_review.ipynb){ .md-button .md-button--primary }

## What this report is for

- review numerical stability behavior across one strike slice
- surface warnings, backend differences, config-sweep behavior, and panel-level
  failure modes without rebuilding diagnostics ad hoc
- keep the actual numerical source of truth under
  `option_pricing.models.heston.fourier`

It does **not** prove price correctness, prove economic smile validity, or cover
calibration, optimizer, or Monte Carlo diagnostics.

## One-call report

```python
import matplotlib
matplotlib.use("Agg")

import numpy as np

from option_pricing.diagnostics.heston import (
    HestonDiagnosticsReport,
    run_heston_pricing_diagnostics,
)
from option_pricing.diagnostics.heston.plot import (
    plot_backend_difference_by_strike,
    plot_cancellation_ratio_by_strike,
    plot_config_sweep,
    plot_panel_contributions,
    plot_smile_with_warning_overlay,
    plot_tail_fraction_by_strike,
)
from option_pricing.models.heston import HestonParams
from option_pricing.types import MarketData

market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)
strike = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=np.float64)

report = run_heston_pricing_diagnostics(
    strike=strike,
    tau=0.75,
    market=market,
    params=params,
)

assert isinstance(report, HestonDiagnosticsReport)
assert set(report.tables) >= {
    "summary",
    "slice",
    "worst_strikes",
    "backend_compare",
    "config_sweep",
    "worst_panels_p0",
    "worst_panels_p1",
}
assert {"probability_p0", "probability_p1", "config_sweep_prices"} <= set(report.arrays)
```

The one-call report is the preferred notebook entrypoint. It returns the frozen
top-level shape:

- `report.meta`
- `report.tables`
- `report.arrays`

`report.meta` carries context such as the chosen backends, forward, and the
current provisional policy block. `report.tables` is the main review surface.
`report.arrays` holds optional dense payloads that the plot helpers consume.

## Inspect the key tables

```python
summary = report.tables["summary"]
slice_table = report.tables["slice"]
worst_strikes = report.tables["worst_strikes"]
backend_compare = report.tables["backend_compare"]
config_sweep = report.tables["config_sweep"]
worst_panels_p0 = report.tables["worst_panels_p0"]
worst_panels_p1 = report.tables["worst_panels_p1"]

summary[["metric", "value", "severity"]]

slice_table[
    [
        "strike",
        "price",
        "implied_vol",
        "warning_count",
        "severity",
        "smoothness_flag",
        "discontinuity_flag",
        "suspicious_flag",
    ]
].head()

worst_strikes[
    [
        "rank",
        "strike",
        "severity",
        "combined_warning_labels",
        "config_price_span",
        "smoothness_signal",
        "discontinuity_signal",
        "suspicious_reasons",
    ]
].head()

backend_compare[["strike", "backend_a", "backend_b", "abs_price_diff"]].head()

config_sweep[
    [
        "config_label",
        "backend",
        "max_abs_price_diff_vs_baseline",
        "p0_max_severity",
        "p1_max_severity",
    ]
]

worst_panels_p0.head()
worst_panels_p1.head()
```

The most useful review pattern is:

- start with `summary`
- move to the canonical `slice` table and `worst_strikes`
- inspect `backend_compare` and `config_sweep`
- use `worst_panels_p0` and `worst_panels_p1` when you need panel-local failure
  detail

## Render the frozen plot set

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0), constrained_layout=True)

plot_panel_contributions(report, probability_key="probability_p1", ax=axes[0, 0])
plot_tail_fraction_by_strike(report, ax=axes[0, 1])
plot_cancellation_ratio_by_strike(report, ax=axes[1, 0])
plot_backend_difference_by_strike(report, ax=axes[1, 1])
plot_config_sweep(report, ax=axes[2, 0])
plot_smile_with_warning_overlay(report, ax=axes[2, 1])

plt.close(fig)
```

These helpers only consume the packaged report artifacts. They do not call the
pricing engine or rebuild diagnostics inside the plotting layer.

## How to read the report

- Convergence: use `config_sweep` and `plot_config_sweep(...)` to see whether
  small configuration changes produce materially different prices or
  probabilities.
- Smoothness and continuity: inspect `smoothness_flag`,
  `discontinuity_flag`, `smoothness_signal`, `discontinuity_signal`, and the
  smile overlay plot.
- Visible failure modes: start from `worst_strikes`, then drill into
  `worst_panels_p0` or `worst_panels_p1` and the panel-contribution plot.
- Backend disagreement: use `backend_compare` and
  `plot_backend_difference_by_strike(...)` to see where the primary and
  comparison backends diverge.

## Policy notes

The report intentionally keeps policy-sensitive semantics explicit instead of
hard-coding them into prose:

- suspiciousness thresholds remain provisional
- continuity and smoothness thresholds remain provisional
- severity labels reflect the current implementation, not owner-approved final
  semantics

If you need lower-level raw Fourier diagnostics, use the existing objects under
`option_pricing.models.heston.fourier` or inspect the packaged
`probability_p0` / `probability_p1` arrays carried by the report.
