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

It does **not** prove price correctness or prove economic smile validity. For
calibration evidence, use `run_heston_calibration_fit_diagnostics(...)`; for
Monte Carlo evidence, use the Heston MC diagnostics.

## Calibration fit diagnostics

`run_heston_calibration_fit_diagnostics(...)` packages fitted Heston evidence
without rerunning the optimizer:

- quote-level price and implied-vol residuals
- smile overlay data by maturity
- long-form IV residual grids
- parameter recovery when synthetic truth is supplied
- multistart run metadata when a `HestonMultistartResult` is supplied
- held-out error summaries when a mask is supplied
- lightweight objective slices around the fitted point

```python
from option_pricing.diagnostics.heston import (
    run_heston_calibration_fit_diagnostics,
)

fit_report = run_heston_calibration_fit_diagnostics(
    quotes=quotes,
    fit=multistart_result,
    true_params=true_params,
    held_out_mask=held_out_mask,
    quad_cfg=quad_cfg,
)

fit_report.tables["residuals"].head()
fit_report.tables["held_out_errors"]
```

Heston calibration may be weakly identifiable from vanilla quotes. Multistart is
diagnostic, not magic: it exposes initialization sensitivity and failed seeds,
but it does not prove parameter uniqueness.

Vega-scaled price residuals can act as an IV-like optimization proxy, but they
are not direct IV RMSE unless model prices are inverted and IV residuals are
reported. The calibration-fit diagnostics report both price residuals and
direct IV residuals when possible.

## Heston versus local-vol comparison

`run_heston_vs_local_vol_comparison(...)` compares Heston with the existing
eSSVI/local-vol-facing path on the same vanilla target.

REVIEW: The current local-vol side uses eSSVI implied-surface repricing as a
proxy and does not run a full Dupire PDE repricer. Conclusions depend on the
target quotes, weights, held-out partition, and local-vol proxy.

## One-call report

```python
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

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

Config provenance is surfaced explicitly rather than hidden behind
`use_recommended_cfg=True`:

- `report.meta["primary_backend_config"]` and
  `report.meta["comparison_backend_config"]` show the effective backend,
  resolution path, and resolved quadrature fields for the two main runs.
- `report.tables["config_sweep"]` repeats the effective config used for each
  sweep case via the `resolved_*` columns.
- `report.arrays["parameter_perturbation_table"]` carries the effective backend
  and resolved config used for each perturbation rerun, so you can verify that a
  recommended config path was actually reused.

If you pass an explicit prebuilt rule, the report still records
`config_resolution="explicit_rule"`, but the individual quadrature fields stay
null because the `CompositeRule` object is treated as authoritative and does not
preserve a round-trippable `QuadratureConfig`.

## Inspect the key tables

```python
summary = report.tables["summary"]
slice_table = report.tables["slice"]
worst_strikes = report.tables["worst_strikes"]
backend_compare = report.tables["backend_compare"]
config_sweep = report.tables["config_sweep"]
parameter_perturbations = pd.DataFrame(report.arrays["parameter_perturbation_table"])
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
        "parameter_sensitivity_flag",
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
        "parameter_sensitivity_reasons",
        "suspicious_reasons",
    ]
].head()

backend_compare[["strike", "backend_a", "backend_b", "abs_price_diff"]].head()

config_sweep[
    [
        "config_label",
        "backend",
        "config_resolution",
        "resolved_u_max",
        "resolved_n_panels",
        "max_abs_price_diff_vs_baseline",
        "p0_max_severity",
        "p1_max_severity",
    ]
]

parameter_perturbations[
    [
        "label",
        "parameter",
        "backend",
        "config_resolution",
        "resolved_u_max",
        "resolved_n_panels",
        "max_abs_price_change",
        "max_relative_price_change",
    ]
]

worst_panels_p0.head()
worst_panels_p1.head()
```

The most useful review pattern is:

- start with `summary`
- move to the canonical `slice` table and `worst_strikes`
- inspect `backend_compare`, `config_sweep`, and the perturbation table
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
- Parameter sensitivity: use `parameter_sensitivity_flag`,
  `parameter_sensitivity_reasons`, and the perturbation table to separate
  meaningful parameter sensitivity from ordinary pricing instability.

## What the Main Flags Mean

- `critical` integration warnings mean "do not trust this result without a
  rerun or root-cause investigation." Non-finite totals and probabilities
  outside `[0, 1]` belong here.
- `warning` and `severe` integration warnings are review signals, not automatic
  rejection. A finite point can still be numerically usable if backend
  agreement, config sweep span, and denser reruns remain tight.
- `suspicious_flag` is the high-level strike review policy. It combines backend
  disagreement, non-OK integration severity, smoothness and discontinuity
  signals, and config-sweep span.
- `parameter_sensitivity_flag` is deliberately separate. It says that small
  parameter bumps moved the price meaningfully; it is not, by itself, evidence
  that the underlying Fourier price is wrong.

## A Severe Flag Does Not Automatically Mean the Price Is Unusable

`severe` should usually be read as "this point needs review," not as "throw the
price away immediately."

A finite Heston price can carry a severe warning because the diagnostics are
intentionally conservative. In practice, the next question is whether the point
stays stable when you:

- compare the primary and comparison backends
- rerun with a denser balanced or robust config
- inspect config-sweep span and worst-panel detail

If those checks stay tight, the point may still be usable even though it
deserves documentation or caution. By contrast, if the price keeps moving
materially under denser settings or backend comparison, treat the severe flag as
evidence of real numerical fragility.

## First Knobs to Try

- Tail or truncation warnings: increase `u_max` first.
- Oscillation or coarse panel structure: increase `n_panels` first, then
  `nodes_per_panel` if the issue looks local within panels.
- Near-origin concentration: keep the warning visible, but try more local
  resolution or clustered spacing before treating it as a failure.
- Backend disagreement with otherwise quiet low-level warnings: rerun with a
  denser balanced or robust Gauss-Legendre config and inspect the config sweep.
- Tiny-price wing perturbation noise: check both the relative and absolute price
  movement before escalating it into a strike-level failure.

## Policy notes

The report intentionally keeps policy-sensitive semantics explicit instead of
hard-coding them into prose:

- suspiciousness thresholds remain provisional
- continuity and smoothness thresholds remain provisional
- perturbation sensitivity is tracked separately from suspiciousness so tiny
  wing prices do not automatically become strike-level failures
- severity labels reflect the current implementation, not owner-approved final
  semantics

If you need lower-level raw Fourier diagnostics, use the existing objects under
`option_pricing.models.heston.fourier` or inspect the packaged
`probability_p0` and `probability_p1` arrays carried by the report.
