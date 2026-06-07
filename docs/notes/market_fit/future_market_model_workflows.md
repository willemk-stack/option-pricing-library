# Future market-model workflows

!!! note "Status: Phase 8C SVI fit helpers"
    This note documents the intended market-fit workflow architecture beyond
    Heston. Phase 8B added SVI preparation from bundle surface inputs. Phase 8C
    adds explicit SVI fit helpers on top of that preparation contract, while
    eSSVI and local-vol fitting helpers described here remain future work.

## Purpose

The model-validation bundle now has enough structure to support a reusable
market-fit pattern without forcing every model through the same API too early.
Heston is the reference implementation because it already separates local
artifact loading, model-specific preparation, and fitting:

```text
load_model_validation_bundle(...)
-> prepare_heston_market_fit(...)
-> fit_heston_market(...) / fit_heston_from_bundle(...)
```

The Phase 8 extension should carry that pattern to volatility-surface models
while preserving explicit model semantics:

```text
surface_inputs.parquet
-> prepare_svi_market_fit(...)
-> fit_svi_market(...) / fit_svi_from_bundle(...)

surface_inputs.parquet
-> prepare_essvi_market_fit(...)
-> fit_essvi_market(...) / fit_essvi_from_bundle(...)
```

The key rule is that bundle artifacts remain candidate inputs. The fitting
workflow decides which rows or points are usable for a specific objective,
records what was rejected, and passes only prepared model objects into the
low-level calibrator.

## Current Heston reference workflow

The current public Heston path is documented in the
[model-ready Heston workflow](../../user_guides/model_ready_heston_workflow.md).
Its responsibilities are split deliberately:

- `load_model_validation_bundle(...)` loads the complete local
    `model_validation_bundle.v1` directory and reconstructs `MarketData`,
    `cleaned_quotes`, bundle-level `rejected_quotes`, `heston_quotes`,
    `surface_inputs`, warnings, and Heston smoke evidence.
- `prepare_heston_market_fit(...)` starts from `bundle.heston_quotes`, applies
    Heston objective-specific readiness checks, and returns
    `selected_quotes` and `rejected_quotes`.
- `fit_heston_market(...)` accepts only a `PreparedHestonMarketFit` and runs
    calibration against the prepared `HestonQuoteSet`.
- `fit_heston_from_bundle(...)` is the one-shot convenience helper for the same
    load -> prepare -> fit ladder.

This design keeps provider adapters out of fitting. Alpaca, FRED, local
fixtures, and any later provider are upstream evidence producers. Once a
model-validation bundle exists, fitting consumes local artifacts and typed
market assumptions, not provider clients or provider response bodies.

`heston_quotes.parquet` is therefore a candidate Heston artifact, not an
automatically calibration-ready artifact. It proves that accepted quote rows can
be represented in the Heston-compatible shape. `prepare_heston_market_fit(...)`
is the layer that decides whether each candidate quote is selected for the
chosen Heston objective or rejected with preparation reasons.

## Phase 8B SVI preparation helper

Phase 8B adds the preparation-only SVI handoff:

```python
from option_pricing.marketdata import (
    load_model_validation_bundle,
    prepare_svi_market_fit,
)

path = "path/to/model_validation_bundle"
bundle = load_model_validation_bundle(path)
prepared = prepare_svi_market_fit(bundle)
```

`prepare_svi_market_fit(...)` starts from `bundle.surface_inputs`, not
`bundle.heston_quotes`, and returns `selected_points`, `rejected_points`,
`stats`, `status`, and `warnings`. It computes SVI-specific fields in memory:
expiry days, forward, discount, log-forward moneyness, total variance, default
square-root weights, and option side. The persisted `surface_inputs.v1`
artifact remains unchanged.

The preparation layer is intentionally not fitting. The Phase 8C SVI workflow
consumes `PreparedSVIMarketFit`, lets users inspect
`prepared.stats.rejection_counts`, and then performs slice-level SVI
calibration only after the preparation contract has selected enough points per
expiry.

## Phase 8C SVI fit helpers

Phase 8C adds the explicit SVI ladder on top of
`prepare_svi_market_fit(...)`:

```python
from option_pricing.marketdata import (
    load_model_validation_bundle,
    prepare_svi_market_fit,
)
from option_pricing.workflows import fit_svi_from_bundle, fit_svi_market

path = "path/to/model_validation_bundle"
bundle = load_model_validation_bundle(path)
prepared = prepare_svi_market_fit(bundle)
result = fit_svi_market(prepared)
```

The one-shot helper runs the same load -> prepare -> fit path:

```python
result = fit_svi_from_bundle(path)
```

`fit_svi_market(...)` accepts only `PreparedSVIMarketFit`. It groups
`prepared.selected_points` by `expiry_years`, passes the prepared
`log_moneyness`, `total_variance`, and `sqrt_weight` arrays into
`calibrate_svi(...)`, and records each expiry independently. The result keeps
the prepared selected/rejected points available, plus per-expiry params and
diagnostics, a `parameter_table`, an optional analytic `VolSurface`, a compact
summary, warnings, and errors.

Partial failure is visible by design. If one expiry fits and another fails, the
result status is `partial` when `allow_partial=True`; if every expiry fails, the
status is `failed`. No generic `fit_market_model(...)` registry is introduced.

## Model eligibility rules

Direct market-fit workflows are reserved for models whose parameters are
estimated from contemporaneous vanilla option market quotes or quote-derived
surface points:

- Eligible now: Heston, because it calibrates stochastic-volatility parameters
    directly to selected vanilla quotes.
- Eligible next: raw SVI, because it fits one expiry slice at a time from
    implied-volatility or total-variance points derived from validated quotes.
- Eligible after SVI: eSSVI, because it fits a cross-maturity volatility
    surface from validated surface points and records node or projection
    diagnostics.
- Derived later: local vol / Dupire, because it should be built from a
    validated fitted surface with stable derivatives, not fit directly from raw
    quote rows.

The following engines are explicitly excluded from direct market-fit workflows:

- Black-Scholes, which is a pricing and implied-volatility inversion baseline
    rather than a multi-parameter market-fit workflow.
- Binomial trees, which are numerical pricing engines for a supplied model, not
    market-fit models.
- Monte Carlo engines, which estimate prices under supplied dynamics and random
    seeds, not market-fit parameters from quotes.
- Finite-difference engines, which solve pricing PDEs for supplied inputs and
    grids, not direct quote-fit models.

Those engines may appear inside diagnostics, pricing checks, or validation
evidence. They should not receive `fit_*_from_bundle(...)` workflows unless a
future issue defines a genuine quote-calibrated model layer above them.

## Recommended support order

The implementation order should stay narrow and reviewable:

1. Heston remains the reference path and public example.
2. SVI comes next, using `surface_inputs.parquet` as the bundle seed for
    slice-level total-variance preparation.
3. eSSVI follows, also using `surface_inputs.parquet`, with cross-maturity
    preparation and node-level diagnostics.
4. Local vol / Dupire comes later as a derived workflow from a validated fitted
    SVI or eSSVI surface.

This order matches the existing volatility notes: the
[SVI calibration design](../volatility/svi_calibration_design.md) keeps
slice-level repair and diagnostics visible, and the
[eSSVI calibration design](../volatility/essvi_calibration_design.md) treats
smooth projection as an explicit Dupire-oriented handoff.

## Proposed public APIs

The next APIs should follow the Heston split and stay explicit:

```python
from option_pricing.marketdata import (
    load_model_validation_bundle,
    prepare_essvi_market_fit,
    prepare_heston_market_fit,
    prepare_svi_market_fit,
)
from option_pricing.workflows import (
    fit_essvi_from_bundle,
    fit_essvi_market,
    fit_heston_from_bundle,
    fit_heston_market,
    fit_svi_from_bundle,
    fit_svi_market,
)
```

SVI should use:

```text
load_model_validation_bundle(path)
-> prepare_svi_market_fit(bundle)
-> fit_svi_market(prepared)
```

and the one-shot helper:

```text
fit_svi_from_bundle(path)
```

eSSVI should use:

```text
load_model_validation_bundle(path)
-> prepare_essvi_market_fit(bundle)
-> fit_essvi_market(prepared)
```

and the one-shot helper:

```text
fit_essvi_from_bundle(path)
```

Do not introduce a generic `fit_market_model(...)` registry yet. The explicit
helpers should come first so the library can learn the real preparation
contracts, result shapes, diagnostics, and naming conventions for Heston, SVI,
and eSSVI before abstracting over them.

Provider adapters should not be passed to any `prepare_*_market_fit(...)` or
`fit_*_market(...)` helper. Preparation starts from `LoadedModelValidationBundle`
and its local artifacts.

## Proposed dataclasses

Heston already establishes the naming pattern:

- `PreparedHestonMarketFit`
- `HestonReadyStats`
- `HestonCalibrationConfig`
- `HestonMarketFitResult`

SVI should mirror that shape while using point terminology:

- `SurfaceReadyStats`
- `PreparedSVIMarketFit`
- `SVIMarketFitConfig`
- `SVIMarketFitResult`

`PreparedSVIMarketFit` should include:

- `model_name="svi"`
- `market_data`
- `selected_points`
- `rejected_points`
- slice grouping metadata such as expiries and point counts
- arrays or slice payloads derived from `surface_inputs`: log-moneyness,
    expiry years, total variance, optional weights, and option side when needed
- preparation status, warnings, and rejection counts

eSSVI should use the same point vocabulary:

- `ESSVIReadyStats`
- `PreparedESSVIMarketFit`
- `ESSVIMarketCalibrationConfig`
- `ESSVIMarketFitResult`

`PreparedESSVIMarketFit` should include:

- `model_name="essvi"`
- `market_data`
- `selected_points`
- `rejected_points`
- cross-maturity arrays derived from `surface_inputs`: log-moneyness,
    maturity, market price, total variance diagnostics, optional weights, and
    option side
- node-readiness metadata, preparation status, warnings, and rejection counts

The selected/rejected naming is intentional. Heston uses
`selected_quotes` and `rejected_quotes` because its fit object is a quote set.
SVI and eSSVI should use `selected_points` and `rejected_points` because
preparation converts the stable surface seed into model-specific surface
points.

## Surface inputs contract

`surface_inputs.v1` should stay minimal and stable. The bundle writer should
continue to write the durable seed columns only:

- `underlying`
- `quote_id`
- `asof`
- `expiry`
- `expiry_years`
- `strike`
- `right`
- `mid`
- `iv`
- `source`
- `cleaning_policy`

Do not add SVI-only or eSSVI-only columns to the bundle artifact just to make a
future fitter easier. Model-specific enrichment belongs in
`prepare_svi_market_fit(...)` and `prepare_essvi_market_fit(...)`, where the
helper can compute forward price, log-moneyness, total variance, option side,
weights, expiry buckets, and rejection reasons under an explicit preparation
configuration.

Like `heston_quotes.parquet`, `surface_inputs.parquet` is a candidate artifact.
It proves that cleaned quotes can be represented as stable surface-building
seed rows. It does not promise that every row has positive IV, enough expiry
coverage, enough strikes per slice, or a valid cross-maturity shape for SVI or
eSSVI calibration.

## Local-vol and Dupire boundary

Local vol should be documented and implemented as a derived workflow:

```text
validated fitted SVI/eSSVI surface
-> validated smooth or differentiable surface representation
-> Dupire/local-vol extraction
-> PDE or diagnostic use
```

It should not be a direct quote-fit workflow:

```text
surface_inputs.parquet
-> fit_local_vol_from_bundle(...)
```

That direct path would hide the most important risk: Dupire depends on stable
strike and maturity derivatives of a fitted surface. The workflow must first
record the fitted surface, no-arbitrage checks, smooth projection or repair
diagnostics, and derivative-readiness evidence.

## Testing strategy

Future implementation tests should be layered like the Heston workflow tests:

- Bundle loading tests prove `surface_inputs.parquet` is present, schema-valid,
    and loaded through `LoadedModelValidationBundle`.
- SVI preparation tests cover missing columns, nonpositive IV, nonpositive
    expiry, invalid strike, too few points per slice, extreme moneyness, and
    selected/rejected point accounting.
- eSSVI preparation tests cover the SVI point checks plus cross-maturity
    coverage, maturity ordering, node-readiness, and selected/rejected point
    accounting.
- Workflow tests prove `fit_svi_market(...)` and `fit_essvi_market(...)`
    require the prepared dataclass, skip empty or blocked inputs clearly, and
    preserve preparation diagnostics in the result.
- One-shot tests prove `fit_svi_from_bundle(...)` and
    `fit_essvi_from_bundle(...)` execute the same load -> prepare -> fit
    sequence as the explicit ladder.
- Regression tests keep provider-specific classes out of preparation and fit
    signatures.

The focused command set for this Phase 8A docs note is:

```powershell
python -m pytest tests/marketdata tests/vol/svi tests/vol/ssvi
ruff check docs src tests
```

Future code phases should add workflow-specific tests under `tests/workflows`
and preparation-specific tests under `tests/marketdata`.

## Documentation strategy

The docs should keep the public path explicit and model-specific:

- Keep the
    [model-ready Heston workflow](../../user_guides/model_ready_heston_workflow.md)
    as the reference implementation page.
- Add or expand an SVI market-fit guide around `prepare_svi_market_fit(...)`,
    `fit_svi_market(...)`, and `fit_svi_from_bundle(...)`.
- Add an eSSVI market-fit guide only when `prepare_essvi_market_fit(...)`,
    `fit_essvi_market(...)`, and `fit_essvi_from_bundle(...)` exist.
- Keep bundle artifact documentation in
    [market snapshot validation](../../user_guides/market_snapshot_validation.md)
    focused on local evidence and candidate artifacts.
- Keep SVI and eSSVI mathematical and diagnostic policies in their volatility
    design notes instead of duplicating them inside bundle docs.
- Document local vol as a downstream surface-derived workflow, with links to
    Dupire and PDE validation, rather than as a bundle-fitting endpoint.

Every user guide should show the explicit ladder first and the one-shot helper
second. The docs should avoid presenting a generic market-model registry until
at least Heston, SVI, and eSSVI have stable public workflow surfaces.

## Non-goals

This phase does not:

- implement eSSVI or local-vol market-fit code
- change the `model_validation_bundle.v1` artifact filenames
- change `surface_inputs.v1`
- add provider-specific logic to fitting
- promote `heston_quotes.parquet` or `surface_inputs.parquet` to
    automatically calibration-ready artifacts
- introduce `fit_market_model(...)` or a model registry
- make Black-Scholes, binomial trees, Monte Carlo engines, or
    finite-difference engines direct market-fit workflows
- claim that synthetic fixture validation proves live-market calibration
    quality
