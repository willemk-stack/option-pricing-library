# Model-Ready Heston Workflow

This guide is the canonical public path from saved marketdata validation
artifacts to a Heston market fit.

The workflow is intentionally split across two namespaces:

- `option_pricing.marketdata` loads and prepares saved marketdata artifacts.
- `option_pricing.workflows` runs the Heston fit workflow.

Use this path when you have a local model-validation bundle from a fixture run,
a provider-backed snapshot, or another local validation artifact set. Advanced
users can still call the low-level Heston calibration APIs directly when they
already have a validated `HestonQuoteSet`.

## Pricing-ready versus calibration-ready

Pricing-ready means the library can reconstruct market assumptions and
model-facing quote rows from local artifacts. A model-validation bundle proves
that the files can be read back into `MarketData`, cleaned quote tables,
rejected quote evidence, surface inputs, and Heston-compatible quote columns.

Calibration-ready is narrower. It means a specific Heston objective can be run
on selected quotes after calibration-specific checks:

- required Heston quote columns are present
- spot, rate, and dividend assumptions are numeric and finite
- time to expiry is positive and inside the configured window
- option type, strike, bid, ask, mid, IV, and vega assumptions match the chosen
  objective
- Heston preflight does not block the selected quote set

`heston_quotes.parquet` is therefore a candidate artifact, not a promise that
every row should enter calibration. It preserves the Heston-compatible input
shape from the saved bundle. `prepare_heston_market_fit(...)` is the layer that
turns that candidate artifact into selected and rejected calibration rows.

## Official Ladder

The explicit path is:

```python
from pathlib import Path

from option_pricing.marketdata import (
    load_model_validation_bundle,
    prepare_heston_market_fit,
)
from option_pricing.workflows import fit_heston_market

path = Path(
    "out/marketdata-demo/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=demo-run"
)

bundle = load_model_validation_bundle(path)
prepared = prepare_heston_market_fit(bundle)
result = fit_heston_market(prepared)

print(result.status)
print(result.summary)
```

The one-shot helper is:

```python
from pathlib import Path

from option_pricing.workflows import fit_heston_from_bundle

path = Path(
    "out/marketdata-demo/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=demo-run"
)

result = fit_heston_from_bundle(path)

print(result.status)
print(result.summary)
```

Both examples use a synthetic local path. They do not require private market
data or provider payloads.

## When To Use Each Helper

| Helper | Namespace | Use it when |
| --- | --- | --- |
| `load_model_validation_bundle(...)` | `option_pricing.marketdata` | You have a bundle directory or `manifest.json` and want the local artifacts loaded into typed Python objects. |
| `prepare_heston_market_fit(...)` | `option_pricing.marketdata` | You want to inspect which Heston-compatible quote rows are selected or rejected before fitting. |
| `fit_heston_market(...)` | `option_pricing.workflows` | You already have a `PreparedHestonMarketFit` and want to run calibration without changing the prepared quote universe. |
| `fit_heston_from_bundle(...)` | `option_pricing.workflows` | You want the canonical one-shot path from bundle path to fit result. |

Use the explicit ladder while reviewing data quality, rejected quotes, warnings,
or preflight output. Use the one-shot helper when you trust the saved bundle and
want the standard orchestration.

## Result Surfaces

`load_model_validation_bundle(...)` returns a `LoadedModelValidationBundle`.
The most important fields are:

- `market_data`: reconstructed `MarketData`
- `cleaned_quotes`: market-sane accepted quotes from the bundle
- `rejected_quotes`: row-level cleaning evidence from the bundle
- `heston_quotes`: Heston-compatible candidate rows
- `surface_inputs`: deterministic surface-building seed rows
- `heston_fit_summary`: optional bundle smoke output
- `warnings`: warnings grouped by workflow, data quality, and Heston smoke

`prepare_heston_market_fit(...)` returns a `PreparedHestonMarketFit`:

- `selected_quotes`: rows that passed the preparation filters for the chosen
  objective
- `rejected_quotes`: rows rejected by preparation, with `reject_reasons`
- `stats`: aggregate input, selected, rejected, expiry, and rejection counts
- `preflight`: Heston preflight result when selected rows exist
- `warnings`: preparation and preflight warnings
- `status`: `ready`, `empty`, or `blocked`

`fit_heston_market(...)` and `fit_heston_from_bundle(...)` return a
`HestonMarketFitResult`:

- `status`: `ok`, `empty`, `blocked`, or `failed`
- `prepared`: the exact `PreparedHestonMarketFit` used by the fit
- `calibration_result`: the low-level multistart result when calibration ran
- `best_params`: fitted Heston parameters when available
- `summary`: compact public counts, objective, status, and fitted-parameter
  summary
- `warnings`: recoverable workflow messages
- `errors`: failure or skip messages with context and next-step guidance

## Status Meanings

Preparation status:

- `ready`: selected quotes exist and preflight did not block calibration.
- `empty`: preparation rejected every candidate quote, so no calibration should
  run.
- `blocked`: selected quotes exist, but Heston preflight recommends blocking
  calibration.

Fit status:

- `ok`: calibration ran and returned a fitted result.
- `empty`: fitting skipped because `prepared.selected_quotes` is empty.
- `blocked`: fitting skipped because preflight blocked the prepared universe.
- `failed`: calibration was attempted but raised or could not produce a usable
  result.

`allow_blocked=True` keeps an advanced escape hatch for explicit reruns. It
does not make blocked data clean, and it should be documented when used.

## Provider-Backed Runs

Provider-backed snapshots can write Bronze provider evidence, Silver cleaned
quotes, Gold market snapshots, rate-curve artifacts, and a model-validation
bundle under the local storage root. Those files are local operator evidence,
not redistributable provider payloads.

After a provider-backed run writes a local model-validation bundle, use the same
public workflow:

```python
from option_pricing.workflows import fit_heston_from_bundle

result = fit_heston_from_bundle(
    "out/marketdata-live/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=local-review"
)
```

Review `result.summary`, `result.prepared.stats`,
`result.prepared.rejected_quotes`, `result.warnings`, and `result.errors`
before interpreting fitted parameters.

## Advanced Low-Level Path

Low-level Heston calibration APIs remain available. They are the right tools
when you already have a validated `HestonQuoteSet`, need custom experimental
controls, or are building diagnostics from synthetic fixtures.

For saved model-validation bundles, prefer the public workflow first. It keeps
the artifact loading, preparation, preflight, fit status, selected quotes,
rejected quotes, summaries, warnings, and errors in one reviewable path.
