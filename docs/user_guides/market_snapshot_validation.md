# Market Snapshot Validation

Phase A keeps market snapshot validation local-first, deterministic, and
auditable. The phase separates fixture evidence, cleaned evidence, and
library-ready conversion artifacts without connecting to live providers,
requiring credentials, adding a CLI, or writing research exports.

## Phase A Boundary

The local model-validation handoff is intentionally narrow:

- Bronze = local fixture evidence loaded from deterministic snapshots.
- Silver = normalized `market_inputs`, `cleaned_quotes`, `rejected_quotes`, and
  the cleaning manifest.
- Gold = library-ready converted artifacts.
- A5 bundle = a local model-validation evidence bundle assembled from the same
  Bronze, Silver, and Gold contracts.

A3 owns the Silver normalization and quote-cleaning evidence. A4 consumes those
Silver outputs and writes Gold artifacts that existing library types can reload.
Through A4, pricing and Heston compatibility are proved by reconstruction only.
A5 adds a local bundle writer and orchestration function; it still does not
refresh providers, call live services, require credentials, add a CLI, or write
research exports.

## A3 Silver Scope

A3 covers these functions:

- `normalize_market_inputs`
- `normalize_option_chain`
- `clean_option_quotes`
- `write_cleaned_quotes_silver`

The A3 outputs are:

- normalized `market_inputs`
- `cleaned_quotes`
- `rejected_quotes`
- `reason_counts`
- `warnings`
- Silver manifest

A3 non-goals are:

- no live providers
- no credentials
- no CLI
- no Gold
- no Heston
- no MarketData/PricingContext construction
- no model-validation bundle
- no research exports

## Normalization

`normalize_market_inputs` accepts one local market-input row and coerces it into
the canonical `market_inputs` schema. The spot used by quote cleaning comes from
this row, not from option-chain inference.

`normalize_option_chain` accepts local option-chain rows, fills missing `mid`
from bid/ask when possible, orders rows deterministically, and normalizes option
rights:

- `C/CALL/call -> call`
- `P/PUT/put -> put`

## Quote Cleaning

`QuoteCleaningPolicyV1` defaults are:

- `max_relative_spread=1.00`
- `intrinsic_tolerance=1e-8`
- `require_iv=False`
- `require_vega=False`
- `day_count="ACT/365"`

Quote cleaning emits one deterministic primary rejection reason per rejected row.
The primary reason is the first applicable reason in policy order; secondary
issues on the same row are not reported as additional reasons.

Rejection reasons are:

- `negative_bid`
- `nonpositive_ask`
- `crossed_market`
- `expired_contract`
- `nonpositive_strike`
- `missing_required_price`
- `invalid_mid`
- `below_intrinsic_tolerance`
- `spread_too_wide`
- `missing_iv_for_iv_required_workflow`
- `missing_vega_for_weighted_calibration`

Expiry and moneyness conventions:

- `expiry_years` uses `ACT/365`.
- Date-only expiry is interpreted as midnight UTC.
- `moneyness = strike / spot`.
- `spot` comes from `market_inputs`, not option-chain inference.

Intrinsic and spread conventions:

- Intrinsic value is simple spot intrinsic:
  `max(spot - strike, 0)` for calls and `max(strike - spot, 0)` for puts.
- `relative_spread = (ask - bid) / mid`.

## Silver Artifacts

`write_cleaned_quotes_silver` writes one Silver evidence set under the local
storage root:

```text
silver/
  market_inputs/
    underlying=<...>/
      date=<...>/
        run_id=<...>/
          market_inputs.parquet
  cleaned_quotes/
    underlying=<...>/
      date=<...>/
        run_id=<...>/
          cleaned_quotes.parquet
          manifest.json
  rejected_quotes/
    underlying=<...>/
      date=<...>/
        run_id=<...>/
          rejected_quotes.parquet
```

At a high level, the manifest records the Silver schema version, cleaning policy,
input/output schema versions, fixture and snapshot identity, `run_id`, local
source type, underlying, valuation timestamp, market-input values, row counts,
`reason_counts`, `warnings`, artifact filenames, and optional library commit.

## A4 Gold Scope

A4 consumes:

- normalized `market_inputs`
- `cleaned_quotes`
- rejected quote summary metadata
- cleaning `reason_counts` and `warnings`

A4 produces:

- `market_data.json`
- `market_snapshot` `manifest.json`
- `heston_quotes.parquet`
- `heston_quotes` `manifest.json`

`market_data.json` reloads into `MarketData`. `PricingContext` is reconstructed
through `MarketData.to_context()`; it is not serialized directly.

`heston_quotes.parquet` uses the existing `HESTON_QUOTES_COLUMNS` exactly.
`HestonQuoteSet` reconstruction is a compatibility proof only.

Optional IV/vega policy:

- The artifact can contain nullable `iv` and `vega`.
- `iv_mid` is passed to `HestonQuoteSet` only when every IV is finite and `> 0`.
- `bs_vega` is passed only when every vega is finite and `>= 0`.

Gold manifests summarize rejected quote counts, reasons, and warnings, but they
do not duplicate rejected quote rows.

A4 non-goals are:

- no live providers
- no credentials
- no CLI
- no provider refresh
- no calibration execution
- no model-validation bundle
- no research exports
- no `surface_inputs.parquet`

## Gold Artifacts

`write_gold_artifacts` writes one Gold output set under the local storage root:

```text
gold/
  market_snapshot/
    underlying=<...>/
      date=<...>/
        run_id=<...>/
          market_data.json
          manifest.json
  heston_quotes/
    underlying=<...>/
      date=<...>/
        run_id=<...>/
          heston_quotes.parquet
          manifest.json
```

The market snapshot manifest references `market_data.json`. The Heston quote
manifest references `heston_quotes.parquet` and records the IV/vega
reconstruction policy. Both manifests record `run_id`, `snapshot_id`, schema
versions, row counts, warnings, reason counts, source fixture metadata, artifact
paths, and optional library commit.

## A5 Local Model-Validation Bundle

A5 is a narrow local orchestration layer over the existing A2-A4 contracts. It
does not introduce live providers, credentials, CLI commands, provider refreshes,
research exports, or new storage layouts.

A5 exposes:

- `write_model_validation_bundle_artifacts`, a public wrapper around the existing
  model-validation bundle artifact writer.
- `run_local_model_validation_pipeline`, a local fixture-to-bundle pipeline that
  requires an explicit `run_id`.

The local pipeline flow is fixed:

1. Load a local snapshot with `LocalSnapshotProvider`.
2. Write Bronze local fixture evidence.
3. Normalize `market_inputs` and `option_chain`.
4. Clean option quotes.
5. Write Silver cleaned quote evidence.
6. Write A4 Gold artifacts.
7. Write the A5 model-validation bundle.
8. Return a typed result with the local snapshot, normalized frames, cleaning
   result, Bronze/Silver/Gold paths, and bundle result.

The A5 bundle is written under:

```text
gold/
  model_validation_bundle/
    underlying=<...>/
      date=<...>/
        run_id=<...>/
          manifest.json
          market_data.json
          cleaned_quotes.parquet
          rejected_quotes.parquet
          heston_quotes.parquet
          surface_inputs.parquet
          heston_fit_summary.csv
          warnings.json
```

The bundle manifest records only summary metadata and artifact filenames; it
does not embed rejected quote row details. `ModelValidationBundleConfig` controls
the Heston smoke behavior, and its defaults are part of the bundle contract.

## Example

```python
from pathlib import Path

from option_pricing.marketdata.bundles import ModelValidationBundleConfig
from option_pricing.marketdata.cleaning import clean_option_quotes
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.gold import write_gold_artifacts
from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.pipeline import run_local_model_validation_pipeline
from option_pricing.marketdata.providers.local import (
    LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
    LocalSnapshotConfig,
    LocalSnapshotProvider,
)
from option_pricing.marketdata.silver import write_cleaned_quotes_silver
from option_pricing.marketdata.storage import LocalStorage

local = LocalSnapshotProvider(
    LocalSnapshotConfig(
        fixture_name=LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1,
        run_id="run-001",
    )
).load_snapshot()

market_inputs = normalize_market_inputs(local.market_inputs_raw)
option_chain = normalize_option_chain(local.option_chain_raw)
cleaning = clean_option_quotes(option_chain, market_inputs)

storage = LocalStorage(StorageConfig(root=Path("out/marketdata")))

silver_paths = write_cleaned_quotes_silver(
    storage,
    local_snapshot=local,
    market_inputs=market_inputs,
    result=cleaning,
)

gold_paths = write_gold_artifacts(
    storage,
    local_snapshot=local,
    market_inputs=market_inputs,
    cleaned_quotes=cleaning.cleaned_quotes,
    rejected_quotes=cleaning.rejected_quotes,
    reason_counts=cleaning.reason_counts,
    warnings=cleaning.warnings,
)

pipeline_result = run_local_model_validation_pipeline(
    storage=storage,
    run_id="run-002",
    bundle_config=ModelValidationBundleConfig(run_heston_smoke=False),
)
```

## A4 Acceptance Checklist

- `marketdata/gold.py` owns Gold conversion logic.
- `market_inputs` converts into `MarketData`.
- `market_data.json` serializes and reloads.
- `MarketData.to_context()` works.
- `cleaned_quotes` converts into exact `HESTON_QUOTES_COLUMNS`.
- `heston_quotes.parquet` writes and reads locally.
- `HestonQuoteSet` reconstructs successfully.
- `expiry_years` is preserved from A3.
- `right` maps deterministically to `is_call`.
- Optional IV/vega behavior is deterministic.
- Gold artifacts partition by underlying, date, and `run_id`.
- Manifests record `run_id`, `snapshot_id`, schema versions, row counts,
  warnings, and artifact paths.
- A4 adds no credentials, providers, CLI, research exports, calibration
  execution, or model-validation bundles.

## A5 Acceptance Checklist

- `write_model_validation_bundle_artifacts` is exported from
  `marketdata/bundles.py`.
- The private bundle writer remains available for backwards compatibility.
- `run_local_model_validation_pipeline` requires an explicit `run_id`.
- The pipeline accepts local storage as `LocalStorage`, `StorageConfig`, or
  `Path`.
- Bronze, Silver, A4 Gold, and A5 bundle artifacts are all written for the same
  local snapshot.
- `overwrite=False` fails on existing deterministic outputs before replacement.
- `overwrite=True` replaces the deterministic local output set.
- No live providers, credentials, CLI, provider refresh, research exports, new
  dependencies, public pricing API changes, or storage layout changes are added.

## Testing

```powershell
ruff check .
black --check .
mypy
pytest -q tests/marketdata/test_gold_conversions.py
pytest -q tests/marketdata/test_heston_gold_conversions.py
pytest -q tests/marketdata/test_a4_gold_integration.py
pytest -q tests/marketdata/test_a5_local_pipeline.py
pytest -q tests
```
