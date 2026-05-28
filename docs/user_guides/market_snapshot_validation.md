# Market Snapshot Validation

Phase A keeps model-validation inputs local, deterministic, and auditable. A3 is
the Silver-layer normalization and quote-cleaning step for local snapshots: it
takes schema-compatible local inputs, normalizes them, separates accepted quotes
from rejected quotes, and writes those evidence frames locally. It does not
connect to live market providers or build pricing/model-validation objects.

## Phase A Boundary

The Phase A local model-validation boundary is intentionally narrow:

- Bronze captures local snapshot evidence as loaded from deterministic fixtures.
- Silver normalizes Bronze-style inputs and records quote-cleaning evidence.
- Gold is the later model-facing layer that will own converted artifacts and
  model-validation bundles.

A3 sits entirely in Silver. It preserves the local-first handoff while making
quote acceptance, rejection, and provenance explicit enough for review.

## A3 Scope

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
silver/market_inputs/underlying=<...>/date=<...>/run_id=<...>/market_inputs.parquet
silver/cleaned_quotes/underlying=<...>/date=<...>/run_id=<...>/cleaned_quotes.parquet
silver/cleaned_quotes/underlying=<...>/date=<...>/run_id=<...>/manifest.json
silver/rejected_quotes/underlying=<...>/date=<...>/run_id=<...>/rejected_quotes.parquet
```

At a high level, the manifest records the Silver schema version, cleaning policy,
input/output schema versions, fixture and snapshot identity, `run_id`, local
source type, underlying, valuation timestamp, market-input values, row counts,
`reason_counts`, `warnings`, artifact filenames, and optional library commit.

## Example

```python
from option_pricing.marketdata.cleaning import clean_option_quotes
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.normalize import (
    normalize_market_inputs,
    normalize_option_chain,
)
from option_pricing.marketdata.providers.local import (
    LocalSnapshotConfig,
    LocalSnapshotProvider,
)
from option_pricing.marketdata.silver import write_cleaned_quotes_silver
from option_pricing.marketdata.storage import LocalStorage

provider = LocalSnapshotProvider(
    LocalSnapshotConfig(
        fixture_name="local_snapshot_synth_schema_v1",
        run_id="run-001",
    )
)
snapshot = provider.load_snapshot()

market_inputs = normalize_market_inputs(snapshot.market_inputs_raw)
option_chain = normalize_option_chain(snapshot.option_chain_raw)
cleaning = clean_option_quotes(option_chain, market_inputs)

storage = LocalStorage(StorageConfig(root="out/marketdata"))
paths = write_cleaned_quotes_silver(
    storage,
    local_snapshot=snapshot,
    market_inputs=market_inputs,
    result=cleaning,
)
```

## Testing

```powershell
ruff check .
black --check .
mypy
pytest -q tests/marketdata/test_normalize.py
pytest -q tests/marketdata/test_quote_cleaning.py
pytest -q tests/marketdata/test_a3_local_snapshot_cleaning_integration.py
pytest -q tests
```
