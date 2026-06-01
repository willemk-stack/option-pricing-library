# Market Snapshot Validation

The market snapshot validation workflow is a deterministic, local-first review
path for the Phase A marketdata artifacts. It runs from the checked-in local
fixture snapshot and writes Bronze, Silver, Gold, and model-validation bundle
artifacts without credentials or live market access.

This page is for reviewers who want to understand what the workflow proves, how
to run it, which artifacts it writes, and where the phase boundary stops.

## What this workflow proves

The local fixture workflow proves that the library can consume one deterministic
market snapshot end to end. In A5/A6 scope, it demonstrates:

- local fixture-to-artifact reproducibility for the same explicit `run_id`
- library-consumption mechanics from normalized market inputs and cleaned quotes
- quote-cleaning visibility, including accepted rows, rejected rows, reason
  counts, and warnings
- `MarketData` JSON serialization and reload into the library-facing type
- `PricingContext` reconstruction through `MarketData.to_context()`
- cleaned quote compatibility with the Heston quote schema
- packaging of a self-contained model-validation bundle under the local storage
  root

The workflow uses the local fixture snapshot only. It does not call live market
provider APIs, network clients, or any credential-backed data source.

## What this workflow does not prove

This workflow does not prove production data quality, live-provider correctness,
calibration quality, trading performance, or empirical research conclusions.

It also does not provide a provider refresh path, credential setup, a CLI
workflow, or research exports. Those are outside A6-S3.

## Requirements

Run the workflow from a normal development checkout. The A6 demo requires either
the focused marketdata extra:

```powershell
pip install -e ".[marketdata]"
```

or the full contributor setup:

```powershell
pip install -e ".[dev]"
```

The only runtime inputs are:

- a writable local storage root, such as `out/marketdata`
- an explicit `run_id`
- the checked-in local fixture snapshot used by
  `run_local_model_validation_pipeline(...)`

No provider credentials, environment variables, network access, or live market
data accounts are required.

## Run the local fixture workflow

Use `run_local_model_validation_pipeline(...)` for the reviewer path. The default
fixture is local and deterministic, and the example below keeps the run
credential-free.

```python
from pathlib import Path

from option_pricing.marketdata.bundles import ModelValidationBundleConfig
from option_pricing.marketdata.pipeline import run_local_model_validation_pipeline

result = run_local_model_validation_pipeline(
    storage=Path("out/marketdata"),
    run_id="reviewer-a6-s3",
    bundle_config=ModelValidationBundleConfig(run_heston_smoke=False),
    overwrite=True,
)

print(result.model_validation_bundle.manifest_path)
```

The returned result includes the loaded local snapshot, normalized frames, quote
cleaning result, Bronze paths, Silver paths, Gold paths, and model-validation
bundle result.

## Expected artifact tree

The local pipeline writes one deterministic output set partitioned by
`underlying`, `date`, and `run_id`.

```text
out/marketdata/
  bronze/
    local_snapshot/
      underlying=<...>/
        date=<...>/
          run_id=<...>/
            manifest.json
            market_inputs.parquet
            option_chain.parquet
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

For the default local fixture and the run ID in the example above, the bundle is
written to:

```text
out/marketdata/gold/model_validation_bundle/underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3/
```

Use the manifest files as the high-level pointers into each artifact set.

## Inspect the generated artifacts

Start by binding the expected bundle directory and checking that every frozen
A5 bundle filename exists.

```python
from pathlib import Path

bundle_dir = Path(
    "out/marketdata/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3"
)
expected = (
    "manifest.json",
    "market_data.json",
    "cleaned_quotes.parquet",
    "rejected_quotes.parquet",
    "heston_quotes.parquet",
    "surface_inputs.parquet",
    "heston_fit_summary.csv",
    "warnings.json",
)

missing = [name for name in expected if not (bundle_dir / name).exists()]
print(missing or "all expected bundle files are present")
```

### Inspect the bundle manifest

`manifest.json` is the summary and routing document for the bundle. It records
schema version, run metadata, row counts, reason counts, warnings, Heston smoke
status, and artifact filenames. It does not contain rejected quote row details;
those rows live in `rejected_quotes.parquet`.

```python
import json
from pathlib import Path

manifest = json.loads(
    Path(
        "out/marketdata/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3/manifest.json"
    ).read_text()
)

print(manifest["artifact_schema_version"])
print(manifest["rows"])
print(manifest["artifacts"])
print(manifest["heston_smoke"]["status"])
print(manifest["heston_smoke"]["message"])
```

Reviewers should expect `artifact_schema_version` to be
`model_validation_bundle.v1`, the artifact map to contain only local filenames,
and the Heston smoke status to be `skipped`, `success`, or `failed` with a
message that explains the outcome.

### Inspect warnings

`warnings.json` separates workflow warnings, data-quality warnings from
library-facing quote conversion, and Heston smoke messages when present.

```python
import json
from pathlib import Path

warnings_payload = json.loads(
    Path(
        "out/marketdata/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3/warnings.json"
    ).read_text()
)

print(warnings_payload["warnings"])
print(warnings_payload["data_quality"])
print(warnings_payload["heston_smoke"])
```

Workflow warnings come from quote cleaning. Data-quality warnings describe
nullable optional inputs that affect reconstruction, such as optional IV or vega
arrays. Heston smoke messages are actionability evidence, not a calibration
quality score.

### Inspect cleaned and rejected quotes

`cleaned_quotes.parquet` contains accepted rows. `rejected_quotes.parquet`
contains the row-level rejection evidence that the manifest intentionally avoids
embedding.

```python
from pathlib import Path

import pandas as pd

bundle_dir = Path(
    "out/marketdata/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3"
)
cleaned = pd.read_parquet(bundle_dir / "cleaned_quotes.parquet")
rejected = pd.read_parquet(bundle_dir / "rejected_quotes.parquet")

print({"accepted": len(cleaned), "rejected": len(rejected)})
print(rejected["rejection_reason"].value_counts(dropna=False))
print(rejected.head())
```

Use the accepted and rejected counts to cross-check `manifest["rows"]`. Use the
reason count output to confirm rejected rows remain explainable and inspectable
outside the manifest.

### Inspect surface inputs

`surface_inputs.parquet` is the stable surface-building seed derived from
cleaned quotes. It should preserve the cleaned quote count and expose only the
surface input schema needed by downstream surface construction.

```python
from pathlib import Path

import pandas as pd

bundle_dir = Path(
    "out/marketdata/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3"
)
cleaned = pd.read_parquet(bundle_dir / "cleaned_quotes.parquet")
surface_inputs = pd.read_parquet(bundle_dir / "surface_inputs.parquet")

print({"cleaned": len(cleaned), "surface_inputs": len(surface_inputs)})
print(surface_inputs.columns.tolist())
print(surface_inputs.head())
```

This artifact is a deterministic seed for surface workflows. It is not a
research export and should not introduce model-comparison or paper-specific
claims.

### Inspect Heston-compatible quotes

`heston_quotes.parquet` is the Heston-compatible quote artifact used to
reconstruct model inputs from cleaned local quotes.

```python
from pathlib import Path

import pandas as pd

heston_quotes = pd.read_parquet(
    Path(
        "out/marketdata/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3/heston_quotes.parquet"
    )
)

print(len(heston_quotes))
print(heston_quotes.columns.tolist())
print(heston_quotes[["strike", "expiry_years", "right", "mid", "iv", "vega"]].head())
```

The Heston-compatible artifact proves that the cleaned local quotes can be
mapped into the library's Heston quote input shape. It does not prove that a
calibration is high quality.

### Inspect Heston smoke output

`heston_fit_summary.csv` is compatibility smoke evidence. It records whether the
optional smoke path was skipped, succeeded, or failed, along with an actionable
message. It is not a calibration-quality report.

```python
from pathlib import Path

import pandas as pd

summary = pd.read_csv(
    Path(
        "out/marketdata/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3/heston_fit_summary.csv"
    )
)

print(summary[["status", "message", "objective_type", "quote_count", "best_cost"]])
```

For manifest-level smoke evidence, inspect the same fields in
`manifest["heston_smoke"]`.

```python
import json
from pathlib import Path

manifest = json.loads(
    Path(
        "out/marketdata/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3/manifest.json"
    ).read_text()
)

print(manifest["heston_smoke"]["status"])
print(manifest["heston_smoke"]["message"])
```

### Inspect MarketData reload evidence

`market_data.json` records the library-facing market inputs and metadata needed
to reload `MarketData`, then reconstruct a `PricingContext` through
`MarketData.to_context()`.

```python
import json
from pathlib import Path

from option_pricing.marketdata.gold import market_data_snapshot_from_json

payload = json.loads(
    Path(
        "out/marketdata/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=reviewer-a6-s3/market_data.json"
    ).read_text()
)
snapshot = market_data_snapshot_from_json(payload)
context = snapshot.market_data.to_context()

print(payload["schema_version"])
print(payload["market_data"])
print(payload["rate_compounding"], payload["day_count"])
print(payload["sources"])
print(payload["run_id"], payload["library_commit"])
print(context)
```

Review the payload for spot, rate, dividend yield, day-count, compounding,
source labels, run ID, and `library_commit` when provided. The JSON should stay
local-only and credential-free.

### What a reviewer should look for

- Bundle directory exists at the expected local partition path.
- Every expected file exists.
- Manifest validates conceptually against `model_validation_bundle.v1`.
- Manifest artifact values are the expected local filenames.
- Row counts are internally consistent across manifest and Parquet artifacts.
- Rejected quote reasons are summarized and rejected rows are inspectable in
  `rejected_quotes.parquet`.
- Heston smoke is `skipped`, `success`, or `failed` with an actionable message.
- `market_data.json` records spot, rate, dividend, day-count, compounding,
  sources, run ID, and library commit if provided.
- Generated artifacts contain no credentials, provider names, secret-looking
  keys, or live-source claims.

### Common failure signals

- The bundle path is missing or partitioned under the wrong underlying, date, or
  run ID.
- Any expected bundle filename is absent or renamed.
- `manifest.json` has an unexpected `artifact_schema_version`.
- Manifest artifact values are paths, URLs, or generated names instead of the
  frozen local filenames.
- Manifest row counts do not match `pd.read_parquet(...)` lengths.
- Rejected quote rows appear embedded in JSON instead of in
  `rejected_quotes.parquet`.
- `warnings.json` merges workflow, data-quality, and Heston smoke messages into
  one ambiguous list.
- Heston smoke has a status outside `skipped`, `success`, or `failed`, or the
  message does not explain what happened.
- `heston_fit_summary.csv` is treated as calibration-quality evidence.
- Artifacts mention credentials, live provider names, secret-looking keys, or
  live-source claims.

## Phase boundary

Phase ownership is intentionally narrow:

- A3 owns Silver normalization and quote cleaning.
- A4 owns Gold `MarketData` and Heston quote artifacts.
- A5 owns the local model-validation bundle writer and fixture-to-bundle
  orchestration.
- A6 owns reviewer reproducibility and documentation/demo clarity.

A6-S3 is documentation-only. It does not modify providers, add provider refresh
logic, introduce a CLI script, change bundle code, or create research exports.

The original A3 Silver-only non-goals were:

- no live providers
- no credentials
- no CLI
- no Gold
- no Heston
- no MarketData/PricingContext construction
- no model-validation bundle
- no research exports

A4/A5 intentionally add Gold artifacts, Heston-compatible reconstruction, and
bundle packaging while retaining the local-only and credential-free boundary.

## Bronze, Silver, Gold, and bundle layers

Bronze preserves the local fixture evidence as loaded from the deterministic
snapshot. It records the fixture identity and writes the raw local
`market_inputs` and `option_chain` frames.

Silver converts that fixture evidence into normalized `market_inputs`, cleaned
quote rows, rejected quote rows, reason counts, warnings, and a cleaning
manifest. A3 owns this normalization and cleaning contract.

Gold converts Silver outputs into library-ready artifacts. A4 writes
`market_data.json` for `MarketData` reload and `heston_quotes.parquet` using the
existing Heston quote column contract. `PricingContext` is reconstructed through
`MarketData.to_context()`; it is not serialized directly.

The model-validation bundle is an A5 packaging layer over the same local
contracts. It collects the reloaded market data payload, cleaned quotes,
rejected quotes, Heston-compatible quotes, surface inputs, warnings, and a
minimal Heston smoke summary into one self-contained local bundle.

## Quote-cleaning policy

Quote cleaning is deterministic. The default `QuoteCleaningPolicyV1` values are:

- `max_relative_spread=1.00`
- `intrinsic_tolerance=1e-8`
- `require_iv=False`
- `require_vega=False`
- `day_count="ACT/365"`

Each rejected row receives one primary rejection reason in policy order. Rejected
quotes remain first-class evidence and are written to `rejected_quotes.parquet`
in Silver and in the model-validation bundle.

Rejected quote rows must not be duplicated into manifest JSON. Manifests may
record rejected row counts, reason counts, warnings, and artifact filenames, but
the row-level evidence stays in Parquet artifacts.

The local cleaning conventions are:

- `expiry_years` uses `ACT/365`
- date-only expiry is interpreted as midnight UTC
- `moneyness = strike / spot`
- `spot` comes from normalized `market_inputs`
- intrinsic value uses simple spot intrinsic for calls and puts
- `relative_spread = (ask - bid) / mid`

Primary rejection reasons are:

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

## Model-validation bundle

The A5 bundle is written under `gold/model_validation_bundle/...` for the same
`underlying`, `date`, and `run_id` as the Bronze, Silver, and Gold artifacts.
The frozen bundle artifact names are:

- `manifest.json`
- `market_data.json`
- `cleaned_quotes.parquet`
- `rejected_quotes.parquet`
- `heston_quotes.parquet`
- `surface_inputs.parquet`
- `heston_fit_summary.csv`
- `warnings.json`

The bundle manifest records summary metadata, row counts, reason counts,
warnings, Heston smoke status, and artifact filenames. It does not embed
rejected quote row details.

## Assumptions and limitations

The workflow assumes the local fixture snapshot is the evidence source under
review. It is suitable for checking deterministic artifact mechanics and
library-facing compatibility, not for evaluating live market data feeds.

The bundle may include a Heston smoke result, but that result is a packaging and
compatibility signal only. It is not a claim about production calibration
quality, model fitness, strategy performance, or empirical validity.

No live providers, no credentials, no CLI refresh, and no research exports are
part of A6-S3.

## Developer checks

Run the local quality and marketdata checks from the repository root:

```powershell
ruff check .
black --check .
mypy
pytest -q tests/marketdata/test_a5_local_pipeline.py
pytest -q tests/marketdata/test_model_validation_bundle.py
```

For rendered documentation review, serve the MkDocs site locally and inspect the
page in light and dark theme at 375, 768, 1280, and 1536 pixel widths.
