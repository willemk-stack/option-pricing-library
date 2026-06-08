# Market Snapshot Validation

The market snapshot validation workflow is a deterministic, local-first review
path for marketdata artifacts. It runs from local synthetic fixture snapshots
and writes Bronze, Silver, Gold, and model-validation bundle artifacts without
credentials or live market access.

This page is for reviewers who want to understand what the workflow proves, how
to run it, which artifacts it writes, and where the current scope stops.

## What this workflow proves

The local fixture workflow proves that the library can consume one deterministic
market snapshot end to end. In the current local-validation release, it
demonstrates:

- local fixture-to-artifact reproducibility for the same explicit `run_id`
- library-consumption mechanics from normalized market inputs and cleaned quotes
- quote-cleaning visibility, including accepted rows, rejected rows, reason
  counts, and warnings through a dedicated rejected-quote fixture/test path
- `MarketData` JSON serialization and reload into the library-facing type
- `PricingContext` reconstruction through `MarketData.to_context()`
- cleaned quote compatibility with the Heston quote schema
- all-quotes-rejected snapshots remain auditable and skip Heston smoke clearly
- packaging of a self-contained model-validation bundle under the local storage
  root

The workflow uses the local fixture snapshot only. It does not call live market
provider APIs, network clients, or any credential-backed data source.

Provider-backed snapshots have a separate confidence path. After a provider run
writes artifacts, `validate_provider_snapshot_bundle(...)` can read back
`market_data.json`, `cleaned_quotes.parquet`, and `heston_quotes.parquet`,
reconstruct `MarketData`, and verify Heston quote-set compatibility. This is a
controlled real-data validation check, not a production data-quality claim.
The same helper is available through:

```bash
option-pricing-marketdata validate-bundle \
  --market-data path/to/market_data.json \
  --cleaned-quotes path/to/cleaned_quotes.parquet \
  --heston-quotes path/to/heston_quotes.parquet
```

## Workflow boundaries

### Local synthetic fixture workflow

The local workflow uses synthetic fixtures from `tests/marketdata/fixtures`.
These inputs are intentionally small, deterministic, redistributable, and
credential-free. They are the clean path for docs, tests, and examples because
they contain no licensed provider data.

### Live provider-backed workflow

The provider-backed snapshot and refresh commands may call Alpaca and FRED,
normalize provider responses, clean option quotes, and write Bronze, Silver,
Gold, and model-validation bundle artifacts under the configured local storage
root. Those outputs are local operator evidence. They prove that the library can
consume provider-derived artifacts and produce model-facing validation bundles;
they are not redistributable market-data artifacts.

Use [Marketdata CLI and private provider runs](marketdata_cli.md) for the
installed CLI commands, private evidence boundaries, and quality-policy flags.
This page keeps the credential-free synthetic fixture workflow as the public
proof path.

For notebook or CLI display, use the compact summary surface instead of parsing
raw provider files:

```python
from pathlib import Path

from option_pricing.marketdata import (
    MarketDataPipeline,
    provider_snapshot_public_summary,
)

pipeline = MarketDataPipeline(storage=Path("data-private-demo"))
result = pipeline.snapshot("SPY")
summary = provider_snapshot_public_summary(result)
print(summary)
```

The `option-pricing-marketdata snapshot --json` payload also includes
`public_summary`. It is useful for notebooks because it contains counts, policy
names, provider/feed labels, freshness shape, warning count, and main local
artifact references without provider response bodies.

### Provider bundle validation

Provider bundle validation is credential-free once the local provider artifacts
already exist. It reads the model-facing `market_data.json`,
`cleaned_quotes.parquet`, and `heston_quotes.parquet` paths and checks that the
library can reconstruct `MarketData` and consume the Heston-compatible quote
shape. This validation is compatibility evidence, not a data redistribution
mechanism and not a production data-quality claim.

For calibration from the packaged local bundle, use the
[model-ready Heston workflow](model_ready_heston_workflow.md):
`load_model_validation_bundle(...)`, `prepare_heston_market_fit(...)`, then
`fit_heston_market(...)`, or the shorthand `fit_heston_from_bundle(...)`.

### Local-only provider artifacts

Real provider-derived outputs stay in the local evidence roots. This includes:

- Bronze provider JSON payloads
- Silver provider-normalized Parquet files
- Gold market snapshots
- model-validation bundles from real provider runs
- generated manifests from live provider runs
- real option-chain summaries, tables, or notebook exports
- local policy files that encode private dividend, rate, or data-quality
  assumptions

The default local roots `data/`, `out/`, and `data-private-demo/` are working
directories for local evidence, separate from the documented synthetic workflow.

## Synthetic vs real-provider evidence

The deterministic local snapshot remains the reproducibility baseline. It is the
right page for local fixture mechanics, Bronze/Silver/Gold artifact contracts,
quote-cleaning auditability, and model-validation bundle packaging.

The companion [Real-market provider evidence](provider_market_evidence.md) page
answers the production-shaped provider question. It publishes sanitized
provider-backed summaries for ingestion, normalization, quote cleaning,
policy capture, model-ready Heston preparation, and fit status while keeping raw
provider payloads and full provider-derived quote rows local/private.

## What this workflow does not prove

This workflow does not prove production data quality, live-provider correctness,
calibration quality, trading performance, or empirical research conclusions.

It also does not exercise provider-backed refresh commands, credential setup,
live-provider snapshot or refresh CLI paths, production CLI workflows, or
research exports. The credential-free provider bundle validation command only
checks already-written model-facing artifacts.

For the local demo workflow itself, the non-goals remain: no CLI, no provider
refresh path, and no provider refresh CLI or production CLI.

## Requirements

Run the workflow from a normal development checkout. The local validation
workflow requires either the focused marketdata extra:

```bash
pip install -e ".[marketdata]"
```

or the full contributor setup:

```bash
pip install -e ".[dev]"
```

The only runtime inputs are:

- a writable local storage root, such as `out/marketdata-demo`
- an explicit `run_id`
- the explicit local fixture root used by
  `run_local_model_validation_pipeline(...)`

No provider credentials, environment variables, network access, or live market
data accounts are required.

## Run the local demo

Use the demo script for the canonical local validation path:

```bash
python scripts/demo_local_market_validation.py --output-dir out/marketdata-demo --run-id demo-run
```

The command runs the deterministic local fixture through Bronze, Silver, Gold,
and model-validation bundle writes. Heston smoke is skipped by default so the
path stays fast and deterministic. The script passes the test fixture root
explicitly instead of relying on package data. The summary prints the key
artifact paths:

```text
Local market validation demo completed.
underlying: SYNTH
run_id: demo-run
valuation_date: 2026-05-22
bronze_manifest: ...
silver_manifest: ...
market_data: ...
heston_quotes: ...
bundle_manifest: ...
warnings: ...
heston_fit_summary: ...
No live providers or credentials were used.
```

Use `--overwrite` when rerunning the same deterministic `run_id`.

## Developer API path

The script is a thin wrapper around `run_local_model_validation_pipeline(...)`.
For direct developer use, call the API with the same local-only defaults:

```python
from pathlib import Path

from option_pricing.marketdata.bundles import ModelValidationBundleConfig
from option_pricing.marketdata.pipeline import run_local_model_validation_pipeline

result = run_local_model_validation_pipeline(
    storage=Path("out/marketdata-demo"),
    run_id="demo-run",
    fixture_root=Path("tests/marketdata/fixtures"),
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
out/marketdata-demo/
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
out/
  marketdata-demo/
    gold/
      model_validation_bundle/
        underlying=SYNTH/
          date=2026-05-22/
            run_id=demo-run/
```

Use the manifest files as the high-level pointers into each artifact set.

## Inspect the generated artifacts

Start by binding the expected bundle directory and checking that every
model-validation bundle filename exists.

```python
from pathlib import Path

bundle_dir = Path(
    "out/marketdata-demo/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=demo-run"
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
        "out/marketdata-demo/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=demo-run/manifest.json"
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
        "out/marketdata-demo/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=demo-run/warnings.json"
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
embedding. The default demo fixture is clean-only, so its rejected file may be
empty; the dedicated `local_snapshot_synth_with_rejections_v1` fixture and
pipeline tests prove that rejected rows and `reason_counts` flow end to end.

```python
from pathlib import Path

import pandas as pd

bundle_dir = Path(
    "out/marketdata-demo/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=demo-run"
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

### Inspect all-quotes-rejected snapshots

If every option quote is rejected, the pipeline still writes the local audit
artifacts instead of failing after partial output. Reviewers should expect:

- `market_data.json`, `manifest.json`, `warnings.json`, and
  `heston_fit_summary.csv` to be present
- empty schema-valid `cleaned_quotes.parquet`, `heston_quotes.parquet`, and
  `surface_inputs.parquet`
- non-empty `rejected_quotes.parquet` with inspectable row-level details
- manifest row counts with zero cleaned/Heston/surface rows and non-zero
  rejected rows
- `manifest["heston_smoke"]["status"] == "skipped"`
- a Heston smoke message saying no cleaned quotes are available

This behavior keeps bad local snapshots auditable without treating them as
successful calibration evidence.

### Inspect surface inputs

`surface_inputs.parquet` is the stable surface-building seed derived from
cleaned quotes. It should preserve the cleaned quote count and expose only the
surface input schema needed by downstream surface construction.

```python
from pathlib import Path

import pandas as pd

bundle_dir = Path(
    "out/marketdata-demo/gold/model_validation_bundle/"
    "underlying=SYNTH/date=2026-05-22/run_id=demo-run"
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
reconstruct model inputs from cleaned local quotes. Treat it as a candidate
artifact: it proves the saved rows have the Heston-compatible shape, not that
every row is calibration-ready. The
[model-ready Heston workflow](model_ready_heston_workflow.md) uses
`prepare_heston_market_fit(...)` to select and reject rows before fitting.

```python
from pathlib import Path

import pandas as pd

heston_quotes = pd.read_parquet(
    Path(
        "out/marketdata-demo/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=demo-run/heston_quotes.parquet"
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
        "out/marketdata-demo/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=demo-run/heston_fit_summary.csv"
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
        "out/marketdata-demo/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=demo-run/manifest.json"
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
        "out/marketdata-demo/gold/model_validation_bundle/"
        "underlying=SYNTH/date=2026-05-22/run_id=demo-run/market_data.json"
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
source labels, policy metadata, run ID, and `library_commit` when provided. The
JSON should stay local-only and credential-free.

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
  sources, policy metadata, run ID, and library commit if provided.
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

## Current scope and exclusions

Component responsibilities are intentionally narrow:

- Silver normalization and quote cleaning produce accepted quotes, rejected
  quotes, reason counts, warnings, and a cleaning manifest.
- Gold conversion writes `MarketData` reload evidence and Heston-compatible
  quote artifacts.
- The model-validation bundle packages the local artifacts into one
  self-contained validation directory.
- The local validation path documents reproducibility, expected artifacts, and
  limitation boundaries.

The local demo remains deterministic and local-only. It does not call providers,
run provider refresh logic, execute the provider-backed CLI, or create research
exports.

The current workflow intentionally excludes:

- no live providers in this local workflow
- no credential use in this local workflow
- no CLI
- no provider refresh path
- no provider refresh CLI or production CLI
- no production data-quality claim
- no trading-performance claim
- no research exports

Gold artifacts, Heston-compatible reconstruction, and bundle packaging remain
inside the local-only and credential-free boundary.

## Bronze, Silver, Gold, and bundle layers

Bronze preserves the local fixture evidence as loaded from the deterministic
snapshot. It records the fixture identity and writes the raw local
`market_inputs` and `option_chain` frames.

Silver converts that fixture evidence into normalized `market_inputs`, cleaned
quote rows, rejected quote rows, reason counts, warnings, and a cleaning
manifest.

Gold converts Silver outputs into library-ready artifacts. It writes
`market_data.json` for `MarketData` reload and `heston_quotes.parquet` using the
existing Heston quote column contract. `PricingContext` is reconstructed through
`MarketData.to_context()`; it is not serialized directly.

The model-validation bundle packages the same local contracts. It collects the
reloaded market data payload, cleaned quotes, rejected quotes, Heston-compatible
quotes, surface inputs, warnings, and a minimal Heston smoke summary into one
self-contained local bundle.

To fit Heston from that bundle, follow the
[model-ready Heston workflow](model_ready_heston_workflow.md) instead of
manually reading `market_data.json` and `heston_quotes.parquet` in notebook
code.

## Quote-cleaning policy

Quote cleaning is deterministic. The default `QuoteCleaningPolicyV1` values are:

- `max_relative_spread=None`
- `intrinsic_tolerance=1e-8`
- `require_iv=False`
- `require_vega=False`
- `day_count="ACT/365"`

Each rejected row receives one primary rejection reason in policy order. Rejected
quotes remain first-class evidence and are written to `rejected_quotes.parquet`
in Silver and in the model-validation bundle.

The provider-backed policy metadata names the staged layers explicitly:

- `raw_option_quotes` preserves provider rows as close to raw as practical
- `clean_option_quotes` contains market-sane recoverable quotes
- `model_validation_quotes` is the stricter model-ready subset

Clean quotes may still be missing IV, Greeks, moneyness, or other derived
fields. The cleaner computes `mid`, `spread`, `relative_spread`,
`time_to_expiry_years`, `moneyness`, `log_moneyness`, and
`option_price_for_model` when the needed inputs are available, and records
readiness flags such as `model_validation_ready`, `iv_validation_ready`, and
`greek_validation_ready`.

Rejected quote rows must not be duplicated into manifest JSON. Manifests may
record rejected row counts, reason counts, warnings, and artifact filenames, but
the row-level evidence stays in Parquet artifacts.

The local cleaning conventions are:

- `expiry_years` uses `ACT/365`
- date-only expiry is interpreted as midnight UTC
- `moneyness = strike / spot`
- `spot` comes from normalized `market_inputs`
- `relative_spread = (ask - bid) / mid`

Primary rejection reasons are:

- `unparseable_contract`
- `bad_expiry`
- `expired_contract`
- `nonpositive_mid`
- `nonpositive_strike`
- `negative_bid`
- `negative_ask`
- `crossed_bid_ask`
- `quote_after_asof`
- `stale_quote`
- `missing_price_source`
- `missing_spot_for_moneyness`
- `missing_rate_for_model`
- `missing_dividend_for_model`
- `missing_time_to_expiry_for_model`
- `missing_iv_for_iv_validation`
- `unsupported_option_right`
- `nonfinite_numeric_field`
- `vanilla_no_arbitrage_violation`
- `nonstandard_or_adjusted_contract`
- `spot_option_chain_mismatch`

## Model-validation bundle

The model-validation bundle is written under `gold/model_validation_bundle/...`
for the same `underlying`, `date`, and `run_id` as the Bronze, Silver, and Gold
artifacts. The frozen bundle artifact names are:

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

When all cleaned quotes are absent, the bundle writes empty schema-valid Heston
and surface artifacts and records Heston smoke as skipped. The smoke result is
compatibility evidence only, not calibration-quality evidence.

## Assumptions and limitations

The workflow assumes the local fixture snapshot is the evidence source under
review. It is suitable for checking deterministic artifact mechanics and
library-facing compatibility, not for evaluating live market data feeds.

The bundle may include a Heston smoke result, but that result is a packaging and
compatibility signal only. It is not a claim about production calibration
quality, model fitness, strategy performance, or empirical validity.

No live providers, credentials, provider-backed CLI execution, or research
exports are part of this local demo workflow.

## Developer checks

Run the local quality and marketdata checks from the repository root:

```powershell
ruff check .
black --check .
mypy
pytest -q tests/marketdata/test_local_snapshot_provider.py
pytest -q tests/marketdata/test_quote_cleaning.py
pytest -q tests/marketdata/test_gold_conversions.py
pytest -q tests/marketdata/test_model_validation_bundle.py
pytest -q tests/test_packaging_metadata.py
```

For rendered documentation review, serve the MkDocs site locally and inspect the
page in light and dark theme at 375, 768, 1280, and 1536 pixel widths.
