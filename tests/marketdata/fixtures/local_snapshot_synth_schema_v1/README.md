# Local Snapshot SYNTH Schema V1

This fixture is synthetic, provider-neutral, and schema-only. It does not
represent historical SYNTH, SPY, or any other real market data.

A2-S1 only proves that a deterministic local snapshot fixture can be discovered,
loaded, parsed, and validated against the A1 marketdata schema helpers. A3 owns
normalization and quote cleaning. A4/A5 own Gold conversion, Heston-compatible
artifacts, and model-validation bundles.

A2-S3 writes one auditable Bronze `local_snapshot` bundle for this fixture. The
bundle contains `manifest.json`, `market_inputs.parquet`, and
`option_chain.parquet` together under the same `underlying`/`date`/`run_id`
partition. These Parquet files are raw-ish, schema-compatible local snapshot
inputs. A2-S3 does not clean, normalize, reject quotes, price options, or
produce Silver or Gold artifacts.

Load the fixture through the local provider:

```python
from option_pricing.marketdata.providers.local import LocalSnapshotProvider

result = LocalSnapshotProvider().load_snapshot("local_snapshot_synth_schema_v1")
```

Write the Bronze evidence bundle by providing a `run_id` and local storage root:

```python
from option_pricing.marketdata.config import StorageConfig
from option_pricing.marketdata.providers.local import (
    LocalSnapshotConfig,
    LocalSnapshotProvider,
    write_local_snapshot_bronze,
)
from option_pricing.marketdata.storage import LocalStorage

provider = LocalSnapshotProvider(
    LocalSnapshotConfig(fixture_name="local_snapshot_synth_schema_v1", run_id="run-001")
)
result = provider.load_snapshot()
storage = LocalStorage(StorageConfig(root="out/marketdata"))
paths = write_local_snapshot_bronze(storage, result)
```

The returned `LocalSnapshotResult` is the A3 handoff. A3 should consume
`fixture_name`, `snapshot_id`, `run_id`, `underlying`, `asof`, `manifest`,
`market_inputs_raw`, `option_chain_raw`, `metadata`, `row_counts`, and
`warnings`, then produce any normalized, cleaned, rejected, pricing, Silver, or
Gold outputs in later code.

Rates and dividend yields are annualized decimals. The rate is already
continuously compounded. The day-count convention is ACT/365. Dividend yield is
zero by explicit assumption.

The fixture contains clean synthetic rows only. It does not include invalid
quotes, provider payloads, credentials, live-provider metadata, or historical
market claims.
