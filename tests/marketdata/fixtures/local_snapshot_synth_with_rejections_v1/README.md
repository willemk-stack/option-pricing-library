# Local Snapshot SYNTH With Rejections V1

This fixture is synthetic, provider-neutral, local-only, and intentionally
small. It contains one valid quote and one crossed-market quote so integration
tests can prove rejected quote evidence, reason counts, and manifest summaries
flow through Silver, Gold, and the model-validation bundle.

It does not represent live or historical market data and does not require
provider credentials, network access, or refresh workflows.
