# Local Snapshot SYNTH Schema V1

This fixture is synthetic, provider-neutral, and schema-only. It does not
represent historical SYNTH, SPY, or any other real market data.

A2-S1 only proves that a deterministic local snapshot fixture can be discovered,
loaded, parsed, and validated against the A1 marketdata schema helpers. A3 owns
normalization and quote cleaning. A4/A5 own Gold conversion, Heston-compatible
artifacts, and model-validation bundles.

Rates and dividend yields are annualized decimals. The rate is already
continuously compounded. The day-count convention is ACT/365. Dividend yield is
zero by explicit assumption.

The fixture contains clean synthetic rows only. It does not include invalid
quotes, provider payloads, credentials, live-provider metadata, or historical
market claims.
