# Feature Data Providers Release Notes

This note summarizes the merge-ready scope of the `Feature/data-providers`
branch for maintainers and reviewers.

## Adds

- Public marketdata exports for loading model-validation bundles, preparing
  Heston market-fit inputs, running the marketdata pipeline, validating saved
  provider snapshot bundles, and reading provider snapshot result summaries.
- Public workflow exports for `fit_heston_market(...)` and
  `fit_heston_from_bundle(...)`.
- Provider-backed snapshot, daily refresh, FRED backfill, equity-bar backfill,
  and bundle-validation CLI plumbing through `option-pricing-marketdata`.
- Deterministic local fixture validation that writes Bronze, Silver, Gold, and
  model-validation bundle artifacts without live providers or credentials.
- Documentation tying the local proof path, private provider-backed evidence,
  and official bundle-to-prepare-to-fit Heston workflow together.

## Local And Private Boundaries

- Synthetic fixture artifacts are redistributable proof-path evidence.
- Alpaca/FRED-backed outputs are local operator evidence and should stay out of
  the repository unless separately sanitized and permitted.
- Provider-backed screenshots or tables are private validation material unless
  the underlying data license permits redistribution.
- Live-provider CI remains out of scope; normal tests use synthetic/static or
  mocked-provider checks.

## Supported Public APIs

- `option_pricing.marketdata.load_model_validation_bundle`
- `option_pricing.marketdata.prepare_heston_market_fit`
- `option_pricing.marketdata.MarketDataPipeline`
- `option_pricing.marketdata.validate_provider_snapshot_bundle`
- `option_pricing.marketdata.provider_snapshot_public_summary`
- `option_pricing.workflows.fit_heston_market`
- `option_pricing.workflows.fit_heston_from_bundle`

## Non-Goals

- no new providers, pricing models, or generic model registry
- no Heston bounds widening or hidden quote filtering
- no historical option-chain backfill, scheduler, or live-provider CI
- no production trading, production data-quality, or universal calibration
  claims
- no redistribution of private provider artifacts or licensed market data

## Future Work

- optional provider-backed quality reports that can be generated locally without
  committing private artifacts
- documented policy presets for repeatable private validation runs
- broader provider confidence checks that remain opt-in and credential-gated
