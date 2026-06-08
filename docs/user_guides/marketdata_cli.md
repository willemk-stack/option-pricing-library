# Marketdata CLI And Private Provider Runs

This guide explains how the installed `option-pricing-marketdata` CLI fits into
the public proof path and the private provider-backed evidence path.

Use it after the credential-free
[market snapshot validation](market_snapshot_validation.md) page. That page is
the redistributable proof path. The CLI provider commands are local/private
operator tools for writing artifacts from Alpaca and FRED, then checking that
the saved model-facing files can be consumed by the library.

## The Three Paths

| Path | Use | Boundary |
| --- | --- | --- |
| Credential-free proof path | `python scripts/demo_local_market_validation.py --output-dir out/marketdata-demo --run-id demo-run` | Synthetic local fixtures, no live providers, no credentials, redistributable docs/tests evidence. |
| Private provider-backed evidence path | `option-pricing-marketdata snapshot`, `refresh-daily`, `backfill-fred`, and `backfill-bars` | Alpaca/FRED-backed local artifacts under operator-owned roots; useful for private validation, screenshots, and tables, but not redistributable. |
| Official model-ready Heston workflow | `load_model_validation_bundle(...) -> prepare_heston_market_fit(...) -> fit_heston_market(...)`, or `fit_heston_from_bundle(...)` | The public Python workflow for turning a saved local bundle into prepared Heston inputs and a fit result. |

The provider-backed commands do not replace the public proof path. They extend
the branch with a private evidence route that can produce the same saved bundle
shape used by the model-ready Heston workflow.

## Install

Install the marketdata extra from a local checkout:

```bash
pip install -e ".[marketdata]"
```

The provider-backed commands read credentials from environment variables:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `FRED_API_KEY`

Do not commit generated provider artifacts, credentials, policy files with
private assumptions, or licensed market data.

## CLI Commands

Inspect the command surface:

```bash
option-pricing-marketdata --help
```

Write one local provider-backed snapshot:

```bash
option-pricing-marketdata snapshot \
  --underlying SPY \
  --asof 2026-05-22T15:31:00Z \
  --data-root out/marketdata-live \
  --library-commit "$(git rev-parse HEAD)"
```

Refresh several underlyings into one aggregate run:

```bash
option-pricing-marketdata refresh-daily \
  --underlyings SPY QQQ \
  --asof 2026-05-22T15:31:00Z \
  --run-id-prefix daily-close \
  --data-root out/marketdata-live
```

Backfill FRED observations:

```bash
option-pricing-marketdata backfill-fred \
  --series DGS3MO DGS1 \
  --start 2026-05-01 \
  --end 2026-05-22 \
  --data-root out/marketdata-live
```

Backfill Alpaca equity bars:

```bash
option-pricing-marketdata backfill-bars \
  --symbols SPY QQQ \
  --start 2026-05-20 \
  --end 2026-05-23 \
  --timeframe 1Day \
  --data-root out/marketdata-live
```

Validate saved model-facing files without live providers:

```bash
option-pricing-marketdata validate-bundle \
  --market-data out/marketdata-live/gold/market_snapshot/underlying=SPY/date=2026-05-22/run_id=<run-id>/market_data.json \
  --cleaned-quotes out/marketdata-live/silver/cleaned_quotes/underlying=SPY/date=2026-05-22/run_id=<run-id>/cleaned_quotes.parquet \
  --heston-quotes out/marketdata-live/gold/heston_quotes/underlying=SPY/date=2026-05-22/run_id=<run-id>/heston_quotes.parquet
```

Use `--json` when you need stable automation output. JSON payloads are intended
to expose counts, policy metadata, warnings, sanitized diagnostics, and artifact
paths without provider response bodies or secret values.

## Public Demo Safety

Safe public demos:

- run `scripts/demo_local_market_validation.py` with the synthetic fixtures
- show the generated synthetic bundle tree and manifest fields
- run `option-pricing-marketdata validate-bundle` against synthetic or already
  sanitized local model-facing files
- show `load_model_validation_bundle(...)`, `prepare_heston_market_fit(...)`,
  and `fit_heston_from_bundle(...)` on redistributable synthetic artifacts

Keep local/private:

- provider Bronze payloads and raw provider response evidence
- provider-normalized Silver and Gold artifacts
- provider-backed model-validation bundles
- local policy files with private dividend, rate, or quality assumptions
- screenshots or tables that include licensed market data unless you have a
  separate redistribution right

## Quality And Heston Knobs

For provider-backed model evidence, start with a warning-only run and inspect
the output before enforcing stricter filters:

```bash
option-pricing-marketdata snapshot \
  --underlying SPY \
  --max-equity-quote-age-seconds 900 \
  --max-option-quote-age-seconds 1800 \
  --data-root out/marketdata-live \
  --json
```

Then tighten the quality policy only when the validation goal calls for it:

```bash
option-pricing-marketdata snapshot \
  --underlying SPY \
  --quote-freshness-mode intraday_strict \
  --max-quote-age-seconds 1800 \
  --stale-quote-action reject \
  --reject-stale-option-quotes \
  --reject-option-quotes-after-asof \
  --min-accepted-contracts 50 \
  --min-accepted-calls 20 \
  --min-accepted-puts 20 \
  --min-expiries 2 \
  --run-heston-smoke \
  --data-root out/marketdata-live
```

`--run-heston-smoke` records compatibility smoke evidence inside the saved
bundle. It is not a production calibration-quality claim. The official fitting
path remains the [model-ready Heston workflow](model_ready_heston_workflow.md),
where `prepare_heston_market_fit(...)` selects and rejects calibration rows
before `fit_heston_market(...)` runs.

## Current Non-Goals

- no new provider beyond the existing Alpaca/FRED-backed paths
- no historical option-chain backfill
- no scheduler, daemon, or live-provider CI
- no redistribution of provider-backed artifacts
- no production trading readiness claim
- no guarantee that a provider-backed Heston fit is universally valid

Use the [Market APIs](market_api.md) page for the lower-level market input API
context, and use [Market snapshot validation](market_snapshot_validation.md) for
the deterministic local artifact proof.

## Publishing sanitized provider evidence

The provider refresh path can produce local/private provider artifacts, but the
public documentation bundle should publish only sanitized summaries. Raw provider
payloads, full provider-derived quote rows, credentials, tokens, and response
bodies stay local/private.

After a provider-backed model-validation bundle has been written locally, build
the public evidence bundle with:

```bash
python scripts/build_provider_evidence_artifacts.py \
  --bundle-root out/marketdata-live/gold/model_validation_bundle/underlying=SPY/date=YYYY-MM-DD/run_id=<run-id> \
  --provider-summary-json out/marketdata-live/provider_public_summary.json \
  --output-dir docs/assets/generated/provider_evidence \
  --profile release
```

Publish these generated summaries on Pages:

- provider/feed labels, as-of timestamp, underlying, and run ID
- raw, normalized, accepted, rejected, and selected counts
- rejection reason counts and stage summaries
- expiry, strike, moneyness, call, and put coverage summaries
- rate, dividend, quality, and data-policy names
- model-ready selection/rejection counts and preflight status
- Heston fit status, objective, best cost, parameter summary, warning count

Keep these artifacts local/private:

- raw Alpaca latest equity quote payloads
- raw Alpaca option-chain JSON payloads
- full provider-normalized `option_chain.parquet`
- full provider-derived `cleaned_quotes.parquet`
- full `heston_quotes.parquet` if it reconstructs provider quote rows
- screenshots or tables exposing raw bid/ask rows
- credentials, private policy files, provider response bodies, tokens, and
  authorization headers

Use [Real-market provider evidence](provider_market_evidence.md) for the
reviewer-facing public proof page.

