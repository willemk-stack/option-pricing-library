# Market APIs

<div class="doc-intro doc-intro--quiet" markdown="1">
<p class="doc-intro__kicker">Market inputs</p>
<p class="doc-intro__lead">The library supports two related ways to describe markets: a flat convenience layer and a curves-first layer.</p>
<p class="doc-intro__support">Use the flat path for compact examples and the curves-first path when discounting and forwards should stay explicit.</p>
</div>

The library supports two related ways to describe markets:

1. **Flat convenience API** using `MarketData` and `PricingInputs`
2. **Curves-first API** using `PricingContext`, `DiscountCurve`, and `ForwardCurve`

Both are valid. The choice is mostly about how explicit you want to be about term structure.

## Flat convenience API

This is the fastest path for examples, tests, and simple workflows.

```python
from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price

market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.02)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

p = PricingInputs(spec=spec, market=market, sigma=0.20, t=0.0)
print(bs_price(p))
```

`MarketData` also gives you small helpers:

```python
print(market.df(1.0))
print(market.fwd(1.0))
```

Important time convention:

- `MarketData.df(T, t=...)` and `MarketData.fwd(T, t=...)` work with absolute times
- `PricingInputs` interprets `OptionSpec.expiry` as the absolute expiry `T`
- `PricingInputs.tau` converts from `(t, T)` to time-to-expiry automatically

## Convert flat inputs into curves-first inputs

If you already have `MarketData`, the bridge is built in:

```python
ctx = market.to_context()
print(ctx.df(1.0))
print(ctx.fwd(1.0))
```

For flat rates and carry, the resulting `PricingContext` is exactly consistent with `MarketData`.

## Curves-first API

This is the better fit when you want the pricing code to consume discount and forward curves directly.

```python
from option_pricing import FlatDiscountCurve, FlatCarryForwardCurve, PricingContext

discount = FlatDiscountCurve(r=0.05)
forward = FlatCarryForwardCurve(spot=100.0, r=0.05, q=0.02)
ctx = PricingContext(spot=100.0, discount=discount, forward=forward)
```

With a `PricingContext`, you call the `*_from_ctx` pricers and pass `tau` directly:

```python
from option_pricing import OptionType, bs_price_from_ctx

price = bs_price_from_ctx(
    kind=OptionType.CALL,
    strike=100.0,
    sigma=0.20,
    tau=1.0,
    ctx=ctx,
)
```

## When to use which API

If you are following the recommended instrument-based workflow, you still choose between `MarketData` and `PricingContext` for the market inputs.

Use `MarketData` + `PricingInputs` when:

- you want the most compact examples
- rates and dividend yield are flat
- you are teaching, testing, or prototyping

Use `PricingContext` when:

- you already think in terms of discount factors and forwards
- you want to plug in term structures
- you want pricing code that does not depend on flat `(r, q)` assumptions at the interface level

## Matching results between the two APIs

For the same flat market assumptions, these should agree:

```python
from option_pricing import bs_price

price_inputs = bs_price(p)
price_ctx = bs_price_from_ctx(
    kind=p.spec.kind,
    strike=p.spec.strike,
    sigma=p.sigma,
    tau=p.tau,
    ctx=p.market.to_context(),
)
```

The same pattern also applies to:

- `bs_greeks_from_ctx`
- `mc_price_from_ctx`
- `binom_price_from_ctx`

## Provider-backed marketdata current scope

The current provider-backed marketdata interface supports one-shot snapshots,
multi-underlying daily refreshes, and two simple backfill artifact sets through
`MarketDataPipeline`. The installable CLI and the compatibility wrapper both
stay thin: they parse arguments and dispatch to the pipeline, while the
pipeline handles provider calls, normalization, quote cleaning, Bronze/Silver
storage, Gold conversion, and the model-validation bundle.

Install the marketdata extra before using provider-backed commands:

```bash
pip install -e ".[marketdata]"
```

Provider-backed runs read credentials from environment variables:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `FRED_API_KEY`

The installed console entry point is:

```bash
option-pricing-marketdata --help
```

The existing wrapper remains available for repo-local usage:

```bash
python scripts/fetch_market_snapshot.py --help
```

Fetch one market snapshot:

```bash
option-pricing-marketdata snapshot --underlying SPY --asof 2026-05-22T15:31:00Z --data-root out/marketdata-live
```

Equity and option data expose separate provider/feed metadata. Use
`--equity-feed` for stock quotes and bars and `--option-feed` for option chains.
The legacy `--feed` flag remains as a deprecated shortcut: snapshot and
refresh commands treat it as an option-feed override, while equity-bar backfills
treat it as an equity-feed override. Snapshot artifacts also record
`equity_provider`, `equity_feed`, `option_provider`, and `option_feed` instead
of relying on one ambiguous top-level feed field.

Use explicit rate, curve, provenance, and freshness controls when preparing a
controlled real-data validation run:

```bash
option-pricing-marketdata snapshot \
  --underlying SPY \
  --rate-lookback-days 45 \
  --curve-series DGS1MO DGS3MO DGS6MO DGS1 \
  --quote-freshness-mode intraday_strict \
  --max-quote-age-seconds 1800 \
  --stale-quote-action reject \
  --max-equity-quote-age-seconds 900 \
  --max-option-quote-age-seconds 1800 \
  --reject-stale-option-quotes \
  --reject-option-quotes-after-asof \
  --min-accepted-contracts 50 \
  --min-accepted-calls 20 \
  --min-accepted-puts 20 \
  --min-expiries 2 \
  --library-commit "$(git rev-parse HEAD)" \
  --data-root out/marketdata-live
```

Pass `--no-rate-curve` to skip the curve artifact when only the selected flat
rate is needed. Pass `--run-heston-smoke` to enable the lightweight Heston
smoke check inside the model-validation bundle.

For a first real provider-backed model run, start warning-only:

```bash
option-pricing-marketdata snapshot \
  --underlying SPY \
  --max-equity-quote-age-seconds 900 \
  --max-option-quote-age-seconds 1800 \
  --library-commit "$(git rev-parse HEAD)" \
  --data-root out/marketdata-live
```

Inspect `quote_freshness`, provider diagnostics, accepted/rejected counts, and
warnings in the JSON output or manifests. Then rerun with stricter rejection
controls such as `--reject-stale-option-quotes`,
`--reject-option-quotes-after-asof`, and the minimum accepted quote flags.

Refresh multiple underlyings with one aggregate run:

```bash
option-pricing-marketdata refresh-daily --underlyings SPY QQQ --asof 2026-05-22T15:31:00Z --run-id-prefix daily-close --data-root out/marketdata-live
```

Backfill FRED observations:

```bash
option-pricing-marketdata backfill-fred --series DGS3MO --start 2026-05-01 --end 2026-05-22 --data-root out/marketdata-live
```

Backfill Alpaca equity bars:

```bash
option-pricing-marketdata backfill-bars --symbols SPY QQQ --start 2026-05-20 --end 2026-05-23 --timeframe 1Day --data-root out/marketdata-live
```

Override the documented dividend assumption for one run:

```bash
option-pricing-marketdata snapshot --underlying SPY --dividend-yield 0.0125 --dividend-yield-source manual_override --data-root out/marketdata-live
```

For repeatable per-symbol dividend assumptions, use a local policy config and
pass it with `--policy-config`:

```json
{
  "dividends": {
    "default_policy": "zero_assumption",
    "static_yields": {
      "SPY": {
        "dividend_yield": 0.0125,
        "source": "manual_static",
        "note": "Approximate annualized dividend yield for validation"
      },
      "TSLA": {
        "dividend_yield": 0.0,
        "source": "manual_static",
        "note": "Explicit no-dividend assumption"
      }
    }
  }
}
```

`manual_static` permits non-zero yields and explicit zero yields. An absent
symbol falls back to `zero_assumption`. The recognized future policies
`provider_trailing_yield` and `implied_carry` are intentionally unsupported in
this pass: the former is a backward-looking provider dividend proxy, and the
latter is a research/diagnostic artifact inferred from option prices.

Emit stable JSON output for automation:

```bash
option-pricing-marketdata refresh-daily --underlyings SPY QQQ --json --data-root out/marketdata-live
```

JSON output includes the snapshot warnings, quality policy, quote freshness
statistics, and sanitized provider operation diagnostics. Human-readable output
stays concise by default.

Documented assumptions and current provider scope are written as warnings or
manifest notes:

- `dividend_yield` defaults to `0.0` with policy `zero_assumption`
- explicit dividend overrides and configured static yields are recorded as
  `manual_static`
- rate policy is `fred_treasury_zero_proxy_linear_cc`
- the rate curve is a FRED Treasury zero-rate proxy with linear interpolation
  on continuously compounded rates
- FRED percentage observations are converted to decimal rates and then to
  continuous rates with `log(1 + r)`
- the curve uses `DGS1MO`, `DGS3MO`, `DGS6MO`, `DGS1`, and `DGS2` when
  available; if fewer than two points are usable, the selected flat FRED rate is
  used as a fallback and recorded
- the proxy curve is not a bootstrapped zero curve; manifests set
  `rate_is_bootstrapped=false`
- no option-chain historical backfill yet
- scheduling, cron, and background refresh services are not enabled
- option cleaning records `staged_recoverable_quotes_v1` policy metadata and
  preserves rejected rows with stable reason codes such as
  `missing_price_source`, `crossed_bid_ask`, `negative_bid`, `negative_ask`,
  `quote_after_asof`, and `stale_quote`
- recoverable option rows are not dropped merely because IV, Greeks,
  moneyness, or model-derived fields are missing
- local provider artifacts under `data/` or `out/marketdata-live/` are
  operator-owned evidence and should not be committed with credentials or
  secrets

Provider diagnostics are written to snapshot manifests and run metadata. They
record provider, operation, status, sanitized request metadata, elapsed time,
retry count, and safe failure metadata. Failed calls expose `failure_kind` and
`sanitized_message` rather than raw exception text. API keys, secret keys,
bearer tokens, authorization headers, passwords, and raw secret values are not
written to text artifacts.

The provider snapshot freshness policy records:

- `quote_freshness_mode`: `demo_lenient`, `end_of_day`, or `intraday_strict`
- `max_quote_age_seconds`, when configured
- equity quote age in seconds
- option quote age min/median/max in seconds
- `quote_age_summary`
- option quotes after the requested `asof`
- stale option quote, stale accepted quote, and stale quote counts
- quote freshness warnings
- accepted call/put/expiry counts

`demo_lenient` warns about stale rows and preserves otherwise usable quotes.
`end_of_day` can allow latest prior-session data with
`--allow-prior-session`, while still recording age and warning metadata.
`intraday_strict` rejects or fails stale rows according to
`--stale-quote-action` and `--max-quote-age-seconds`.
Use `--reject-stale-option-quotes` with `--max-option-quote-age-seconds` to move
stale accepted options into `rejected_quotes` with reason `stale_quote`. Option
quotes with `quote_ts > asof` are rejected from clean quotes with reason
`quote_after_asof`.
Minimum-shape flags `--min-accepted-contracts`, `--min-accepted-calls`,
`--min-accepted-puts`, and `--min-expiries` fail the snapshot when the accepted
set is too small for the intended model run. Use `--reject-stale-equity-quote`
to fail when the equity quote violates the freshness or on-or-before-`asof`
policy.

Provider-backed snapshots keep three option quote layers distinct:

- `raw_option_quotes` preserves provider rows as close to raw as practical
- `clean_option_quotes` keeps market-sane recoverable quotes and computes
  fields such as `mid`, `spread`, `relative_spread`, `time_to_expiry_years`,
  `moneyness`, `log_moneyness`, and `option_price_for_model` when possible
- `model_validation_quotes` is the stricter model-ready subset; provider IV and
  Greeks are not required unless the validation target specifically needs them

Validate model-facing artifacts after a provider-backed snapshot with the
credential-free helper:

```bash
option-pricing-marketdata validate-bundle \
  --market-data out/marketdata-live/gold/market_snapshot/underlying=SPY/date=2026-05-22/run_id=<run-id>/market_data.json \
  --cleaned-quotes out/marketdata-live/silver/cleaned_quotes/underlying=SPY/date=2026-05-22/run_id=<run-id>/cleaned_quotes.parquet \
  --heston-quotes out/marketdata-live/gold/heston_quotes/underlying=SPY/date=2026-05-22/run_id=<run-id>/heston_quotes.parquet
```

The command calls `validate_provider_snapshot_bundle(...)` and verifies that
`market_data.json`, `cleaned_quotes.parquet`, and `heston_quotes.parquet` can be
read by the library-facing contracts. Add `--json` for stable automation output.

Snapshot output includes the documented rate/dividend source metadata, explicit
equity/option provider and feed metadata, policy metadata, rate series, raw and
normalized option contract counts when available, dropped/provider-rejected
counts, accepted/rejected counts, warnings, and the main artifact paths. The
`refresh-daily` aggregate result summarizes those same counts across all
requested underlyings and records one aggregate run alongside the child snapshot
runs.

Normal tests use mocked providers and do not require credentials. Optional live
smoke tests are skipped unless all three credentials, `alpaca-py`, and `pyarrow`
are available in the local environment:

```bash
pytest -q tests/marketdata/test_provider_confidence_checks.py::test_live_provider_snapshot_smoke_optional
```

The live smoke path runs a narrow SPY snapshot into a temporary local storage
root and checks the same documented assumptions, current provider scope, key
artifacts, accepted quote count, provider diagnostics, freshness warnings, and
text-artifact secret hygiene. Provider-backed snapshots are suitable for
controlled real-data validation; they are not presented as production-grade
market data, scheduling, or historical option-chain backfill.

## Related guides

- [Quickstart](quickstart.md)
- [Instruments](instruments.md)
- [Black-Scholes](black_scholes.md)
