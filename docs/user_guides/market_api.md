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

Override the documented dividend assumption:

```bash
option-pricing-marketdata snapshot --underlying SPY --dividend-yield 0.0125 --dividend-yield-source manual_override --data-root out/marketdata-live
```

Emit stable JSON output for automation:

```bash
option-pricing-marketdata refresh-daily --underlyings SPY QQQ --json --data-root out/marketdata-live
```

Documented assumptions and current provider scope are written as warnings or
manifest notes:

- `dividend_yield` defaults to `0.0` with source `assumption`
- the default rate series is FRED `DGS3MO`
- rate selection uses one FRED series observation; curve interpolation is not
  enabled yet
- dividend inference is not enabled yet
- no option-chain historical backfill yet
- scheduling, cron, and background refresh services are not enabled
- Alpaca option contracts without usable bid/ask may be dropped before quote
  cleaning; snapshot warnings include the raw, normalized, and dropped counts

Snapshot output includes the documented rate/dividend source metadata, feed,
rate series, raw and normalized option contract counts when available,
dropped/provider-rejected counts, accepted/rejected counts, warnings, and the
main artifact paths. The `refresh-daily` aggregate result summarizes those same
counts across all requested underlyings and records one aggregate run alongside
the child snapshot runs.

Normal tests use mocked providers and do not require credentials. Live provider
smoke tests are optional and are skipped unless the relevant credentials and
provider SDK/runtime dependency are present.

## Related guides

- [Quickstart](quickstart.md)
- [Instruments](instruments.md)
- [Black-Scholes](black_scholes.md)
