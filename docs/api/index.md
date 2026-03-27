# API

<div class="doc-intro doc-intro--quiet" markdown="1">
<p class="doc-intro__kicker">Reference map</p>
<p class="doc-intro__lead">Start here when you need the public surface, but choose the interface style first: instrument-based for the default workflow, flat inputs for compact scripts, or curves-first for explicit term structures.</p>
<p class="doc-intro__support">This page stays quiet on purpose. Use it to choose a path, then drill into the dedicated reference pages or the snippets below.</p>
</div>

## Choose an interface

<p class="doc-section-lead">Most callers should start with reusable instruments. The other two paths stay available when compact examples or explicit market curves matter more than the contract object.</p>

- **Instrument-based workflow**: the recommended public path for reusable contracts, explicit payoff semantics, and cross-method pricing.
- **Flat-input workflow**: the compact `PricingInputs` wrapper for tutorials, tests, and short scripts.
- **Curves-first workflow**: `PricingContext`, discount curves, and forward curves when the market structure should stay explicit.

## Reference pages

<p class="doc-section-lead">These pages split the namespace by responsibility so the generated API stays easy to scan.</p>

<div class="doc-card-grid doc-card-grid--quiet" markdown="1">
[<span class="doc-card__eyebrow">Top-level reference</span><span class="doc-link-card__title">Public API</span><span class="doc-link-card__copy">Root-level types, instruments, configuration objects, and common shared exports.</span>](public.md){ .doc-link-card .doc-link-card--quiet }

[<span class="doc-card__eyebrow">Term structures</span><span class="doc-link-card__title">Curves-first API</span><span class="doc-link-card__copy"><code>PricingContext</code>, discount curves, and forward curves for explicit market structure.</span>](curves.md){ .doc-link-card .doc-link-card--quiet }

[<span class="doc-card__eyebrow">Execution layer</span><span class="doc-link-card__title">Pricers</span><span class="doc-link-card__copy">Black-Scholes, Monte Carlo, and Binomial entry points across all three public styles.</span>](pricers.md){ .doc-link-card .doc-link-card--quiet }

[<span class="doc-card__eyebrow">Volatility stack</span><span class="doc-link-card__title">Volatility</span><span class="doc-link-card__copy">Implied-vol inversion, smiles, surfaces, local-vol objects, and the eSSVI toolbox.</span>](vol.md){ .doc-link-card .doc-link-card--quiet }

[<span class="doc-card__eyebrow">Error handling</span><span class="doc-link-card__title">Exceptions</span><span class="doc-link-card__copy">Exception types to catch when validation, bracketing, or root-finding fails.</span>](exceptions.md){ .doc-link-card .doc-link-card--quiet }
</div>

## Quick snippets

<p class="doc-section-lead">The snippets below show the same pricing intent expressed through each public interface style.</p>

### Recommended API (instrument-based)

```python
from option_pricing import (
    ExerciseStyle,
    MarketData,
    OptionType,
    VanillaOption,
    bs_price_instrument,
)

inst = VanillaOption(
    expiry=1.0,
    strike=100.0,
    kind=OptionType.CALL,
    exercise=ExerciseStyle.EUROPEAN,
)

market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
price = bs_price_instrument(inst, market=market, sigma=0.2)
```

### Convenience API (flat inputs)

```python
from option_pricing import MarketData, OptionSpec, PricingInputs, OptionType, bs_price

market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
p = PricingInputs(spec=spec, market=market, sigma=0.2)

price = bs_price(p)
```

### Advanced API (curves-first)

```python
from option_pricing import (
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    OptionType,
    PricingContext,
    bs_price_from_ctx,
)

ctx = PricingContext(
    spot=100.0,
    discount=FlatDiscountCurve(r=0.03),
    forward=FlatCarryForwardCurve(spot=100.0, r=0.03, q=0.01),
)

price = bs_price_from_ctx(
    kind=OptionType.CALL,
    strike=100.0,
    sigma=0.2,
    tau=1.0,
    ctx=ctx,
)
```

### Instrument-based API (alternate view)

See the recommended snippet above, or use the [Pricers](pricers.md) page for the full list of instrument entry points.

## Notes

- Times are expressed in years.
- `PricingInputs` uses absolute expiry `T` together with valuation time `t`, and exposes `tau` as `T - t`.
- With the default `t=0`, the numeric value of `expiry` happens to equal `tau` in the flat-input examples.
- `PricingContext` and instrument-based pricers work directly with `tau` (time to expiry).
