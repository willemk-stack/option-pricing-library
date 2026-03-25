# API

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Reference map</p>
<p class="doc-intro__lead">The <code>option_pricing</code> package exposes three user-facing styles: a recommended instrument workflow, a compact flat-input path, and a curves-first path for explicit term structures.</p>
<p class="doc-intro__support">Use this page to choose the right interface first, then drill into the specific reference pages for public types, pricers, volatility objects, and exceptions.</p>
<div class="doc-pill-row">
  <span class="doc-pill">Instrument-based</span>
  <span class="doc-pill">Flat-input</span>
  <span class="doc-pill">Curves-first</span>
</div>
</div>

<div class="doc-card-grid" markdown="1">
<div class="doc-card doc-card--accent" markdown="1">
<p class="doc-card__eyebrow">Recommended</p>
<p class="doc-card__title">Instrument-based API</p>
- `VanillaOption`
- `ExerciseStyle`
- `bs_price_instrument`, `mc_price_instrument`, `binom_price_instrument`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Compact path</p>
<p class="doc-card__title">Flat convenience API</p>
- `MarketData`
- `OptionSpec`
- `PricingInputs`
- `bs_price`, `mc_price`, `binom_price`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Advanced path</p>
<p class="doc-card__title">Curves-first API</p>
- `PricingContext`
- `DiscountCurve`
- `ForwardCurve`
- `*_from_ctx` pricers
</div>
</div>

## Recommended API path

<p class="doc-section-lead">Pick the usage style based on how much structure you need in the contract and the market data.</p>

- **Recommended API**: instrument-based workflow (`VanillaOption` plus instrument pricers). This is the intended public entry point for most users.
- **Convenience API**: flat-input workflow (`PricingInputs`). Use this for compact tutorials and quick checks.
- **Advanced API**: curves-first workflow (`PricingContext`) and the volatility/PDE modules for term-structure or surface-heavy use cases.

## Overview

<p class="doc-section-lead">These pages split the surface by responsibility so the reference stays scannable.</p>

<div class="doc-card-grid" markdown="1">
[<span class="doc-card__eyebrow">Top-level reference</span><span class="doc-link-card__title">Public API</span><span class="doc-link-card__copy">Root-level types, instruments, configs, and common objects re-exported from <code>option_pricing</code>.</span>](public.md){ .doc-link-card }

[<span class="doc-card__eyebrow">Term structures</span><span class="doc-link-card__title">Curves-first API</span><span class="doc-link-card__copy"><code>PricingContext</code> plus discount and forward curves for explicit market structure.</span>](curves.md){ .doc-link-card }

[<span class="doc-card__eyebrow">Execution layer</span><span class="doc-link-card__title">Pricers</span><span class="doc-link-card__copy">Black-Scholes, Monte Carlo, and Binomial entry points across all three API styles.</span>](pricers.md){ .doc-link-card }

[<span class="doc-card__eyebrow">Volatility stack</span><span class="doc-link-card__title">Volatility</span><span class="doc-link-card__copy">Implied-vol inversion, smiles, surfaces, local-vol objects, and the eSSVI toolbox.</span>](vol.md){ .doc-link-card }

[<span class="doc-card__eyebrow">Error handling</span><span class="doc-link-card__title">Exceptions</span><span class="doc-link-card__copy">Exception types you may want to catch when validation or root-finding fails.</span>](exceptions.md){ .doc-link-card }
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
