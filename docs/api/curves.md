# Curves-first API

<div class="doc-intro doc-intro--quiet" markdown="1">
<p class="doc-intro__kicker">Explicit market structure</p>
<p class="doc-intro__lead">The curves-first interface makes discounting and forwards explicit. All pricers in this style accept a <code>PricingContext</code> plus <code>tau</code> (time to expiry).</p>
<p class="doc-intro__support">Use this layer when the market container should stay explicit instead of being flattened into convenience inputs.</p>
</div>

## Context and curves

<p class="doc-section-lead">These are the core market-context objects behind the curves-first pricing path.</p>

::: option_pricing.market.curves
    options:
      members:
        - PricingContext
        - DiscountCurve
        - ForwardCurve
        - FlatDiscountCurve
        - FlatCarryForwardCurve

## Notes

- `PricingContext.df(tau)` returns the discount factor to maturity `tau`.
- `PricingContext.fwd(tau)` returns the forward price for maturity `tau`.
- `MarketData.to_context()` is the bridge from the flat convenience API into this one.
