# Pricers

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Execution layer</p>
<p class="doc-intro__lead">The library exposes parallel entry points for the three supported usage styles: compact <code>PricingInputs</code> wrappers, curves-first functions, and instrument-based functions.</p>
<p class="doc-intro__support">The point of this page is not to repeat the entire API surface in prose. It is to show how the entry points are grouped, and what kind of caller each group is meant to serve.</p>
</div>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Compact path</p>
<p class="doc-card__title"><code>PricingInputs</code></p>
- Best for short scripts, quick checks, and compact tutorials.
- Keeps all required inputs in one typed container.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Explicit market path</p>
<p class="doc-card__title">Curves-first</p>
- Use `*_from_ctx` functions when discount and forward curves should stay explicit.
- Best fit for term-structure-heavy workflows.
</div>
<div class="doc-card doc-card--accent" markdown="1">
<p class="doc-card__eyebrow">Recommended path</p>
<p class="doc-card__title">Instrument-based</p>
- Use reusable contracts such as `VanillaOption`.
- Best default public interface for pricing and Greeks.
</div>
</div>

## Black-Scholes / Black-76 with `PricingInputs`

<p class="doc-section-lead">These are the compact wrappers for one-container inputs.</p>

::: option_pricing.pricers.black_scholes
    options:
      members:
        - bs_price
        - bs_greeks

## Monte Carlo with `PricingInputs`

<p class="doc-section-lead">Use this path when you want the compact workflow but still need simulation-based pricing.</p>

::: option_pricing.pricers.mc
    options:
      members:
        - mc_price

## Binomial CRR with `PricingInputs`

<p class="doc-section-lead">This is the compact tree-based pricing entry point for the flat-input style.</p>

::: option_pricing.pricers.tree
    options:
      members:
        - binom_price

## Curves-first pricers

<p class="doc-section-lead">These functions keep the market structure explicit through <code>PricingContext</code> and curve objects.</p>

::: option_pricing.pricers.black_scholes
    options:
      members:
        - bs_price_from_ctx
        - bs_greeks_from_ctx

::: option_pricing.pricers.mc
    options:
      members:
        - mc_price_from_ctx

::: option_pricing.pricers.tree
    options:
      members:
        - binom_price_from_ctx

## Instrument-based pricers

<p class="doc-section-lead">This is the recommended public layer because the reusable contract object carries payoff and exercise semantics directly.</p>

::: option_pricing.pricers.black_scholes
    options:
      members:
        - bs_price_instrument
        - bs_price_instrument_from_ctx
        - bs_greeks_instrument
        - bs_greeks_instrument_from_ctx

::: option_pricing.pricers.mc
    options:
      members:
        - mc_price_instrument
        - mc_price_instrument_from_ctx

::: option_pricing.pricers.tree
    options:
      members:
        - binom_price_instrument
        - binom_price_instrument_from_ctx

## Notes

- `mc_price*` functions return `(price, std_error)`.
- Black-Scholes closed-form instrument pricers support European exercise only.
- Tree-based instrument pricers can use the instrument's exercise style, including American exercise where supported.
