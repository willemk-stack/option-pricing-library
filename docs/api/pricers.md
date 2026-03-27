# Pricers

<div class="doc-intro doc-intro--quiet" markdown="1">
<p class="doc-intro__kicker">Execution layer</p>
<p class="doc-intro__lead">The library exposes parallel entry points for the three supported usage styles: compact <code>PricingInputs</code> wrappers, curves-first functions, and instrument-based functions.</p>
<p class="doc-intro__support">This page stays reference-first. Use the section headings to jump to the interface style you are calling.</p>
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
