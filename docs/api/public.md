# Public API

<div class="doc-intro doc-intro--quiet" markdown="1">
<p class="doc-intro__kicker">Root-level exports</p>
<p class="doc-intro__lead">This page covers the main objects re-exported from <code>option_pricing</code>.</p>
<p class="doc-intro__support">Use it to find the shared public types before diving into the dedicated pricer, curves, or volatility reference pages.</p>
</div>

<p class="doc-meta">For pricing functions, use <a href="../pricers/">Pricers</a>. This page stays focused on shared types, contracts, configuration objects, and common volatility exports.</p>

## Core types

<p class="doc-section-lead">These are the root-level typed containers used throughout the convenience API and as bridges into the richer market-context workflows.</p>

::: option_pricing.types
    options:
      members:
        - OptionType
        - MarketData
        - OptionSpec
        - PricingInputs

## Instrument layer

<p class="doc-section-lead">The instrument workflow is the recommended public path because the contract carries payoff and exercise semantics explicitly.</p>

::: option_pricing.instruments.base
    options:
      members:
        - ExerciseStyle
        - TerminalInstrument
        - PathInstrument

::: option_pricing.instruments.vanilla
    options:
      members:
        - VanillaPayoff
        - VanillaOption

## Market context and curves

<p class="doc-section-lead">These names are re-exported from the package root so callers can move from flat inputs to explicit market curves without changing namespaces.</p>

- `PricingContext`
- `DiscountCurve`
- `ForwardCurve`
- `FlatDiscountCurve`
- `FlatCarryForwardCurve`

See the dedicated [Curves-first API](curves.md) page for the canonical class documentation.

## Configuration and solver selection

<p class="doc-section-lead">These configuration types control Monte Carlo sampling and implied-vol inversion behavior without being tied to a single pricing style.</p>

::: option_pricing.config
    options:
      members:
        - ImpliedVolConfig

::: option_pricing.monte_carlo
    options:
      members:
        - RandomConfig
        - MCConfig
        - MonteCarloResult
        - TerminalSimulator
        - PathSimulator

::: option_pricing.numerics.root_finding
    options:
      members:
        - RootMethod

## Volatility objects

<p class="doc-section-lead">The package root keeps the common smile and surface objects close to the pricing API, while the broader volatility namespace carries the heavier calibration stack.</p>

- `VolSurface`
- `Smile`

See the dedicated [Volatility](vol.md) page for the canonical class documentation.

The broader volatility namespace `option_pricing.vol` also exposes the eSSVI toolbox, including:

- `ESSVIImpliedSurface`
- `ESSVINodalSurface`
- `ESSVISmoothedSurface`
- `calibrate_essvi`
- `project_essvi_nodes`
