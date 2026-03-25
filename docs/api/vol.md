# Volatility

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Surface and handoff layer</p>
<p class="doc-intro__lead">The volatility namespace covers implied-vol inversion, smile/surface objects, local-vol extraction, and the eSSVI toolbox used by the proof-path pages.</p>
<p class="doc-intro__support">This page groups those objects by role so the reference is easier to scan than one continuous symbol list.</p>
</div>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Point tools</p>
<p class="doc-card__title">Implied-vol inversion</p>
- `implied_vol_bs`
- `implied_vol_bs_result`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Surface objects</p>
<p class="doc-card__title">Smiles and surfaces</p>
- `VolSurface`
- `Smile`
- `LocalVolSurface`
</div>
<div class="doc-card doc-card--accent" markdown="1">
<p class="doc-card__eyebrow">Proof-path stack</p>
<p class="doc-card__title">eSSVI toolbox</p>
- Term structures, nodal and smoothed surfaces
- Calibration, projection, and validation entry points
</div>
</div>

## Implied volatility inversion

<p class="doc-section-lead">These functions cover scalar Black-Scholes/Black-76 implied-vol inversion.</p>

::: option_pricing.vol.implied_vol_scalar
    options:
      members:
        - implied_vol_bs
        - implied_vol_bs_result

## Volatility surfaces and smiles

<p class="doc-section-lead">These are the common surface objects exposed near the pricing API and used throughout the guides.</p>

::: option_pricing.vol.surface_core
    options:
      members:
        - VolSurface
::: option_pricing.vol.smile_grid
    options:
      members:
        - Smile
::: option_pricing.vol.local_vol_surface
    options:
      members:
        - LocalVolSurface

## eSSVI toolbox

<p class="doc-section-lead">The main eSSVI entrypoints are re-exported from <code>option_pricing.vol</code> and grouped under <code>option_pricing.vol.ssvi</code>.</p>

::: option_pricing.vol.ssvi
    options:
      members:
        - ThetaTermStructure
        - PsiTermStructure
        - EtaTermStructure
        - ESSVITermStructures
        - ESSVINodeSet
        - ESSVIImpliedSurface
        - ESSVINodalSurface
        - ESSVISmoothedSurface
        - ESSVICalibrationConfig
        - ESSVIFitResult
        - ESSVIProjectionConfig
        - ESSVIProjectionResult
        - build_theta_term_from_quotes
        - calibrate_essvi
        - project_essvi_nodes
        - evaluate_essvi_constraints
        - validate_essvi_nodes
        - validate_essvi_continuous

## Notes

- Implied-vol inversion configuration lives in `ImpliedVolConfig` on the [Public API](public.md) page.
- The eSSVI objects are available from `option_pricing.vol`, not from the package root `option_pricing`.
- `LocalVolSurface` remains demo-grade for generic slice-stack interpolation, but it can also consume a time-differentiable surface such as `ESSVISmoothedSurface`.
