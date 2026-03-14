# Volatility

## Implied volatility inversion

::: option_pricing.vol.implied_vol_scalar
    options:
      members:
        - implied_vol_bs
        - implied_vol_bs_result

## Volatility surfaces and smiles

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

The main eSSVI entrypoints are re-exported from `option_pricing.vol` and grouped under `option_pricing.vol.ssvi`.

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
