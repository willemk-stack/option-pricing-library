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

## Notes

- Implied-vol inversion configuration lives in `ImpliedVolConfig` on the [Public API](public.md) page.
- `LocalVolSurface` is currently a demo-grade bridge from an implied surface to a local-vol surface.
