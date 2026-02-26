# Volatility

## Implied volatility inversion

::: option_pricing.vol.implied_vol
    options:
      members:
        - implied_vol_bs
        - implied_vol_bs_result

## Volatility surfaces and smiles

::: option_pricing.vol.surface
    options:
      members:
        - VolSurface
        - Smile
        - LocalVolSurface

## Notes

- Implied-vol inversion configuration lives in `ImpliedVolConfig` on the [Public API](public.md) page.
- `LocalVolSurface` is currently a demo-grade bridge from an implied surface to a local-vol surface.
