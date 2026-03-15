# Flagship demos

The repo now presents its strongest volatility and numerics signals through a **split demo suite**, not a single catch-all capstone.

That split is intentional:

- **SVI** is the flagship for **static-surface engineering**: per-slice fitting, no-arbitrage diagnostics, butterfly repair, and interpolation judgment.
- **eSSVI** is the flagship bridge for a **Dupire-ready smooth term structure**: nodal calibration, explicit projection, and analytic `w_T`.
- **Local vol + PDE** is the flagship for **numerical engineering**: diagnostics-first local vol, PDE repricing, and convergence evidence.

## Decision guide

| If you need to show... | Start here | Why |
| --- | --- | --- |
| Static no-arb diagnostics, SVI fitting, and repair | `demos/06_surface_noarb_svi_repair.ipynb` | This is the cleanest surface-engineering story in the repo. |
| Why a smooth eSSVI projection is the preferred Dupire handoff | `demos/07_essvi_smooth_surface_for_dupire.ipynb` | It makes the `w_T` / term-structure argument explicit. |
| Local-vol diagnostics, PDE repricing, and convergence | `demos/08_localvol_pde_repricing.ipynb` | This is the numerics flagship. |
| PDE credibility before talking about surfaces | `demos/05_pde_pricing_and_diagnostics.ipynb` | It isolates the solver story. |
| The full workflow connected end to end | `demos/09_surface_to_localvol_pde_integration.ipynb` | This is the integration proof, not the main flagship. |

## Public positioning

- Use **SVI** when the claim is: "this library can engineer and repair a static implied surface."
- Use **eSSVI** when the claim is: "this library can produce a smoother Dupire-oriented implied surface with analytic time derivatives."
- Use **LocalVolSurface + PDE** when the claim is: "this library treats local vol and PDE pricing as a validation-heavy numerical workflow."

## Recommended path through the suite

1. Read the [Surface flagship](flagship_surface.md) page and run `demos/06_surface_noarb_svi_repair.ipynb`.
2. Move to the [eSSVI bridge](flagship_essvi_bridge.md) page and run `demos/07_essvi_smooth_surface_for_dupire.ipynb`.
3. Finish with the [Local vol + PDE flagship](flagship_localvol_pde.md) page and run `demos/08_localvol_pde_repricing.ipynb`.

Keep `demos/09_surface_to_localvol_pde_integration.ipynb` for interview walkthroughs, end-to-end sanity checks, or when you want to prove the pieces connect without forcing every reader through the whole stack first.

## Published visuals

The canonical publish path now lives in one script:

```bash
python scripts/build_visual_artifacts.py all --profile publish
```

That command writes the versioned data bundle to `out/visual_bundles/profile_publish_seed_7/`
and refreshes the committed figure presets under `docs/assets/generated/`.
