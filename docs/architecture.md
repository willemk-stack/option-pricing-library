# Architecture & Design

This page summarizes the architecture and design choices for option_pricing.

## Design goals

- Provide a typed, test-backed quant library rather than a notebook-only collection
- Support multiple pricing engines (Black-Scholes, CRR binomial, Monte Carlo, PDE) behind a coherent API
- Treat volatility tooling as first-class (implied vol inversion, smiles/surfaces, no-arb checks, SVI fit/repair, eSSVI calibration/projection)
- Make local-vol + PDE workflows diagnostics-heavy (invalid masks, denominator failures, stability cues)
- Preserve layered APIs: flat inputs for quick use, instrument-based for clarity, curves-first for term structures
- Emphasize validation and convergence (analytic baselines, remedies for discontinuities, repricing consistency, CI-executed notebooks)

## Public API map (canonical vs convenience)

Recommended public entry point: instrument-based API
VanillaOption + instrument pricers are the recommended entry point

Top-level exports (public API)
The package re-exports public symbols from __init__.py, including:

- Types: OptionType, OptionSpec, MarketData, PricingInputs
- Instruments: ExerciseStyle, VanillaPayoff, VanillaOption
- Curves-first: PricingContext, DiscountCurve, ForwardCurve, FlatDiscountCurve, FlatCarryForwardCurve
- Pricing entrypoints:  `bs_price*`, `bs_greeks*`, `mc_price*`, `binom_price*`
- Implied vol: `implied_vol_bs`, `implied_vol_bs_result`
- Vol objects: VolSurface, Smile

The volatility namespace `option_pricing.vol` additionally exposes the eSSVI stack:

- Term structures: `ThetaTermStructure`, `PsiTermStructure`, `EtaTermStructure`, `ESSVITermStructures`
- Surface objects: `ESSVIImpliedSurface`, `ESSVINodalSurface`, `ESSVISmoothedSurface`
- Calibration and validation: `calibrate_essvi`, `project_essvi_nodes`, `validate_essvi_continuous`, `validate_essvi_nodes`

Canonical vs convenience vs advanced paths

- Canonical: instrument-based pricing (VanillaOption + `bs_price_instrument`, `mc_price_instrument`, `binom_price_instrument`)
- Convenience / legacy: flat input path (`PricingInputs`, `bs_price`, `mc_price`, `binom_price`)
- Advanced: curves-first + surfaces + local-vol + PDE (`PricingContext`, `VolSurface`, `LocalVolSurface`, PDE pricers)

## Layered architecture (with dependency direction)

Layer responsibilities

- Types and instruments: contract specs and payoff interfaces (`OptionSpec`, `DigitalSpec`, `VanillaOption`, `DigitalOption`, `TerminalPayoff`)
- Market / curves: discount and forward curves, curves-first context (`PricingContext`, `FlatDiscountCurve`, `FlatCarryForwardCurve`)
- Models: coefficient builders for BS and local-vol PDEs (`bs_pde_coeffs`, `local_vol_pde_coeffs`)
- Numerics: grids, PDE operators, solvers, remedies (GridConfig, `solve_pde_1d`, LinearParabolicPDE1D, ICRemedy)
- Volatility: implied vol inversion, smiles/surfaces, no-arb checks, SVI/eSSVI calibration, local-vol extraction (VolSurface, Smile, `ESSVIImpliedSurface`, `ESSVINodalSurface`, `ESSVISmoothedSurface`, `check_surface_noarb`, `LocalVolSurface`, `local_vol_from_call_grid`)
- Pricers: public engines (BS / MC / CRR / PDE)
- Diagnostics: surface diagnostics and repricing audits

Dependency direction (audit)

- Core pricing paths depend on numerics/models/volatility modules but do not import diagnostics in the PDE pricers or numerics solver inspected
- Diagnostics explicitly depend on vol + pricers (e.g., `run_surface_diagnostics` calls `check_surface_noarb`; repricing diagnostics call `local_vol_price_pde_european`)
- No `option_pricing.diagnostics` imports appear in core pricing modules beyond the diagnostics package itself

<figure markdown class="diagram">
  ![Layered architecture](assets/diagrams/architecture_layers.light.svg){ .diagram-img .diagram-light }
  ![Layered architecture](assets/diagrams/architecture_layers.dark.svg){ .diagram-img .diagram-dark }
  <figcaption>Dependencies flow downward and right; diagnostics sit below the pricing stack.</figcaption>
</figure>

## Flagship end-to-end workflow (surface -> local vol -> PDE -> validation)

1) Quotes -> smiles -> surface
  - Build surface from quotes: `VolSurface.from_grid(rows, forward=...)`
  - Alternative: per-expiry SVI fit via `VolSurface.from_svi(..., calibrate_kwargs=...)` using `calibrate_svi`

2) No-arbitrage diagnostics + SVI fit/repair
  - Surface checks: `check_smile_price_monotonicity`, `check_smile_call_convexity`, `check_surface_noarb`
  - SVI calibration: `calibrate_svi(...)` -> `SVIFitResult`
  - SVI repair hooks: `repair_butterfly_raw`, `repair_butterfly_jw_optimal`, `repair_butterfly_with_fallback`
  - Diagnostics table: `svi_fit_table(surface)`

3) eSSVI calibration + projection
  - Direct parameter surface: `ESSVIImpliedSurface(params)`
  - Market calibration: `calibrate_essvi(...)` -> `ESSVIFitResult`
  - Exact nodal interpolation: `ESSVINodalSurface(fit.nodes)`
  - Smooth Dupire-oriented projection: `project_essvi_nodes(fit.nodes)` -> `ESSVIProjectionResult`
  - Validation: `evaluate_essvi_constraints`, `validate_essvi_nodes`, `validate_essvi_continuous`

4) Local vol extraction
  - Gatheral (from implied surface): `LocalVolSurface.from_implied(...)`, `LocalVolSurface.local_var(...)`, `LocalVolSurface.local_var_diagnostics(...)`
  - Preferred continuous-time surface for Dupire: `ESSVISmoothedSurface`
  - Dupire (from call price grids): `local_vol_from_call_grid(...)` and diagnostics `local_vol_from_call_grid_diagnostics(...)`
  - Diagnostics reason codes: `LVInvalidReason`, `GatheralLVReport`, `DupireLVReport`

5) PDE wiring + solve
  - Domain selection: BSDomainConfig, `bs_compute_bounds`
  - PDE wiring: `bs_pde_wiring` and `local_vol_pde_wiring` for vanilla options
  - Solver: `solve_pde_1d(problem, grid_cfg=..., method=..., advection=...)`
  - Convergence remedies for discontinuous payoffs: `ICRemedy`, `ic_cell_average`, `ic_l2_projection`
  - Rannacher time stepping: `RannacherCN1D`

6) Repricing and validation loop
  - Repricing grid: `localvol_pde_repricing_grid(...)` compares PDE prices to implied Black-76 targets
  - Convergence sweep: `localvol_pde_single_option_convergence_sweep(...)`
  - CI notebook execution: pytest -q demos --nbmake

<figure markdown class="diagram" style="--diagram-max-width: 1100px">
  ![Surface to local-vol to PDE workflow](assets/diagrams/workflow_surface_to_pde.light.svg){ .diagram-img .diagram-light }
  ![Surface to local-vol to PDE workflow](assets/diagrams/workflow_surface_to_pde.dark.svg){ .diagram-img .diagram-dark }
  <figcaption>Two surface construction paths converge into local-vol, then feed PDE pricing and repricing.</figcaption>
</figure>

## Key domain objects (definitions)

- `OptionSpec`, `DigitalSpec`, `PricingInputs` (core typed inputs)
- `MarketData`, `PricingContext` (flat and curves-first market containers)
- `VanillaOption`, `DigitalOption`, ExerciseStyle (instrument layer)
- `VolSurface`, `Smile` (implied vol objects)
- `SVIParams`, `SVISmile`, `SVIFitResult` (SVI model objects)
- `ESSVITermStructures`, `ESSVINodeSet`, `ESSVIImpliedSurface`, `ESSVINodalSurface`, `ESSVISmoothedSurface` (eSSVI model objects)
- `LocalVolSurface`, `LocalVolResult`, `GatheralLVReport`, `DupireLVReport` (local-vol objects + diagnostics)
- `GridConfig`, `PDESolution1D`, `LinearParabolicPDE1D` (PDE numerics core)

## Extension points

- Add a pricer: implement a new pricer module in pricers/ that follows the PricingInputs or instrument-based pattern (e.g., `bs_price_instrument`, `mc_price_instrument`)
- Add a vol model: create a new SmileSlice implementation and plug into VolSurface slices or extend `VolSurface.from_*` builders
- Add a PDE method: register a new method factory via `register_method` in numerics.pde.methods
- Add diagnostics: extend `run_surface_diagnostics` with new tables or arrays

## Invariants and guardrails (numerical engineering)

- Surface construction validates forward/strike positivity and strict log-moneyness ordering
- Smile interpolation enforces monotone-safe interpolation (Fritsch-Carlson, U-split fallback, linear fallback)
- No-arb proxies: monotonicity and convexity checks use reconstructed Black-76 call prices; calendar checks use total variance monotonicity in T
- Local-vol diagnostics: invalid masks with reason codes for denominators, curvature, non-finite inputs, negativity
- PDE solver checks: grid sizes must be valid (Nx>=4, Nt>=2), grids strictly increasing, boundary enforcement performed at each step
- Domain bounds: BS-specific LOG_NSIGMA bands include spot/strike and enforce log-flooring for S>0
- Discontinuous payoff remedies: ICRemedy supports cell-average and L2 projection, used in digital PDE workflows

## Validation and confidence (tests + CI evidence)

Strongest tests (8 to 12)

- `test_dupire_recovers_constant_vol` (Dupire local vol recovery from Black-76 grid)
- `test_localvol_pde_vanilla_matches_bs_on_constant_surface` (local-vol PDE vs BS baseline)
- `test_bs_price_pde_matches_black_scholes` (BS PDE vs analytic)
- `test_bs_digital_price_pde_matches_analytic_cell_avg_return_solution` (digital PDE with IC remedy)
- `test_convergence_remedies_digital` (Rannacher + IC remedies behavior for discontinuities)
- `test_localvol_digital_matches_quantlib_fd_on_flat_svi_surface` (QuantLib cross-check)
- `test_surface_calendar_variance_ok` and `test_surface_calendar_variance_fails_when_w_decreases` (no-arb calendar checks)
- `test_calibrate_svi_fits_synthetic_smile_linear_no_regularization` (SVI calibration)
- `test_repair_butterfly_raw_returns_feasible` (SVI repair)
- `test_noarb_worst_points_and_run_surface_diagnostics` (diagnostic tables + JSON report)
- `test_localvol_digital_convergence_sweep_over_strikes_and_maturities` (local-vol digital convergence sweep)

Showcase notebooks executed in CI

- CI runs pytest -q demos --nbmake
- Showcase demos: demos/05_pde_pricing_and_diagnostics.ipynb, demos/06_vol_surfaces_localvol_pde.ipynb

<figure markdown class="diagram">
  ![Validation stack](assets/diagrams/validation_stack.light.svg){ .diagram-img .diagram-light }
  ![Validation stack](assets/diagrams/validation_stack.dark.svg){ .diagram-img .diagram-dark }
  <figcaption>Validation layers from analytic baselines to local-vol PDE and cross-checks.</figcaption>
</figure>

## Optional refactors / maintenance risks (repo-grounded)

- The generic SVI-derived local-vol path still uses piecewise-linear time interpolation and can show banding/instability; the eSSVI smooth-projection path is now the time-differentiable alternative the docs should steer users toward
- Diagnostics depend on pandas/matplotlib in tests; matplotlib is optional and skipped via pytest.importorskip
- QuantLib cross-check is optional and skipped when QuantLib is missing
