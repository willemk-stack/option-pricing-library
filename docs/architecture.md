# Architecture

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Typed library map</p>
<p class="doc-intro__lead"><code>option_pricing</code> is organized as a layered quant library: public pricing interfaces at the top, volatility/models/numerics in the core, and diagnostics at the edge where they can inspect workflows without polluting the engines.</p>
<p class="doc-intro__support">If you are reviewing the repo rather than just installing it, pair this page with the <a href="user_guides/decision_guide.md">Decision guide</a>, <a href="user_guides/localvol_pde_validation.md">Local-vol and PDE validation</a>, and <a href="performance.md">Performance evidence</a>. Together they show what the library exposes, how the pieces depend on one another, and where the reviewer-facing proof lives.</p>
<div class="doc-pill-row">
  <span class="doc-pill">Layered APIs</span>
  <span class="doc-pill">Diagnostics at the edge</span>
  <span class="doc-pill">Proof-first workflows</span>
  <span class="doc-pill">Typed numerics core</span>
</div>
</div>

## Design goals

<p class="doc-section-lead">The architecture is opinionated about what should stay simple, what should stay explicit, and where engineering evidence belongs.</p>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Library posture</p>
<p class="doc-card__title">Typed, reusable entry points</p>
- Provide a typed, test-backed quant library rather than a notebook-only collection.
- Support Black-Scholes, CRR binomial, Monte Carlo, and PDE engines behind a coherent API surface.
- Preserve layered APIs: flat inputs for quick use, instrument-based for clarity, curves-first for term structures.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Volatility and numerics</p>
<p class="doc-card__title">Quant workflow support</p>
- Treat implied vol inversion, smiles/surfaces, no-arb checks, SVI repair, and eSSVI calibration as first-class tooling.
- Make local-vol plus PDE workflows diagnostics-heavy, with invalid masks, denominator failures, and stability cues visible.
- Keep coefficient builders, grids, solvers, and remedies explicit enough to inspect rather than hide behind one black-box engine.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Review posture</p>
<p class="doc-card__title">Proof over polish</p>
- Emphasize validation and convergence through analytic baselines, discontinuity remedies, repricing consistency, and CI-executed notebooks.
- Split the strongest reviewer-facing workflow across focused pages and demos rather than one overloaded notebook.
- Keep diagnostics dependent on the pricing stack, not embedded inside the core engines.
</div>
</div>

## Public API map

<p class="doc-section-lead">The package exposes one recommended path, two alternative usage styles, and a broader volatility namespace for the surface-heavy workflows.</p>

<div class="doc-card-grid" markdown="1">
<div class="doc-card doc-card--accent" markdown="1">
<p class="doc-card__eyebrow">Recommended path</p>
<p class="doc-card__title">Instrument-based entry point</p>
- `VanillaOption` plus instrument pricers are the intended public starting point.
- Canonical functions: `bs_price_instrument`, `mc_price_instrument`, `binom_price_instrument`.
- Use this path when the contract itself should carry expiry, strike, payoff, and exercise semantics.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Package root exports</p>
<p class="doc-card__title">Top-level symbols</p>
- Types: `OptionType`, `OptionSpec`, `MarketData`, `PricingInputs`
- Instruments: `ExerciseStyle`, `VanillaPayoff`, `VanillaOption`
- Curves-first: `PricingContext`, `DiscountCurve`, `ForwardCurve`, `FlatDiscountCurve`, `FlatCarryForwardCurve`
- Pricing entrypoints: `bs_price*`, `bs_greeks*`, `mc_price*`, `binom_price*`
- Implied vol: `implied_vol_bs`, `implied_vol_bs_result`
- Vol objects: `VolSurface`, `Smile`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Volatility namespace</p>
<p class="doc-card__title"><code>option_pricing.vol</code></p>
- Term structures: `ThetaTermStructure`, `PsiTermStructure`, `EtaTermStructure`, `ESSVITermStructures`
- Surface objects: `ESSVIImpliedSurface`, `ESSVINodalSurface`, `ESSVISmoothedSurface`
- Calibration and validation: `calibrate_essvi`, `project_essvi_nodes`, `validate_essvi_continuous`, `validate_essvi_nodes`
- This is the namespace that carries the proof-path surface and Dupire handoff story.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Usage tiers</p>
<p class="doc-card__title">When to use which API</p>
- Use the instrument-based path for the default public workflow and most tutorials.
- Use the flat-input path (`PricingInputs`, `bs_price`, `mc_price`, `binom_price`) for compact checks and legacy-style scripts.
- Use curves-first plus surfaces plus local-vol plus PDE (`PricingContext`, `VolSurface`, `LocalVolSurface`, PDE pricers) when term structures or proof-path numerics matter.
</div>
</div>

## Layered architecture (with dependency direction)

<p class="doc-section-lead">The system is deliberately shaped so the public API stays readable, the pricing core stays reusable, and diagnostics remain inspectable without becoming an implicit dependency of the engines.</p>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Layer 1</p>
<p class="doc-card__title">Public interfaces</p>
- Contract specs and typed inputs: `OptionSpec`, `DigitalSpec`, `PricingInputs`
- Instrument layer: `VanillaOption`, `DigitalOption`, `TerminalPayoff`
- Market containers: `MarketData`, `PricingContext`
</div>
<div class="doc-card doc-card--accent" markdown="1">
<p class="doc-card__eyebrow">Layer 2</p>
<p class="doc-card__title">Core pricing stack</p>
- Volatility: implied-vol inversion, smiles, surfaces, SVI/eSSVI, local-vol extraction
- Models and wiring: `bs_pde_coeffs`, `local_vol_pde_coeffs`, `bs_pde_wiring`, `local_vol_pde_wiring`
- Numerics and pricers: `GridConfig`, `solve_pde_1d`, `LinearParabolicPDE1D`, `ICRemedy`, BS/MC/CRR/PDE engines
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Layer 3</p>
<p class="doc-card__title">Diagnostics and review</p>
- Surface diagnostics, repricing audits, and worst-point reports live here.
- These modules consume the pricing and volatility stack rather than being imported by it.
- That separation keeps proof tooling powerful without entangling the public engines.
</div>
</div>

<figure markdown class="diagram diagram--architecture-overview">
  ![Layered architecture](assets/diagrams/architecture_layers.light.svg){ .diagram-img .diagram-light }
  ![Layered architecture](assets/diagrams/architecture_layers.dark.svg){ .diagram-img .diagram-dark }
  <figcaption>High-level dependency picture: public interfaces feed the core stack, and diagnostics inspect the resulting workflow from outside the engines.</figcaption>
</figure>

### Dependency direction

- Core pricing paths depend on models, numerics, and volatility modules, but the inspected PDE pricers and numerics solver do not import diagnostics.
- Diagnostics explicitly depend on vol plus pricers. For example, `run_surface_diagnostics` calls `check_surface_noarb`, and repricing diagnostics call `local_vol_price_pde_european`.
- No `option_pricing.diagnostics` imports appear in the core pricing modules beyond the diagnostics package itself.

<div class="doc-panel" markdown="1">
<p class="doc-panel__label">Why this matters</p>
The dependency direction is part of the design contract, not just an implementation accident. It keeps the pricing core testable and composable, while still allowing the docs and reviewer workflows to surface rich diagnostics on top.
</div>

## Proof workflow (surface -> eSSVI -> local vol -> PDE)

<p class="doc-section-lead">The strongest reviewer-facing workflow is intentionally split across focused demos and pages so each transition stays explicit: static repair, smooth Dupire handoff, local-vol extraction, and PDE repricing validation.</p>

<div class="doc-panel" markdown="1">
<p class="doc-panel__label">Notebook chain</p>
The page sequence maps directly to `demos/06_surface_noarb_svi_repair.ipynb`, `demos/07_essvi_smooth_surface_for_dupire.ipynb`, `demos/08_localvol_pde_repricing.ipynb`, and `demos/09_surface_to_localvol_pde_integration.ipynb`, with `demos/05_pde_pricing_and_diagnostics.ipynb` as the PDE-only appendix.
</div>

<ol class="doc-step-list" markdown="1">
<li markdown="1">
<p class="doc-step__title">Quotes -> smiles -> surface</p>
<p class="doc-meta">Primary surfaces come either from quoted grids or from per-expiry SVI fits.</p>
- Build surfaces from quotes with `VolSurface.from_grid(rows, forward=...)`.
- Build per-expiry SVI surfaces with `VolSurface.from_svi(..., calibrate_kwargs=...)` via `calibrate_svi`.
</li>
<li markdown="1">
<p class="doc-step__title">No-arbitrage diagnostics + SVI repair</p>
<p class="doc-meta">Static-surface engineering stays visible before any smooth-hand-off step begins.</p>
- Run surface checks: `check_smile_price_monotonicity`, `check_smile_call_convexity`, `check_surface_noarb`
- Fit slices: `calibrate_svi(...)` -> `SVIFitResult`
- Repair butterflies: `repair_butterfly_raw`, `repair_butterfly_jw_optimal`, `repair_butterfly_with_fallback`
- Summarize fit stress with `svi_fit_table(surface)`
</li>
<li markdown="1">
<p class="doc-step__title">eSSVI calibration + projection</p>
<p class="doc-meta">This is the transition from exact nodal fit quality to a Dupire-oriented smooth surface.</p>
- Direct parameter surface: `ESSVIImpliedSurface(params)`
- Market calibration: `calibrate_essvi(...)` -> `ESSVIFitResult`
- Exact nodal interpolation: `ESSVINodalSurface(fit.nodes)`
- Smooth projection: `project_essvi_nodes(fit.nodes)` -> `ESSVIProjectionResult`
- Validation: `evaluate_essvi_constraints`, `validate_essvi_nodes`, `validate_essvi_continuous`
</li>
<li markdown="1">
<p class="doc-step__title">Local-vol extraction</p>
<p class="doc-meta">Both the preferred smooth-implied route and the call-grid Dupire route remain inspectable.</p>
- Gatheral route: `LocalVolSurface.from_implied(...)`, `LocalVolSurface.local_var(...)`, `LocalVolSurface.local_var_diagnostics(...)`
- Preferred continuous-time input surface: `ESSVISmoothedSurface`
- Call-grid Dupire route: `local_vol_from_call_grid(...)`, `local_vol_from_call_grid_diagnostics(...)`
- Diagnostics reason codes: `LVInvalidReason`, `GatheralLVReport`, `DupireLVReport`
</li>
<li markdown="1">
<p class="doc-step__title">PDE wiring + solve</p>
<p class="doc-meta">The PDE stage keeps the domain, method, and payoff-remedy choices explicit.</p>
- Domain selection: `BSDomainConfig`, `bs_compute_bounds`
- Vanilla wiring: `bs_pde_wiring`, `local_vol_pde_wiring`
- Solver: `solve_pde_1d(problem, grid_cfg=..., method=..., advection=...)`
- Discontinuity remedies: `ICRemedy`, `ic_cell_average`, `ic_l2_projection`
- Time stepping: `RannacherCN1D`
</li>
<li markdown="1">
<p class="doc-step__title">Repricing and validation loop</p>
<p class="doc-meta">The final question is not "does it run?" but "does it reprice, localize error, and refine sensibly?"</p>
- Repricing grid: `localvol_pde_repricing_grid(...)` compares PDE prices with implied Black-76 targets
- Convergence sweep: `localvol_pde_single_option_convergence_sweep(...)`
- CI notebook execution: `pytest -q demos --nbmake`
</li>
</ol>

<figure markdown class="diagram" style="--diagram-max-width: 1100px">
  ![Surface to local-vol to PDE workflow](assets/diagrams/workflow_surface_to_pde.light.svg){ .diagram-img .diagram-light }
  ![Surface to local-vol to PDE workflow](assets/diagrams/workflow_surface_to_pde.dark.svg){ .diagram-img .diagram-dark }
<figcaption>SVI owns the static-surface repair story; the smoothed eSSVI path is the preferred Dupire handoff into local-vol extraction and PDE repricing.</figcaption>
</figure>

## Key domain objects (definitions)

<p class="doc-section-lead">The same object families reappear across the guides, demos, and API reference. Grouping them by role is more useful than listing them in one flat block.</p>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Typed inputs</p>
<p class="doc-card__title">Contracts and market containers</p>
- `OptionSpec`, `DigitalSpec`, `PricingInputs`
- `MarketData`, `PricingContext`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Instrument layer</p>
<p class="doc-card__title">Contract objects</p>
- `VanillaOption`, `DigitalOption`
- `ExerciseStyle`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Implied-vol layer</p>
<p class="doc-card__title">Surface and SVI objects</p>
- `VolSurface`, `Smile`
- `SVIParams`, `SVISmile`, `SVIFitResult`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Smooth handoff layer</p>
<p class="doc-card__title">eSSVI and local vol</p>
- `ESSVITermStructures`, `ESSVINodeSet`
- `ESSVIImpliedSurface`, `ESSVINodalSurface`, `ESSVISmoothedSurface`
- `LocalVolSurface`, `LocalVolResult`, `GatheralLVReport`, `DupireLVReport`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Numerics core</p>
<p class="doc-card__title">PDE primitives</p>
- `GridConfig`
- `PDESolution1D`
- `LinearParabolicPDE1D`
</div>
</div>

## Extension points

<p class="doc-section-lead">The repo is layered so new engines, models, methods, or diagnostics can be added without rewriting the rest of the workflow.</p>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Pricers</p>
<p class="doc-card__title">Add a new engine</p>
- Implement a pricer module under `pricers/`.
- Follow either the `PricingInputs` pattern or the instrument-based pattern used by `bs_price_instrument` and `mc_price_instrument`.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Volatility</p>
<p class="doc-card__title">Add a new vol model</p>
- Create a new `SmileSlice` implementation.
- Plug it into `VolSurface` slices or extend the `VolSurface.from_*` builders.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">PDE numerics</p>
<p class="doc-card__title">Add a numerical method</p>
- Register a new factory via `register_method` in `numerics.pde.methods`.
- Keep the domain, grid, and time-stepping choices explicit enough to validate independently.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Diagnostics</p>
<p class="doc-card__title">Add new review outputs</p>
- Extend `run_surface_diagnostics` with new tables, arrays, or reports.
- Keep diagnostics downstream of the pricing and surface layers.
</div>
</div>

## Invariants and guardrails (numerical engineering)

<p class="doc-section-lead">The numerics are only defensible because the library enforces a set of preconditions and diagnostic reason codes before results are treated as trustworthy.</p>

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Surface construction</p>
<p class="doc-card__title">Input and interpolation checks</p>
- Surface construction validates forward/strike positivity and strict log-moneyness ordering.
- Smile interpolation enforces monotone-safe interpolation with Fritsch-Carlson, U-split fallback, and linear fallback.
- No-arb proxies use reconstructed Black-76 call prices for monotonicity and convexity, with total-variance monotonicity in `T` for calendar checks.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Local-vol extraction</p>
<p class="doc-card__title">Failure reporting</p>
- Local-vol diagnostics expose invalid masks with reason codes for denominators, curvature, non-finite inputs, and negativity.
- Gatheral and call-grid Dupire reports keep the failure mode visible instead of silently clipping bad regions.
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">PDE solve</p>
<p class="doc-card__title">Domain and payoff safeguards</p>
- PDE grids must be valid: `Nx >= 4`, `Nt >= 2`, and strictly increasing.
- Boundary conditions are enforced at each step.
- BS domain bounds use log-space bands that include spot/strike and enforce a positive-price floor.
- Discontinuous payoff remedies use `ICRemedy`, `ic_cell_average`, and `ic_l2_projection` in the digital workflows.
</div>
</div>

## Validation and confidence (tests + CI evidence)

<p class="doc-section-lead">Confidence comes from several independent layers: analytic baselines, no-arb and fit diagnostics, local-vol/PDE repricing checks, and notebook execution in CI.</p>

<div class="doc-panel" markdown="1">
<p class="doc-panel__label">Confidence model</p>
The architecture is not justified by abstraction alone. It is justified because the same layering supports analytic cross-checks, convergence evidence, surface diagnostics, and reviewer-facing docs that all agree on the same workflow.
</div>

### Representative tests

<div class="doc-card-grid" markdown="1">
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Analytic and PDE baselines</p>
<p class="doc-card__title">Pricing correctness</p>
- `test_bs_price_pde_matches_black_scholes`
- `test_localvol_pde_vanilla_matches_bs_on_constant_surface`
- `test_bs_digital_price_pde_matches_analytic_cell_avg_return_solution`
- `test_convergence_remedies_digital`
- `test_localvol_digital_matches_quantlib_fd_on_flat_svi_surface`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Surface and repair checks</p>
<p class="doc-card__title">Vol workflow validity</p>
- `test_dupire_recovers_constant_vol`
- `test_surface_calendar_variance_ok`
- `test_surface_calendar_variance_fails_when_w_decreases`
- `test_calibrate_svi_fits_synthetic_smile_linear_no_regularization`
- `test_repair_butterfly_raw_returns_feasible`
- `test_noarb_worst_points_and_run_surface_diagnostics`
</div>
<div class="doc-card" markdown="1">
<p class="doc-card__eyebrow">Workflow stress</p>
<p class="doc-card__title">Integrated numerical evidence</p>
- `test_localvol_digital_convergence_sweep_over_strikes_and_maturities`
- Repricing grids and convergence sweeps are published on the proof-path pages.
- The same artifacts feed the performance page and reviewer-facing docs.
</div>
</div>

### Showcase notebooks executed in CI

- CI runs `pytest -q demos --nbmake`.
- The main showcase demos are `demos/05_pde_pricing_and_diagnostics.ipynb`, `demos/06_surface_noarb_svi_repair.ipynb`, `demos/07_essvi_smooth_surface_for_dupire.ipynb`, `demos/08_localvol_pde_repricing.ipynb`, and `demos/09_surface_to_localvol_pde_integration.ipynb`.

<figure markdown class="diagram">
  ![Validation stack](assets/diagrams/validation_stack.light.svg){ .diagram-img .diagram-light }
  ![Validation stack](assets/diagrams/validation_stack.dark.svg){ .diagram-img .diagram-dark }
  <figcaption>Validation layers from analytic baselines to local-vol PDE repricing and cross-checks.</figcaption>
</figure>

## Optional refactors / maintenance risks (repo-grounded)

<div class="doc-panel" markdown="1">
<p class="doc-panel__label">Maintenance notes</p>

- The generic SVI-derived local-vol path still uses piecewise-linear time interpolation and can show banding or instability; the docs should continue steering users toward the smoothed eSSVI handoff when the Dupire path matters.
- Diagnostics depend on pandas and matplotlib in tests; matplotlib remains optional and is skipped via `pytest.importorskip`.
- The QuantLib cross-check is optional and skipped when QuantLib is not installed.
</div>
