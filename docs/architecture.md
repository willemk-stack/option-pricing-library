# Architecture

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Recruiter-facing systems view</p>
<p class="doc-intro__lead"><code>option_pricing</code> is organized as a layered quant library: readable public entry points at the top, reusable volatility and numerics in the middle, and diagnostics downstream where they can inspect workflows without becoming hidden dependencies of the engines.</p>
<p class="doc-intro__support">Read this page as a systems story rather than a symbol inventory: what the package is optimizing for, how the proof workflow moves through the stack, and where safeguards make the numerics reviewable instead of fragile.</p>
</div>

## What The Architecture Is Optimizing For

<p class="doc-section-lead">The library is opinionated about three things: public routes should stay readable, numerical machinery should stay explicit, and reviewer-facing evidence should stay downstream of the pricing engines.</p>

<div class="doc-card-grid doc-card-grid--quiet" markdown="1">
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Public posture</p>
<p class="doc-card__title">Typed entry points</p>
- Support a real library workflow rather than a notebook-only workflow.
- Keep the recommended instrument-based path easy to discover.
- Preserve compact and curves-first routes for comparison and explicit market structure work.
</div>
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Quant posture</p>
<p class="doc-card__title">Explicit vol and numerics</p>
- Treat implied-vol inversion, surfaces, eSSVI, local vol, and PDE wiring as inspectable layers.
- Keep domain choice, remedies, solver wiring, and validation hooks visible.
- Avoid one black-box "advanced pricer" that hides the decision points.
</div>
<div class="doc-card doc-card--quiet" markdown="1">
<p class="doc-card__eyebrow">Review posture</p>
<p class="doc-card__title">Proof lives at the edge</p>
- Diagnostics consume the stack instead of being imported by it.
- Strong reviewer pages are built from the same workflow the code actually runs.
- Confidence comes from surfaced evidence, not from abstraction alone.
</div>
</div>

## System Picture

<p class="doc-section-lead">The dependency direction is the architecture contract: public objects feed the core stack, and diagnostics inspect the resulting workflow from outside the engines.</p>

<figure markdown class="diagram diagram--architecture-overview diagram--hero">
  ![Layered architecture](assets/diagrams/architecture_layers.light.svg){ .diagram-img .diagram-light }
  ![Layered architecture](assets/diagrams/architecture_layers.dark.svg){ .diagram-img .diagram-dark }
  <figcaption>Public APIs stay small and typed, the volatility and numerics core remains reusable, and diagnostics stay downstream where they can inspect the workflow without polluting the engines.</figcaption>
</figure>

| What to notice | Why it matters |
| --- | --- |
| Public interfaces sit above the pricing and volatility machinery | The library reads like a product surface first, not like a bag of internal helper functions |
| Diagnostics point inward at the workflow instead of being imported by the core pricers | Review tooling can stay rich without turning the engines into documentation-aware code |
| The proof path crosses several layers without collapsing them together | The repo shows systems thinking rather than isolated notebooks or isolated engines |

## Review Workflow And Safeguards

<p class="doc-section-lead">The strongest reviewer-facing path is intentionally explicit: surface repair, smooth eSSVI handoff, local-vol extraction, and PDE validation each expose a different engineering question.</p>

<div class="doc-panel doc-panel--strong" markdown="1">
<p class="doc-panel__label">Why this matters</p>
The architecture is not trying to hide complexity. It is trying to put complexity where it can be inspected. Each transition in the proof workflow answers a different reviewer question and keeps the failure mode visible before the next stage begins.
</div>

<ol class="doc-step-list" markdown="1">
<li markdown="1">
<p class="doc-step__title">Repair the raw surface first</p>
<p class="doc-meta">Start with quotes, smiles, and SVI repair rather than jumping straight into local vol.</p>
- `VolSurface`, `calibrate_svi(...)`, and no-arbitrage diagnostics keep slice quality visible.
- The reviewer can inspect quoted data, repaired slices, and fit stress before any time-smoothing story begins.
</li>
<li markdown="1">
<p class="doc-step__title">Make the time-direction choice explicit</p>
<p class="doc-meta">The handoff is a design decision, not a hidden interpolation detail.</p>
- `ESSVINodalSurface` preserves the exact calibrated nodes.
- `project_essvi_nodes(...)` and `ESSVISmoothedSurface` create the Dupire-oriented smooth surface only after explicit validation.
- `validate_essvi_nodes(...)` and `validate_essvi_continuous(...)` separate nodal validity from continuous-surface admissibility.
</li>
<li markdown="1">
<p class="doc-step__title">Keep local vol and PDE wiring inspectable</p>
<p class="doc-meta">The numerical stage exposes domain, method, and payoff-remedy choices instead of burying them in one opaque engine.</p>
- `LocalVolSurface.from_implied(...)` keeps the handoff object visible.
- `bs_pde_wiring`, `local_vol_pde_wiring`, `GridConfig`, `ICRemedy`, and `RannacherCN1D` make the solver choices reviewable.
</li>
<li markdown="1">
<p class="doc-step__title">Close the loop with published evidence</p>
<p class="doc-meta">The final question is not whether the stack runs, but whether it reprices, localizes error, and behaves sensibly as the grid changes.</p>
- Repricing grids, convergence sweeps, benchmark artifacts, and `pytest -q demos --nbmake` are all downstream of the same workflow.
- That is what lets the docs, tests, and performance page tell one coherent story.
</li>
</ol>

<figure markdown class="diagram" style="--diagram-max-width: 1100px">
  ![Surface to local-vol to PDE workflow](assets/diagrams/workflow_surface_to_pde.light.svg){ .diagram-img .diagram-light }
  ![Surface to local-vol to PDE workflow](assets/diagrams/workflow_surface_to_pde.dark.svg){ .diagram-img .diagram-dark }
  <figcaption>The strongest review path moves from static repair into a smooth eSSVI handoff, then into local-vol extraction and PDE validation with the diagnostic checkpoints kept visible.</figcaption>
</figure>

## Public Routes Into The Stack

<p class="doc-section-lead">The package exposes several entry styles, but they are not equal in purpose. The point is to keep the default public path readable without hiding the more explicit routes needed for quant workflows.</p>

| Route | Use it when | Representative objects and functions |
| --- | --- | --- |
| Instrument-based | You want the recommended typed public workflow | `VanillaOption`, `bs_price_instrument`, `mc_price_instrument`, `binom_price_instrument` |
| Flat-input | You want compact scripts or method comparisons | `PricingInputs`, `bs_price`, `mc_price`, `binom_price` |
| Curves-first | You want discounting and forwards to stay explicit | `PricingContext`, `DiscountCurve`, `ForwardCurve` |
| Volatility namespace | You are working on surfaces, eSSVI, local vol, or the proof path | `VolSurface`, `ESSVINodalSurface`, `ESSVISmoothedSurface`, `LocalVolSurface` |

## Confidence Model

<p class="doc-section-lead">Confidence comes from layered safeguards rather than one final integration test. Each stage has its own failure mode, its own guardrails, and its own published evidence.</p>

| Stage | Main guardrails | Evidence surfaced to the reviewer |
| --- | --- | --- |
| Surface construction and repair | Strike ordering, forward positivity, no-arbitrage proxies, fit diagnostics | Quoted-versus-repaired plots, slice stress, no-arb summaries |
| Smooth eSSVI handoff | Constraint checks, seam inspection, nodal vs continuous validation | Seam-jump table, projection diagnostics, invalid-count summary |
| Local-vol extraction | Invalid masks, reason codes, denominator and curvature checks | Gatheral and Dupire reports, heatmaps, difference views |
| PDE solve | Domain bounds, boundary handling, remedy choice, convergence sweeps | Repricing grids, convergence plots, digital-remedy benchmarks |
| Review and CI loop | Notebook execution plus committed benchmark artifacts | Proof pages, benchmark page, CI-executed demos |

## Extension And Maintenance

<div class="doc-panel doc-panel--quiet" markdown="1">
<p class="doc-panel__label">Extension points</p>
- Add new pricers under `pricers/` without rewriting the surface or diagnostics layers.
- Add new volatility models under `vol/` without changing the public entry patterns.
- Add new PDE methods by extending the numerical factory layer instead of hard-coding them into one workflow.
- Add new diagnostics downstream of the pricing stack so evidence stays rich without creating reverse dependencies.
</div>

<div class="doc-panel doc-panel--quiet" markdown="1">
<p class="doc-panel__label">Maintenance notes</p>
- The generic SVI-derived local-vol path remains demo-grade for Dupire because its time-direction behavior is only piecewise smooth.
- Diagnostics depend on optional plotting and comparison tooling in places, so some cross-checks remain conditional.
- The docs should continue steering Dupire-oriented readers toward the smooth eSSVI handoff rather than implying all surface paths are equally defensible.
</div>
