# Architecture

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Recruiter-facing systems view</p>
<p class="doc-intro__lead"><code>option_pricing</code> is organized as a layered quant library: typed public routes enter a reviewable quant core, and diagnostics stay downstream where they can inspect the same workflow that produces the proof pages.</p>
<p class="doc-intro__support">The page walks the same stack as a workflow and safeguards story: repair, smooth handoff, local-vol extraction, PDE validation, and published proof outputs.</p>
</div>

## Recruiter-Facing System Picture

<p class="doc-section-lead">The architecture contract is simple: keep the public surface readable, keep the quant checkpoints explicit, and keep reviewer-facing evidence outside the engines so it can inspect the workflow rather than disappear inside it.</p>

<figure markdown class="diagram diagram--architecture-overview">
  <p>
    <picture>
      <source media="(max-width: 560px)" srcset="../assets/diagrams/architecture_layers.mobile.light.svg?v=20260328m">
      <img alt="Layered architecture" class="diagram-img diagram-light" src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" srcset="../assets/diagrams/architecture_layers.svg?v=20260328m 1x" />
    </picture>
    <picture>
      <source media="(max-width: 560px)" srcset="../assets/diagrams/architecture_layers.mobile.dark.svg?v=20260328m">
      <img alt="Layered architecture" class="diagram-img diagram-dark" src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" srcset="../assets/diagrams/architecture_layers.dark.svg?v=20260328m 1x" />
    </picture>
  </p>
  <figcaption>The main systems graphic centers the reviewable workflow itself: public routes feed into repair, smooth handoff, local-vol extraction, and PDE validation, while proof outputs remain downstream where they can expose the safeguards without becoming hidden engine dependencies.</figcaption>
</figure>

<div class="architecture-reading-line" markdown="1">
<p><strong>Public surface</strong> Typed routes stay readable and library-first.</p>
<p><strong>Workflow spine</strong> Repair, handoff, local-vol, and PDE remain explicit checkpoints.</p>
<p><strong>Proof edge</strong> Diagnostics stay downstream so the evidence can stay rich without polluting the engines.</p>
</div>

## Workflow And Safeguards Story

<p class="doc-section-lead">The technical story is the five checkpoints below: each one answers a different reviewer question before the workflow is allowed to move forward.</p>

<div class="doc-panel doc-panel--quiet" markdown="1">
<p class="doc-panel__label">Why this matters</p>
The architecture is not trying to hide complexity. It is trying to put complexity where it can be inspected. The proof workflow becomes credible because each transition exposes the next failure mode before the next layer starts relying on it.
</div>

<ol class="doc-step-list" markdown="1">
<li markdown="1">
<p class="doc-step__title">Repair the raw surface first</p>
<p class="doc-meta">Start with quotes, smiles, and SVI repair rather than jumping straight into local vol.</p>
<ul>
<li><code>VolSurface</code>, <code>calibrate_svi(...)</code>, and no-arbitrage diagnostics keep slice quality visible.</li>
<li>The reviewer can inspect quoted data, repaired slices, and fit stress before any time-smoothing story begins.</li>
</ul>
</li>
<li markdown="1">
<p class="doc-step__title">Make the smooth handoff explicit</p>
<p class="doc-meta">The handoff is a design decision, not a hidden interpolation detail.</p>
<ul>
<li><code>ESSVINodalSurface</code> preserves the exact calibrated nodes.</li>
<li><code>project_essvi_nodes(...)</code> and <code>ESSVISmoothedSurface</code> create the Dupire-oriented smooth surface only after explicit validation.</li>
<li><code>validate_essvi_nodes(...)</code> and <code>validate_essvi_continuous(...)</code> separate nodal validity from continuous-surface admissibility.</li>
</ul>
</li>
<li markdown="1">
<p class="doc-step__title">Extract local vol with failure visibility</p>
<p class="doc-meta">The Dupire stage stays inspectable instead of turning into a silent derivative black box.</p>
<ul>
<li><code>LocalVolSurface.from_implied(...)</code> keeps the handoff object visible.</li>
<li>Invalid masks, denominator checks, and curvature diagnostics preserve the reason why extraction succeeds or fails instead of collapsing it into one opaque output.</li>
</ul>
</li>
<li markdown="1">
<p class="doc-step__title">Validate the PDE wiring</p>
<p class="doc-meta">The numerical stage exposes domain, method, and payoff-remedy choices instead of burying them in one opaque engine.</p>
<ul>
<li><code>bs_pde_wiring</code>, <code>local_vol_pde_wiring</code>, <code>GridConfig</code>, <code>ICRemedy</code>, and <code>RannacherCN1D</code> make the solver choices reviewable.</li>
<li>The final question at this stage is not whether the stack runs, but whether it reprices, localizes error, and behaves sensibly as the grid changes.</li>
</ul>
</li>
<li markdown="1">
<p class="doc-step__title">Publish proof outputs and diagnostics</p>
<p class="doc-meta">The workflow only closes when the evidence is visible to a reviewer rather than trapped inside the implementation.</p>
<ul>
<li>Repricing grids, convergence sweeps, benchmark artifacts, and <code>pytest -q demos --nbmake</code> are all downstream of the same workflow.</li>
<li>That is what lets the docs, tests, and performance page tell one coherent story.</li>
</ul>
</li>
</ol>

<figure markdown class="diagram diagram--architecture-support" style="--diagram-max-width: 980px">
  ![Surface to local-vol to PDE workflow](assets/diagrams/workflow_surface_to_pde.svg?v=20260328b){ .diagram-img .diagram-light }
  ![Surface to local-vol to PDE workflow](assets/diagrams/workflow_surface_to_pde.dark.svg?v=20260328b){ .diagram-img .diagram-dark }
  <figcaption>The supporting workflow view stays quieter than the main systems graphic: it traces the same path from repaired surface to local-vol extraction, PDE solve, and downstream diagnostics without becoming a second competing hero.</figcaption>
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
| Proof outputs and CI loop | Notebook execution plus committed benchmark artifacts | Proof pages, benchmark page, CI-executed demos |

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
