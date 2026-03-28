hide:
    - toc

<h1 class="homepage-hero__title homepage-page-title">Option Pricing Library</h1>

<div class="portfolio-hero homepage-hero" markdown="1">
<p class="hero-kicker">Typed pricing, surface repair, and validated numerics</p>
<p class="hero-copy homepage-hero__copy">This repo is strongest when the review follows one workflow: repair noisy implied-vol quotes, make the Dupire handoff smooth enough to differentiate, and validate the local-vol/PDE leg with repricing and convergence evidence.</p>
<p class="hero-copy homepage-hero__copy">The library surface is typed and reusable, but the main signal is judgment rather than abstraction. The docs keep assumptions, safeguards, and failure modes visible instead of hiding them behind one cleaned-up result.</p>
<div class="cta-row cta-row--trio homepage-cta-row" markdown="1">
[See the eSSVI handoff](user_guides/essvi_smooth_handoff.md){ .md-button .md-button--primary }
[Start with surface repair](user_guides/surface_workflow.md){ .md-button }
[Review performance evidence](performance.md){ .md-button }
</div>
</div>

<div class="doc-intro doc-intro--quiet" markdown="1">
<p class="doc-intro__kicker">Why this library matters</p>
<p class="doc-intro__lead">The project is not just a pricing API. It shows the full proof path from noisy market quotes to a smooth surface, then through local-vol extraction and PDE validation, with the delicate parts left visible for review.</p>
<p class="doc-intro__support">If you are evaluating quant depth or engineering maturity, the fastest route is the proof sequence rather than the full API catalog.</p>
</div>

## Signature proof moment

<p class="doc-section-lead">The proof object is the smoothed eSSVI surface used for the Dupire handoff, with smaller diagnostics showing the repair stage before it and the local-vol/PDE validation that follows.</p>

<div class="homepage-signature-layout" markdown="1">
<figure class="diagram diagram--hero homepage-signature-figure" style="--diagram-max-width: 980px" markdown="1">
![3D surface of the smoothed eSSVI implied-vol handoff across moneyness and maturity](assets/generated/showcase/homepage_essvi_surface_3d.light.png){ .diagram-img .diagram-light }
![3D surface of the smoothed eSSVI implied-vol handoff across moneyness and maturity](assets/generated/showcase/homepage_essvi_surface_3d.dark.png){ .diagram-img .diagram-dark }
<figcaption>The standout object is the <a href="user_guides/essvi_smooth_handoff.md">smooth eSSVI handoff</a>: the surface that makes <code>w_T</code> usable for Dupire-oriented local-vol extraction instead of leaving the workflow at a repaired slice stack.</figcaption>
</figure>

<div class="snapshot-grid homepage-support-grid" markdown="1">
<figure class="diagram diagram--quiet homepage-support-card" style="--diagram-max-width: 100%" markdown="1">
![Quote fit comparison across maturity for repaired SVI and eSSVI surfaces](assets/generated/static/quote_surface_compare.light.png){ .diagram-img .diagram-light }
![Quote fit comparison across maturity for repaired SVI and eSSVI surfaces](assets/generated/static/quote_surface_compare.dark.png){ .diagram-img .diagram-dark }
<figcaption><a href="user_guides/surface_workflow.md">Surface repair</a> stays visible first, so the hero surface does not hide the quoted-versus-repaired comparison that starts the proof path.</figcaption>
</figure>

<figure class="diagram diagram--quiet homepage-support-card" style="--diagram-max-width: 100%" markdown="1">
![Scatter plot comparing local-vol PDE repriced values with target prices](assets/generated/numerics/pde_roundtrip_scatter.light.png){ .diagram-img .diagram-light }
![Scatter plot comparing local-vol PDE repriced values with target prices](assets/generated/numerics/pde_roundtrip_scatter.dark.png){ .diagram-img .diagram-dark }
<figcaption><a href="user_guides/localvol_pde_validation.md">Local-vol / PDE validation</a> closes the loop with <code>154</code> repriced options and a published mean abs price error of <code>8.1e-4</code>.</figcaption>
</figure>
</div>
</div>

## Open the proof pages

<div class="doc-card-grid doc-card-grid--quiet homepage-route-grid" markdown="1">
[<span class="doc-card__eyebrow">Proof path step 1</span><span class="doc-link-card__title">Surface repair workflow</span><span class="doc-link-card__copy">See the quoted surface, the repaired SVI fit, and the slice-level stress before any smoothing tradeoff is accepted.</span>](user_guides/surface_workflow.md){ .doc-link-card .doc-link-card--quiet }

[<span class="doc-card__eyebrow">Proof path step 2</span><span class="doc-link-card__title">eSSVI smooth handoff</span><span class="doc-link-card__copy">See why time continuity matters, how seam stress drops, and why the smoother surface is the preferred Dupire input.</span>](user_guides/essvi_smooth_handoff.md){ .doc-link-card .doc-link-card--quiet }

[<span class="doc-card__eyebrow">Proof path step 3</span><span class="doc-link-card__title">Local-vol / PDE validation</span><span class="doc-link-card__copy">See repricing scatter, error localization, and convergence evidence for the final numerical leg of the workflow.</span>](user_guides/localvol_pde_validation.md){ .doc-link-card .doc-link-card--quiet }

[<span class="doc-card__eyebrow">Benchmark case study</span><span class="doc-link-card__title">Performance evidence</span><span class="doc-link-card__copy">Open the authored benchmark page for implied-vol scaling, PDE runtime/error tradeoffs, digital remedies, and stage budgets.</span>](performance.md){ .doc-link-card .doc-link-card--quiet }
</div>

## Secondary routes

- [Architecture](architecture.md) for the recruiter-facing systems view and safeguard story
- [Decision guide](user_guides/decision_guide.md) if you want the proof sequence in one routing page
- [API reference](api/index.md) for the typed public surface and generated symbol docs
- [Installation](installation.md) for local setup and editable development
