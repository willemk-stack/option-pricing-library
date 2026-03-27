hide:
    - toc

<h1 class="homepage-hero__title homepage-page-title">Option Pricing Library</h1>

<div class="portfolio-hero homepage-hero" markdown="1">
<p class="hero-kicker">Typed pricing, surface repair, and validated numerics</p>
<p class="hero-copy homepage-hero__copy">This project is strongest when the review moves past vanilla pricing and into the workflow proof: repair a noisy implied-vol surface, make the Dupire handoff smooth enough to trust, and validate the local-vol/PDE path with repricing and convergence evidence.</p>
<p class="hero-copy homepage-hero__copy">It is built like a reusable library rather than a notebook collection, but the point is not abstraction for its own sake. The docs surface the engineering judgment: which assumptions matter, where the workflow can break, and what evidence supports the chosen defaults.</p>
<div class="cta-row cta-row--trio homepage-cta-row" markdown="1">
[See the eSSVI handoff](user_guides/essvi_smooth_handoff.md){ .md-button .md-button--primary }
[Review performance evidence](performance.md){ .md-button }
[Start with surface repair](user_guides/surface_workflow.md){ .md-button }
</div>
</div>

<div class="doc-intro doc-intro--quiet" markdown="1">
<p class="doc-intro__kicker">Why this library matters</p>
<p class="doc-intro__lead">The interesting part of the repo is not that it prices options. It is that pricing, surface repair, smooth eSSVI projection, and local-vol/PDE validation are connected into one reviewer-visible workflow instead of being hidden across disconnected demos.</p>
<p class="doc-intro__support">If you are evaluating engineering maturity, the fastest route is the proof path rather than the full API catalog.</p>
</div>

<div class="homepage-proof-section" markdown="1">
  <div class="homepage-proof-section__header">
    <p class="homepage-section-kicker">Featured proof panel</p>
    <p class="homepage-proof-section__copy">One scan across quoted-to-repaired surface evidence, the smooth Dupire handoff, and the local-vol/PDE validation loop.</p>
  </div>
  <figure class="diagram homepage-proof-panel" style="--diagram-max-width: 1180px">
    <img alt="Reviewer proof panel showing surface repair, eSSVI smoothing, local-vol extraction, and PDE repricing with tracked evidence callouts" src="assets/generated/showcase/reviewer_proof_panel.light.svg" class="diagram-img diagram-light" />
    <img alt="Reviewer proof panel showing surface repair, eSSVI smoothing, local-vol extraction, and PDE repricing with tracked evidence callouts" src="assets/generated/showcase/reviewer_proof_panel.dark.svg" class="diagram-img diagram-dark" />
    <figcaption>The proof path is designed to make judgment visible: first repair, then smooth the handoff, then show the numerical consequences.</figcaption>
  </figure>
</div>

## Published proof strip

<div class="snapshot-grid homepage-snapshot-grid">
  <div class="snapshot-card homepage-snapshot-card homepage-snapshot-card--transition">
    <p class="snapshot-label">Seam control</p>
    <p class="snapshot-copy">Worst observed <code>w_T</code> seam jump</p>
    <p class="metric-pair">
      <span class="metric-number">8.07e-2</span>
      <span class="metric-arrow">&rarr;</span>
      <span class="metric-number metric-number--good">8.17e-5</span>
    </p>
  </div>
  <div class="snapshot-card homepage-snapshot-card homepage-snapshot-card--zero">
    <p class="snapshot-label">Dupire handoff</p>
    <p class="snapshot-copy"><code>projection_dupire_invalid_count</code></p>
    <p class="metric-line">
      <span class="metric-number metric-number--good">0</span>
    </p>
  </div>
  <div class="snapshot-card homepage-snapshot-card homepage-snapshot-card--detail">
    <p class="snapshot-label">PDE repricing</p>
    <p class="snapshot-copy">Published local-vol repricing sweep</p>
    <p class="metric-line">
      <span class="metric-number">154</span>
      <span class="metric-caption">options repriced</span>
    </p>
    <p class="metric-inline">
      <span class="metric-key">Mean abs price error</span>
      <span class="metric-value">8.1e-4</span>
    </p>
    <p class="metric-inline">
      <span class="metric-key">Max abs IV error</span>
      <span class="metric-value">18.7 bp</span>
    </p>
  </div>
</div>

## Start With The Strongest Pages

<div class="doc-card-grid" markdown="1">
[<span class="doc-card__eyebrow">Priority route 1</span><span class="doc-link-card__title">eSSVI smooth handoff</span><span class="doc-link-card__copy">The clearest page for why time continuity matters before local-vol extraction, and how the chosen projection is validated.</span>](user_guides/essvi_smooth_handoff.md){ .doc-link-card }

[<span class="doc-card__eyebrow">Priority route 2</span><span class="doc-link-card__title">Performance evidence</span><span class="doc-link-card__copy">Committed benchmark artifacts for implied-vol scaling, PDE runtime/error tradeoffs, digital remedies, and end-to-end stage budgets.</span>](performance.md){ .doc-link-card }

[<span class="doc-card__eyebrow">Priority route 3</span><span class="doc-link-card__title">Surface repair workflow</span><span class="doc-link-card__copy">Quoted-versus-repaired evidence, static no-arbitrage diagnostics, and the point where the proof path starts.</span>](user_guides/surface_workflow.md){ .doc-link-card }

[<span class="doc-card__eyebrow">Priority route 4</span><span class="doc-link-card__title">Local-vol / PDE validation</span><span class="doc-link-card__copy">Repricing scatter, error localization, and convergence evidence for the final numerical leg of the workflow.</span>](user_guides/localvol_pde_validation.md){ .doc-link-card }
</div>

## Secondary routes

- [Architecture](architecture.md) for the recruiter-facing systems view and safeguard story
- [API reference](api/index.md) for the typed public surface and generated symbol docs
- [Installation](installation.md) for local setup and editable development
- [Decision guide](user_guides/decision_guide.md) if you want the proof sequence in one routing page
