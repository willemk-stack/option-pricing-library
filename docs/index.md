---
hide:
  - navigation
  - toc
---

# Option Pricing Library

<div class="portfolio-hero">
  <p class="hero-kicker">Start with the strongest proof</p>
  <p class="hero-copy">This repo already has the depth. The goal of this docs front door is to get you from the library overview to the best engineering evidence quickly: surface repair, eSSVI smoothing, and local-vol/PDE validation.</p>
  <div class="cta-row cta-row--trio">
    <a class="md-button md-button--primary" href="user_guides/flagship_capstone2_page/">Read the decision guide</a>
    <a class="md-button" href="user_guides/instruments/">Start with the instrument API</a>
    <a class="md-button" href="api/">Browse the API reference</a>
  </div>
</div>

<figure class="figure-frame">
  <img src="assets/generated/showcase/reviewer_proof_panel.svg" alt="Reviewer proof panel showing the static surface to eSSVI to local-vol to PDE workflow, four published proof metrics, and supporting tracked visuals" />
  <figcaption>One panel to scan before opening the split proof pages: surface repair, eSSVI smoothing, local-vol diagnostics, and PDE repricing evidence.</figcaption>
</figure>

## Recommended path

1. [Decision guide](user_guides/flagship_capstone2_page.md)
2. [Surface flagship](user_guides/flagship_surface.md)
3. [eSSVI bridge](user_guides/flagship_essvi_bridge.md)
4. [Local vol + PDE flagship](user_guides/flagship_localvol_pde.md)

If you want the cleanest public API first, start with [Instruments](user_guides/instruments.md). The flat `PricingInputs` workflow remains available as the [convenience quickstart](user_guides/quickstart.md).

## Validation snapshot

<div class="snapshot-grid">
  <div class="snapshot-card">
    <p class="snapshot-label">Seam control</p>
    <p class="snapshot-copy">Worst observed <code>w_T</code> seam jump</p>
    <p class="metric-pair">
      <span class="metric-number">8.07e-2</span>
      <span class="metric-arrow">&rarr;</span>
      <span class="metric-number metric-number--good">8.17e-5</span>
    </p>
  </div>
  <div class="snapshot-card">
    <p class="snapshot-label">Dupire handoff</p>
    <p class="snapshot-copy"><code>projection_dupire_invalid_count</code></p>
    <p class="metric-line">
      <span class="metric-number metric-number--good">0</span>
    </p>
  </div>
  <div class="snapshot-card">
    <p class="snapshot-label">PDE repricing</p>
    <p class="snapshot-copy">Published repricing sweep</p>
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

## What the library is strongest at today

- **Vanilla pricing engines**: Black-Scholes(-Merton), CRR binomial trees, Monte Carlo under GBM
- **Volatility engineering**: implied-vol inversion, smiles, surfaces, no-arbitrage checks, SVI fitting and repair
- **Smooth Dupire handoff**: eSSVI calibration, validation, and explicit projection
- **Numerical validation**: local-vol diagnostics, PDE repricing, convergence checks, and CI-run notebooks

## First clicks

- [Installation](installation.md)
- [Instruments](user_guides/instruments.md)
- [Convenience quickstart](user_guides/quickstart.md)
- [Flagship demos](user_guides/flagship_capstone2_page.md)
- [Architecture](architecture.md)
- [Performance](performance.md)
- [Future work](roadmap.md)

## Documentation map

- [Quickstart](user_guides/instruments.md) for the recommended instrument workflow
- [Flagship demos](user_guides/flagship_capstone2_page.md) for the portfolio-grade proof path
- [API reference](api/index.md) for the public surface
- [Notes index](notes/index.md) for background and theory
- [Future work](roadmap.md) for what is exploratory next
