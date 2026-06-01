# Future work

<div class="portfolio-hero">
  <p class="hero-kicker">Stable today vs exploratory next</p>
  <p class="hero-copy">Stable today: typed vanilla pricing engines, implied-vol and surface tooling, SVI/eSSVI workflows, local-vol/PDE diagnostics, the namespaced Heston pricing and calibration stack, README freshness checks, docs builds, and CI-executed notebooks.</p>
  <p class="hero-copy">Exploratory next: hedging experiments, richer portfolio/reporting workflows, and additional evidence pages that tighten the story around the existing model layers.</p>
</div>

## Current capabilities

- **Public API and packaging**
  - typed package installable on Python 3.12+
  - recommended instrument workflow plus convenience and curves-first paths
  - stable, minimal top-level exports
- **Validation and CI**
  - README freshness enforced in CI
  - notebook execution with `pytest -q demos --nbmake`
  - headless notebook runs via `MPLBACKEND=Agg`
  - docs build and visual artifact generation in GitHub Actions
- **Portfolio-ready numerics**
  - Black-Scholes, Monte Carlo, and binomial CRR pricing routes
  - implied-vol solving, SVI fitting, and SVI repair
  - eSSVI projection and handoff diagnostics
  - local-vol extraction and PDE repricing checks
  - Heston Fourier vanilla pricing and Monte Carlo cross-checks
  - Heston calibration diagnostics and model-comparison workflows
- **Local market snapshot validation**
  - deterministic local fixture ingestion
  - Bronze, Silver, Gold, and model-validation bundle artifacts
  - credential-free reviewer workflow with explicit scope limits

## Near-term improvements

- keep install ergonomics smooth outside editable local workflows
- keep proof pages aligned with regenerated visual bundles and notebook outputs
- broaden local market snapshot examples while keeping live-provider and
  credential-backed workflows out of scope
- continue tightening Heston benchmark provenance and environment-scope
  disclosure around regenerated artifacts
- make documentation checks easier to reproduce locally across supported widths
  and themes

## Longer-term research directions

### Hedging realism

- delta/vega hedging simulator
- misspecification experiments that hedge under one model and simulate under
  another
- P&L attribution and reporting notebook

### Market-data validation

- separate validation track for provider-backed data quality
- explicit credential and refresh design before any live-provider workflow is
  presented as evidence
- calibration diagnostics that distinguish data quality issues from model and
  optimizer behavior

### Model-comparison depth

- broader held-out and stress-test protocols around Heston calibration stability
- direct IV-space objective only as a distinct, validated optimization path
- expanded comparison grids that keep local-vol PDE error and model error
  easier to separate
