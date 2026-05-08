# Future work

<div class="portfolio-hero">
  <p class="hero-kicker">Stable today vs exploratory next</p>
  <p class="hero-copy">Stable today: typed vanilla pricing engines, implied-vol and surface tooling, SVI/eSSVI workflows, local-vol/PDE diagnostics, the namespaced Heston pricing and calibration stack, README freshness checks, docs builds, and CI-executed notebooks.</p>
  <p class="hero-copy">Exploratory next: hedging experiments, richer portfolio/reporting workflows, and capstone pages that tighten the evidence story around the existing model layers.</p>
</div>

## What is already solid

- **Public API and packaging**
  - typed package installable on Python 3.12+
  - recommended instrument workflow plus convenience and curves-first paths
- **Validation and CI**
  - README freshness enforced in CI
  - notebook execution with `pytest -q demos --nbmake`
  - headless notebook runs via `MPLBACKEND=Agg`
  - docs build and visual artifact generation in GitHub Actions
- **Portfolio-ready numerics**
  - SVI fitting and repair
  - eSSVI projection and seam diagnostics
  - local-vol extraction and PDE repricing checks
  - Heston Fourier vanilla pricing and Monte Carlo cross-checks
  - Heston calibration diagnostics and model-comparison workflows

## Active polish

- keep `option_pricing/__init__.py` exports stable and minimal
- continue improving install ergonomics outside editable local workflows
- keep the published proof pages aligned with the latest visual bundles and notebook outputs

## Research roadmap

### Capstone 3 - Heston stochastic volatility and model comparison

Status: active polish / release candidate.

Implemented:
- Heston Fourier vanilla pricing
- Heston Monte Carlo with full-truncation Euler and Andersen QE schemes
- calibration with bounded transforms and multistart
- calibration fit diagnostics and synthetic benchmark diagnostics
- Heston vs eSSVI/local-vol comparison layer

Remaining polish:
- final reviewer-facing model-comparison page
- Heston benchmark/performance evidence integration
- citation and claim-support audit for Heston notes
- README/homepage proof-card refresh once generated Heston visuals are ready

### Capstone 4 - Hedging realism

- delta/vega hedging simulator
- misspecification experiments (hedge under one model, simulate under another)
- P&L attribution and reporting notebook

## Suggested GitHub labels

- Capstones: `capstone:1`, `capstone:2`, `capstone:3`, `capstone:4`
- Type: `type:bug`, `type:enhancement`, `type:docs`, `type:chore`
- Area: `area:pricing`, `area:vol`, `area:diagnostics`, `area:ci`, `area:docs`
- Priority: `priority:p0`, `priority:p1`, `priority:p2`

## Release tags

- `v0.1.x`: BS/MC/tree baseline
- `v0.2.x`: implied vol solver hardened
- `v0.3.x`: vol surface + diagnostics
- `v0.4.x`: local vol + PDE
- `v0.5.x`: Heston + calibration
- `v1.0.0`: stable API + polished docs + end-to-end demos
