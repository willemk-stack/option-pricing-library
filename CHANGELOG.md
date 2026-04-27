# Changelog
All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog*, and this project aims to follow *Semantic Versioning*:
- MAJOR: breaking changes to the public API
- MINOR: new features, backwards-compatible
- PATCH: bug fixes, backwards-compatible

## [Unreleased]
### Added
- (Add bullets here as you work; move them into the next release at tag time.)
- Cached reusable Gauss-Legendre quadrature rules for the Heston Fourier path, including `option_pricing.numerics` exports for `QuadratureConfig`, `PanelSpacing`, `CompositeRule`, `CompositeIntegrationResult`, and `build_gauss_legendre_rule`.
- Heston simulation and Monte Carlo wrappers now support the public `quadratic_exponential` scheme, wiring Andersen QE through the path and terminal simulators.
- Heston Monte Carlo comparison diagnostics under `option_pricing.diagnostics.heston`, including bias-vs-timestep, runtime-vs-error, scheme-comparison summaries, and table-driven plotting helpers.

### Changed
- Docs workflow architecture notes now describe the single-workflow `docs-ci` validate/build/deploy contract and the split between fast local pre-push checks and heavier manual/CI browser validation.
- Public Heston simulation and Monte Carlo pricing entrypoints now default to `quadratic_exponential`; pass `scheme="euler_full_truncation"` explicitly when you want the baseline Euler scheme for comparisons or educational runs.
- Heston pricer entrypoints can now accept either a reusable quadrature config or a prebuilt composite rule and pass it through to the Fourier backend without changing the default numerical path.
- Monte Carlo pricing entrypoints now return `MonteCarloResult` instead of `(price, stderr)` tuples, and the canonical direct path-payoff GBM entrypoint is `mc_price_path_payoff_from_ctx(...)`.
- Monte Carlo engine orchestration now lives under `option_pricing.monte_carlo`, while the reusable GBM simulator adapters live under `option_pricing.models.gbm`, leaving `option_pricing.pricers.mc` as a thin public GBM wrapper layer.

### Fixed
- Locked the docs workflow contract in tests so blocking CI remains check-only for committed generated docs assets, deployment stays in `docs-ci`, and the docs-impact-selected authoritative visual suites remain the blocking source of truth.

### Deprecated

### Removed
- The stateful `McGBMModel` wrapper has been removed; use the stateless `mc_price*` entrypoints instead.
- The generic simulator-backed terminal/path engine entrypoints are no longer re-exported from `option_pricing.pricers.mc` or the package root; use `option_pricing.monte_carlo` for those advanced APIs.


## [0.4.0] - 2026-02-28
### Added
- End-to-end local volatility + PDE workflow:
  - Local vol computed from implied call grids (Dupire) with diagnostics (invalid masks + reason codes).
  - Local var computed from total variance (Gatheral) with denominator diagnostics.
- 1D finite-difference PDE solver stack:
  - Grid construction utilities (including clustered spacing).
  - Theta-family stepping (incl. Crank–Nicolson) with tridiagonal solve.
  - Discontinuous-payoff remedies (e.g., Rannacher-style warmup + IC projection/averaging).
- Local-vol PDE pricers for numerically sensitive payoffs (e.g., digitals), plus validation harnesses.
- Convergence and stability experiments:
  - Regression tests guarding numerical drift.
  - Script to reproduce/approximate literature-style convergence tables for discontinuous payoffs.

### Changed
- Public-facing docs and demos expanded to include the “Surface → Local vol → PDE price → convergence” narrative.

### Fixed
- Numerical guardrails around local-vol denominators/curvature thresholds and trimming regions.

### Deprecated
- Legacy/obsolete module paths retained only as stubs; use canonical local-vol modules instead.

## [0.3.0] - 2026-01-20
### Added
- Volatility surface tooling:
  - Smile/surface objects and interpolation layer.
  - No-arbitrage diagnostics (calendar/spread/convexity-style checks).
- SVI parameterization:
  - SVI fitting utilities and surface-level diagnostics.
  - Repair/cleanup utilities to improve surface sanity for downstream usage.

### Changed
- Documentation expanded with user guides and math notes for IV surfaces and no-arb diagnostics.

### Fixed
- Edge-case handling for smile construction and interpolation boundaries.

## [0.2.0] - 2025-12-10
### Added
- Robust implied volatility inversion:
  - Scalar implied vol result object with solver diagnostics.
  - Vectorized slice implied vol across strikes (Black76-style) with parity handling.
- Dedicated root-finding utilities and configuration objects for stable inversion.

### Fixed
- Price bound validation and tau==0 behavior for implied vol edge cases.

## [0.1.0] - 2025-11-01
### Added
- Core pricing engines for vanilla options:
  - Analytic Black–Scholes(-Merton) price + Greeks.
  - Binomial (CRR) pricing including American exercise where applicable.
  - Monte Carlo under GBM with standard error reporting.
- Foundational structure:
  - `src/` layout, typed dataclasses, and a top-level re-exported public API.
  - CI: lint/typecheck/test pipeline.
  - Docs scaffolding and example scripts.