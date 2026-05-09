# Notes

These notes are the library's maintained proof and implementation map. The
intended reading path is still public and linear:

surface repair -> eSSVI handoff -> local-vol/PDE -> Heston/model comparison.

Each topic folder now separates mathematical background, implementation policy,
validation evidence, and provisional heuristics so the section can grow without
turning into one long flat list.

## Start here

1. [Risk-neutral pricing](foundations/risk_neutral_pricing.md) - pricing
   measure, discounting, and martingale foundations.
2. [Brownian motion and Ito](foundations/brownian_motion_and_ito.md) - the
   stochastic calculus background used by diffusion models.
3. [Geometric Brownian motion](foundations/gbm.md) - the baseline lognormal
   stock model.
4. [Black-Scholes pricing](pricing/bs_pricing.md) - the closed-form and PDE
   reference model for European vanilla options.
5. [Implied volatility](volatility/implied_volatility.md) - price inversion,
   bounds, and practical solver notes.
6. [SVI calibration design](volatility/svi_calibration_design.md) - the
   slice-level surface-repair policy.
7. [eSSVI calibration design](volatility/essvi_calibration_design.md) - the
   cross-maturity handoff policy for Dupire-oriented workflows.
8. [Dupire local vol](local-vol-pde/dupire_local_vol.md) and
   [finite differences](local-vol-pde/finite_difference_pde.md) - the
   local-vol/PDE bridge and its numerical risks.
9. [Heston stochastic volatility](heston/heston_stochastic_vol.md) and
   [Heston pricing conventions](heston/heston_pricing_conventions.md) - the
   stochastic-volatility branch.
10. [Heston versus local volatility](heston/heston_vs_local_vol.md) - the
    model-comparison evidence surface.

## Topic groups

### Foundations

- [Risk-neutral pricing](foundations/risk_neutral_pricing.md) - pricing
  measure, discounting, and martingale foundations.
- [Brownian motion and Ito](foundations/brownian_motion_and_ito.md) -
  stochastic-calculus background for continuous-time models.
- [Geometric Brownian motion](foundations/gbm.md) - the lognormal stock model
  used as the baseline diffusion.

### Pricing

- [Black-Scholes pricing](pricing/bs_pricing.md) - PDE and closed-form vanilla
  pricing foundations.
- [Binomial CRR](pricing/binomial_crr.md) - discrete-time replication and
  convergence intuition.
- [Monte Carlo](pricing/mc.md) - simulation pricing, error bars, and variance
  reduction.

### Volatility

- [Implied volatility](volatility/implied_volatility.md) - inversion, bounds,
  and the practical meaning of quoted vol.
- [SVI calibration design](volatility/svi_calibration_design.md) - constrained
  raw-SVI slice calibration, repair hooks, and diagnostics.
- [eSSVI calibration design](volatility/essvi_calibration_design.md) -
  nodal eSSVI calibration and smooth projection policy.
- [Interpolation](volatility/interpolation.md) - review notes on interpolation
  choices and arbitrage-aware alternatives.

### Local-vol and PDE

- [Dupire local vol](local-vol-pde/dupire_local_vol.md) - local-vol
  construction and differentiation risks.
- [Finite differences](local-vol-pde/finite_difference_pde.md) - the
  discretization structure behind PDE pricing.
- [PDE convergence](local-vol-pde/pde_convergence.md) - convergence remedies
  and open implementation notes.

### Heston

- [Heston stochastic volatility](heston/heston_stochastic_vol.md) - model
  setup, notation, and characteristic-function context.
- [Heston pricing conventions](heston/heston_pricing_conventions.md) - pricing
  state, probability-index semantics, discounting, and Jacobian conventions.
- [Heston Fourier diagnostics](heston/heston_fourier_diagnostics.md) - warning
  interpretation, rerun guidance, and backend checks.
- [Heston quadrature policy](heston/heston_quadrature_policy.md) - deterministic
  integration policy for pricing, calibration, and published evidence.
- [Heston calibration evidence](heston/heston_calibration.md) - fit diagnostics,
  residual interpretation, and multistart review.
- [Heston calibration seeds](heston/heston_calibration_seeds.md) - deterministic
  seed choices and multistart coverage.
- [Heston calibration benchmark diagnostics](heston/heston_calibration_benchmark_diagnostics.md)
  - synthetic benchmark artifacts for calibration diagnostics.
- [Heston Monte Carlo diagnostics](heston/heston_monte_carlo.md) - bias/runtime
  sweeps, confidence intervals, and scheme comparison.
- [Andersen QE scheme](heston/heston_qe_notes.md) - QE implementation
  interpretation and limitations.
- [Heston versus local volatility](heston/heston_vs_local_vol.md) - model-choice
  outputs on a shared vanilla target.

### Diagnostics and numerical methods

- [Diagnostics notes](diagnostics/diagnostics_notes.md) - review-oriented notes
  for reading diagnostics outputs.
- [Fixed-rule quadrature](diagnostics/integration_quadrature.md) - Gaussian
  quadrature and Heston integration rationale.
- [Docs workflow architecture](diagnostics/docs_workflow_architecture.md) -
  documentation and publishing policy for generated artifacts.

## Taxonomy for future notes

Use **Foundations** for mathematical prerequisites that do not depend on this
repository's implementation. Use **Pricing** for model pricing recipes and
baseline numerical estimators. Use **Volatility** for implied-volatility,
surface-fitting, repair, and cross-maturity handoff design. Use
**Local-vol and PDE** for Dupire extraction, local-vol pricing, grids, and
convergence. Use **Heston** for stochastic-volatility pricing, calibration,
simulation, and model-comparison evidence. Use **Diagnostics and numerical
methods** for report interpretation, integration rules, generated-artifact
policy, and provisional review notes.

## Status language

Status blocks should make the source of each claim visible:

- **Mathematical fact** - a model identity or theorem-like statement, with a
  reference when it is not elementary.
- **Literature-backed convention** - a published method or parameterization
  adopted by the repo.
- **Repository policy** - an implementation default, validation rule, or
  review workflow chosen by this codebase.
- **Validation evidence** - a claim backed by tests, generated artifacts,
  notebooks, or the validation matrix.
- **Provisional heuristic** - a practical rule that is intentionally scoped,
  tunable, or awaiting stronger evidence.

Avoid using "robust", "production", "gold standard", or "must" unless the
sentence names whether it is a mathematical requirement, a repository policy,
or bounded validation evidence.

## Conventions and notation

- Time \(t\) is measured in years.
- Rates are continuously compounded unless stated otherwise.
- \(S_t\): underlying price at time \(t\).
- \(r\): continuously-compounded risk-free rate in basic models.
- \(B_t = e^{rt}\): money-market account in the basic setting.
- \(W_t\): Brownian motion.
- \(\mathbb{P}\): real-world probability measure.
- \(\mathbb{Q}\): risk-neutral measure under which discounted traded prices are
  martingales.

When a continuous dividend yield \(q\) is relevant, the risk-neutral drift is
\(r-q\).
