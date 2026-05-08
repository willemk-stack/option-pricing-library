# Notes

These notes are a compact “story” of the standard option-pricing toolkit used throughout the library:
**no-arbitrage → risk-neutral pricing → Brownian motion/Itô → GBM → Black–Scholes → implied vol → numerical methods → local vol → stochastic vol**.

They are written to be readable linearly, but each note also stands on its own.

## Suggested reading order

1. [Risk-neutral pricing](risk_neutral_pricing.md) — why we price under \(\mathbb{Q}\) and why the drift becomes \(r\).
2. [Brownian motion and Itô](brownian_motion_and_Ito.md) — the noise and calculus behind continuous-time models.
3. [Geometric Brownian motion](gbm.md) — the lognormal stock model and distribution of returns.
4. [Black–Scholes pricing](bs_pricing.md) — PDE and closed-form European option prices, plus intuition.
5. [Implied volatility](IV.md) — “vol as the price”; inversion, bounds, and practical solver notes.
6. [Monte Carlo](mc.md) — simulation pricing, error bars, and variance reduction.
7. [Binomial CRR](binomial_crr.md) — discrete-time replication and convergence to Black–Scholes.
8. [Finite Differences](finite_difference_pde.md) — how PDE pricing is discretized in practice.
9. [Dupire local vol](dupire_local_vol.md) — deterministic volatility surfaces that fit vanilla prices.
10. [SVI calibration design](svi_calibration_design.md) — why the raw smile fit is wrapped in transforms, regularization, and repair before downstream use.
11. [Heston stochastic volatility](heston_stochastic_vol.md) — affine stochastic vol, characteristic functions, and the bridge to capstone 3.

## Notes by purpose

### Theory and modeling foundations

- [Risk-neutral pricing](risk_neutral_pricing.md) - pricing measure, discounting, and martingale foundations.
- [Brownian motion and Itô](brownian_motion_and_Ito.md) - stochastic calculus background for continuous-time models.
- [Geometric Brownian motion](gbm.md) - the lognormal stock model used as the baseline diffusion.
- [Black-Scholes pricing](bs_pricing.md) - PDE and closed-form vanilla pricing foundations.
- [Implied volatility](IV.md) - inversion, bounds, and the practical meaning of quoted vol.
- [Heston stochastic volatility](heston_stochastic_vol.md) - stochastic-variance model setup, notation, and characteristic-function context.

### Surface and calibration implementation policy

- [SVI calibration design](svi_calibration_design.md) - constrained transforms, repair hooks, and calibration-time safeguards.

### Local-vol / PDE validation

- [Finite Differences](finite_difference_pde.md) - discretization structure behind PDE pricing.
- [Dupire local vol](dupire_local_vol.md) - local-vol construction and its modeling risks.

### Heston implementation policy

- [Heston pricing conventions](heston_pricing_conventions.md) - pricing state, probability indices, discounting, and Jacobian conventions.
- [Heston quadrature policy](heston_quadrature_policy.md) - documented integration defaults and rerun tiers.
- [Heston calibration seed design](heston_calibration_seeds.md) - deterministic seed choices and multi-start coverage.
- [Andersen's QE Heston simulation scheme](heston_qe_notes.md) - QE implementation interpretation and limitations.

### Heston validation and diagnostics

- [Heston calibration](heston_calibration.md) - fit diagnostics, residual interpretation, and multistart review.
- [Heston calibration benchmark diagnostics](heston_calibration_benchmark_diagnostics.md) - synthetic benchmark artifacts for calibration diagnostics.
- [Heston Fourier diagnostics](heston_fourier_diagnostics.md) - warning interpretation, rerun guidance, and backend checks.
- [Heston Monte Carlo diagnostics](heston_monte_carlo.md) - bias/runtime sweeps, confidence intervals, and scheme comparison.
- [Heston versus local volatility](heston_vs_local_vol.md) - capstone model-comparison outputs on a shared quote target.

### Provisional or review-policy notes

- [Diagnostics notes](diagnostics_notes.md) - review-oriented notes for reading diagnostics outputs.
- [Docs workflow architecture](docs_workflow_architecture.md) - documentation and publishing policy for generated artifacts.

## Surface-fitting implementation notes

- [SVI calibration design](svi_calibration_design.md) - why raw SVI calibration is wrapped in constrained transforms, regularization, robust loss, and post-fit repair.

## Heston implementation notes

- [Heston stochastic volatility](heston_stochastic_vol.md) - model setup, notation, characteristic functions, and calibration transform context.
- [Heston pricing conventions](heston_pricing_conventions.md) - pricing state, probability indices, discounting, call/put formulas, and Jacobian conventions.
- [Heston Fourier diagnostics](heston_fourier_diagnostics.md) - branch stability, finite-price warnings, and backend review checks.
- [Heston quadrature policy](heston_quadrature_policy.md) - deterministic integration policy for pricing, calibration, and published evidence.
- [Heston calibration seed design](heston_calibration_seeds.md) - default seeds, compact seed grids, and multi-start sensitivity rationale.
- [Heston calibration benchmark diagnostics](heston_calibration_benchmark_diagnostics.md) - benchmark outputs, analytic Jacobian checks, and recovery diagnostics.
- [Heston Monte Carlo diagnostics](heston_monte_carlo.md) - path simulation comparisons, convergence sweeps, and control-variate notes.
- [Andersen's QE Heston simulation scheme](heston_qe_notes.md) - QE variance transition logic and why Euler is only a baseline.

## Conventions and notation

- Time \(t\) is measured in **years**.
- Rates are **continuously compounded** unless stated otherwise.
- \(S_t\): underlying price at time \(t\).
- \(r\): continuously-compounded risk-free rate (constant in basic models).
- \(B_t = e^{rt}\): money-market account (numéraire in the basic setting).
- \(W_t\): Brownian motion.
- \(\mathbb{P}\): real-world (physical) probability measure.
- \(\mathbb{Q}\): risk-neutral measure under which discounted traded prices are martingales.

When a continuous dividend yield \(q\) is relevant, the risk-neutral drift becomes \(r-q\).
