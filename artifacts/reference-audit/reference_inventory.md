# Reference inventory

Scope: authored documentation scanned under `docs/notes`, `docs/user_guides`, `docs/api`, `docs/architecture`, `docs/validation_matrix.md`, `README.md`, and `README.template.md`.

Raw scan files:

- `reference_sections.txt`: pages with `## References` sections.
- `reference_mentions.txt`: raw author/title keyword hits.
- `suspicious_markers.txt`: TODO/source-needed marker scan.
- `strong_claims_raw.txt`: raw strong-language scan.

## Final normalized references by page

### `docs/notes/foundations/brownian_motion_and_ito.md`
- Shreve, *Stochastic Calculus for Finance II*.

### `docs/notes/foundations/gbm.md`
- Shreve, *Stochastic Calculus for Finance II*.
- Hull, *Options, Futures, and Other Derivatives*.

### `docs/notes/foundations/risk_neutral_pricing.md`
- Shreve, *Stochastic Calculus for Finance II*.
- Hull, *Options, Futures, and Other Derivatives*.

### `docs/notes/pricing/binomial_crr.md`
- Shreve, *Stochastic Calculus for Finance I*.
- Hull, *Options, Futures, and Other Derivatives*.

### `docs/notes/pricing/bs_pricing.md`
- Shreve, *Stochastic Calculus for Finance II*.
- Hull, *Options, Futures, and Other Derivatives*.

### `docs/notes/pricing/mc.md`
- Glasserman, *Monte Carlo Methods in Financial Engineering*.

### `docs/notes/local-vol-pde/dupire_local_vol.md`
- Dupire, "Pricing and hedging with smiles".
- Gatheral, *The Volatility Surface*.

### `docs/notes/local-vol-pde/finite_difference_pde.md`
- Duffy, *Finite Difference Methods in Financial Engineering*.
- Tavella and Randall, *Pricing Financial Instruments: The Finite Difference Method*.
- Pooley, Vetzal, and Forsyth, "Convergence remedies for non-smooth payoffs in option pricing".

### `docs/notes/local-vol-pde/pde_convergence.md`
- Pooley, Vetzal, and Forsyth, "Convergence remedies for non-smooth payoffs in option pricing".

### `docs/notes/diagnostics/integration_quadrature.md`
- Davis and Rabinowitz, *Methods of Numerical Integration*.
- Gautschi, *Orthogonal Polynomials: Computation and Approximation*.
- Hale and Townsend, "Fast and accurate computation of Gauss-Legendre and Gauss-Jacobi quadrature nodes and weights".

### `docs/notes/volatility/implied_volatility.md`
- Hull, *Options, Futures, and Other Derivatives*.
- Gatheral, *The Volatility Surface*.
- Sinclair, *Volatility Trading*.

### `docs/notes/volatility/interpolation.md`
- Fritsch and Carlson, "Monotone piecewise cubic interpolation".
- Wolberg and Alfy, "An energy-minimization framework for monotonic cubic spline interpolation".
- Fengler, "Arbitrage-free smoothing of the implied volatility surface".
- Busing, "Monotone regression: a simple and fast O(n) PAVA implementation".
- Cr?pey, "Tikhonov regularization".
- de Boor, *A Practical Guide to Splines*.

### `docs/notes/volatility/svi_calibration_design.md`
- Gatheral, *The Volatility Surface*.
- Gatheral, "Arbitrage-free SVI volatility surfaces" presentation.
- Gatheral and Jacquier, "Arbitrage-free SVI volatility surfaces".
- Lee, "The moment formula for implied volatility at extreme strikes".

### `docs/notes/volatility/essvi_calibration_design.md`
- Gatheral and Jacquier, "Arbitrage-free SVI volatility surfaces".
- Hendriks and Martini, "The extended SSVI volatility surface".
- Mingone, "No arbitrage global parametrization for the eSSVI volatility surface".
- Pasquazzi, "eSSVI surface calibration".

### `docs/notes/heston/heston_stochastic_vol.md`
- Heston, "A closed-form solution for options with stochastic volatility...".
- Gatheral, *The Volatility Surface*.
- Albrecher, Mayer, Schoutens, and Tistaert, "The Little Heston Trap".

### `docs/notes/heston/heston_pricing_conventions.md`
- Heston; Gatheral; Albrecher et al.; Cui, del Ba?o Rollin, and Germano; Christoffersen and Jacobs; Andersen; Glasserman; Davis and Rabinowitz; Gautschi; Hale and Townsend.

### `docs/notes/heston/heston_fourier_diagnostics.md`
- Heston; Gatheral; Albrecher et al.; Davis and Rabinowitz; Hale and Townsend.

### `docs/notes/heston/heston_quadrature_policy.md`
- Heston; Gatheral; Albrecher et al.; Davis and Rabinowitz; Gautschi; Hale and Townsend; Trefethen.

### `docs/notes/heston/heston_monte_carlo.md`
- Heston; Andersen; Broadie and Kaya; Glasserman.

### `docs/notes/heston/heston_qe_notes.md`
- Andersen; Broadie and Kaya.

### `docs/notes/heston/heston_calibration.md`
- Heston; Gatheral; Cui, del Ba?o Rollin, and Germano; Christoffersen and Jacobs.

### `docs/notes/heston/heston_calibration_seeds.md`
- Heston; Gatheral; Cui, del Ba?o Rollin, and Germano.

### `docs/notes/heston/heston_calibration_benchmark_diagnostics.md`
- Heston; Gatheral; Cui, del Ba?o Rollin, and Germano.

### `docs/notes/heston/heston_vs_local_vol.md`
- Heston; Dupire; Gatheral; Gatheral and Jacquier.

## Pages without formal reference sections

The scan found no formal `## References` sections in `docs/user_guides`, `docs/api`, `docs/architecture`, `docs/validation_matrix.md`, `README.md`, or `README.template.md`; those pages link to proof artifacts and notes rather than maintaining bibliographies.
