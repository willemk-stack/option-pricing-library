# Heston Pricing Conventions

!!! note "Status"
    This note gives model background and notation for the repository's Heston implementation.
    Mathematical claims should be tied to the references below; repository-specific defaults
    are documented separately in the implementation-policy notes.

This note fixes the pricing conventions used by the Heston Fourier pricer in this repository. It is intended to remove ambiguity around the state variable, log-moneyness input, probability indices, discounting convention, and call/put formulas.

The purpose is not to re-derive the full Heston model. The purpose is to make the implementation contract explicit enough that pricing, calibration, diagnostics, and Monte Carlo validation can all refer to the same convention.

---

## 1. Model setup

Under the risk-neutral pricing measure, the Heston model is

\[
\begin{aligned}
dS_t &= (r_t - q_t) S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S, \\
dv_t &= \kappa(\bar v - v_t)\,dt + \eta\sqrt{v_t}\,dW_t^v, \\
d\langle W^S, W^v\rangle_t &= \rho\,dt.
\end{aligned}
\]

The repository parameter object uses the following names:

| Repository name | Common notation | Meaning |
|---|---:|---|
| `kappa` | \(\kappa\) | Mean-reversion speed of variance |
| `vbar` | \(\bar v\) or \(\theta\) | Long-run mean variance |
| `eta` | \(\eta\) or \(\xi\) | Volatility of variance / vol-of-vol |
| `rho` | \(\rho\) | Spot-variance correlation |
| `v` | \(v_0\) | Initial variance |

The parameter object validates basic admissibility:

- \(\kappa > 0\)
- \(\bar v \ge 0\)
- \(\eta \ge 0\)
- \(v_0 \ge 0\)
- \(\rho \in [-1, 1]\)

The implementation does **not** hard-enforce the Feller condition

\[
2\kappa\bar v \ge \eta^2.
\]

This is intentional. The Heston model can be priced outside the Feller region.
Calibration diagnostics report the Feller value/margin and whether the fitted
parameter set violates it. A soft Feller penalty can be enabled through
regularization configuration, but violation is not blocked by default.

---

## 2. Market convention

The public pricer uses a curves-first market context.

For a maturity \(T\), define:

\[
D(0,T)
\]

as the discount factor and

\[
F(0,T)
\]

as the forward price for delivery at \(T\).

For flat rates and dividend yield,

\[
D(0,T) = e^{-rT},
\qquad
F(0,T) = S_0 e^{(r-q)T}.
\]

The Heston Fourier integrals are implemented in a forward-normalized convention. Prices are finally discounted by \(D(0,T)\).

---

## 3. Log-moneyness convention

The implementation defines the Fourier log-moneyness input as

\[
x = \log\left(\frac{F(0,T)}{K}\right).
\]

This is **forward over strike**, not strike over forward.

Consequences:

- At-the-money forward means \(x = 0\).
- In-the-money calls have \(x > 0\).
- Out-of-the-money calls have \(x < 0\).

The public Heston pricers compute this internally from the market context and strike:

\[
x = \log(F/K).
\]

Users of the public call/put pricers should not pass `x` directly. Low-level probability diagnostics may expose `x` because they operate directly on the Fourier integral.

---

## 4. Characteristic-function convention

The low-level characteristic function is written for the forward log-return style variable used inside the Fourier inversion. The implementation uses affine coefficients \(C\) and \(D\) such that the transform factor has the form

\[
\exp\left(C(u,T)\bar v + D(u,T)v_0 + iu x\right).
\]

The `x` phase is applied only once in the Fourier integrand. A key implementation invariant is:

\[
\text{integrand}(u,x,T)
=
\Re\left[
\frac{e^{iux} A_j(u,T)}{iu}
\right],
\]

where \(A_j(u,T)\) is the Heston affine factor for probability index \(j\).

The production code should avoid any convention where the log-moneyness phase is applied both inside the characteristic function and again inside the integrand.

---

## 5. Probability-index convention

The repository uses two semantic Heston probabilities:

\[
P_K(K,T)
\quad\text{and}\quad
P_F(K,T).
\]

The public call formula is

\[
C(K,T)
=
D(0,T)\left(F(0,T)P_F(K,T) - K P_K(K,T)\right).
\]

Therefore:

- `P_K` / probability index `0` multiplies the strike/cash term.
- `P_F` / probability index `1` multiplies the forward/asset term.
- `p0`, `p1`, `P_0`, and `P_1` are legacy aliases for the same public convention.

Equivalently:

| Repository probability | Formula role |
|---|---|
| `P_K`, `p0`, `probability_index=0` | strike-discount term |
| `P_F`, `p1`, `probability_index=1` | forward-discount term |

This mapping is verified against `option_pricing.pricers.heston`, where the
call helper computes `df * (forward * P_F - strike * P_K)` and passes
probability index `0` to the strike leg and `1` to the forward leg.

This naming should be treated as part of the public implementation contract.

---

## 6. Call price formula

For a European call with strike \(K\) and maturity \(T\):

\[
C(K,T)
=
D(0,T)\left(F(0,T)P_F(K,T) - K P_K(K,T)\right).
\]

In implementation terms:

```python
call = df * (forward * P_F - strike * P_K)
```

where

```python
forward = ctx.fwd(tau)
df = ctx.df(tau)
x = log(forward / strike)
P_K = heston_probability(x=x, tau=tau, params=params, probability_index=0)
P_F = heston_probability(x=x, tau=tau, params=params, probability_index=1)
```

For zero maturity, the public pricer should bypass Fourier integration and return intrinsic value:

\[
C(K,0) = \max(S_0 - K, 0).
\]

---

## 7. Put price formula

The put price is implemented using the same probabilities and put-call parity:

\[
P(K,T)
=
D(0,T)\left(K(1-P_K(K,T)) - F(0,T)(1-P_F(K,T))\right).
\]

In implementation terms:

```python
put = df * (strike * (1.0 - P_K) - forward * (1.0 - P_F))
```

For zero maturity, the public pricer should bypass Fourier integration and return intrinsic value:

\[
P(K,0) = \max(K - S_0, 0).
\]

---

## 8. Put-call parity check

The implementation should satisfy

\[
C(K,T) - P(K,T)
=
D(0,T)(F(0,T)-K).
\]

Using the repository formulas:

\[
\begin{aligned}
C - P
&=
D\left(FP_F - KP_K\right)
-
D\left(K(1-P_K) - F(1-P_F)\right) \\
&=
D\left(FP_F - KP_K - K + KP_K + F - FP_F\right) \\
&=
D(F-K).
\end{aligned}
\]

This should be a permanent regression test across representative Heston parameter regimes.

---

## 9. Deterministic-variance limit

When \(\eta = 0\), the variance path becomes deterministic:

\[
v_t
=
\bar v + (v_0 - \bar v)e^{-\kappa t}.
\]

The integrated variance over \([0,T]\) is

\[
\int_0^T v_t\,dt
=
\bar v T
+
(v_0 - \bar v)\frac{1-e^{-\kappa T}}{\kappa}.
\]

In this limit, Heston prices should collapse to Black-76 prices with equivalent volatility

\[
\sigma_{\mathrm{eq}}
=
\sqrt{
\frac{1}{T}
\left(
\bar v T
+
(v_0 - \bar v)\frac{1-e^{-\kappa T}}{\kappa}
\right)
}.
\]

This is one of the most important correctness tests because it checks:

- discounting convention,
- forward convention,
- strike convention,
- probability-index mapping,
- zero-vol-of-vol handling.

The pricing implementation uses an explicit deterministic-variance fallback
when

```python
abs(params.eta) <= HESTON_ETA_DETERMINISTIC_THRESHOLD  # 1e-8
```

or when `tau == 0`. This keeps vanilla prices stable in the singular
near-zero-vol-of-vol regime. The analytic parameter-Jacobian formulas are not
the deterministic fallback; they remain the stochastic-volatility formulas and
therefore require positive nonzero Fourier frequencies and
`eta >= HESTON_ANALYTIC_JAC_ETA_MIN` (`1e-6`). This floor is a numerical
safeguard for the Cui-style analytic gradients, not a financial lower bound.

---

## 10. Backend convention

The repository supports two probability integration backends:

| Backend | Role |
|---|---|
| `gauss_legendre` | Production/default fixed-rule backend |
| `quad` | Reference/comparison backend using SciPy adaptive integration |

The intended default is `gauss_legendre`, because it is vectorizable and gives better control over diagnostics. The `quad` backend is useful as an independent comparison path, especially during validation.

Analytic Heston parameter Jacobians are default-on only for the production
fixed Gauss-Legendre path:

- `backend="gauss_legendre"`,
- nonzero fixed-rule quadrature nodes,
- `eta >= HESTON_ANALYTIC_JAC_ETA_MIN`,
- parameters inside the documented bounded calibration ranges.

Ordinary non-Jacobian prices may still use `backend="quad"` for reference
checks. Unsupported analytic-Jacobian calibration requests raise clearly rather
than silently falling back. Users who need another backend should request
finite differences explicitly with `use_analytic_jac=False`.

A production pricing run should record:

- backend,
- quadrature configuration,
- maximum frequency cutoff,
- number of panels,
- nodes per panel,
- any diagnostic warnings,
- whether a recommended or explicit quadrature configuration was used.

---

## 11. Diagnostics policy

Fourier pricing should not be treated as trustworthy merely because it returned a finite value.

Calibration quote policy separates hard invalidation from review:

- `block`: non-finite total integral or non-finite probability;
- `quarantine`: probability materially outside \([0,1]\) or persistent backend
  disagreement after robust/diagnostics rerun;
- `review`: high tail fraction, cancellation warnings, isolated oscillation
  spikes, near-origin underresolution, too many bad panels, or large quadrature
  error estimates;
- `ok`: no diagnostic warning.

Blocked and quarantined quotes should not silently enter calibration. Reviewed
quotes may be retained, but calibration summaries should report the count and
reason categories.

---

## 12. Calibration implications

Calibration uses the same pricing convention. This matters because a small sign or index error can still produce smooth prices while corrupting parameter estimates.

Calibration diagnostics should therefore report:

- objective function,
- backend and quadrature settings,
- fitted parameters,
- Feller condition value, margin, and status,
- in-sample residuals,
- held-out residuals when available,
- multi-start dispersion,
- parameter sensitivity,
- blocked/quarantined/review quote counts,
- whether the fit used all supplied quotes or a filtered quote set.

The price Jacobian convention follows the call formula:

\[
\frac{\partial C}{\partial \theta_i}
=
D(0,T)
\left(
F(0,T)\frac{\partial P_F}{\partial \theta_i}
-
K\frac{\partial P_K}{\partial \theta_i}
\right).
\]

For puts, because the constants vanish under differentiation,

\[
\frac{\partial P}{\partial \theta_i}
=
D(0,T)
\left(
F(0,T)\frac{\partial P_F}{\partial \theta_i}
-
K\frac{\partial P_K}{\partial \theta_i}
\right).
\]

Thus, under this convention, call and put parameter Jacobians have the same Heston-parameter derivative expression for the same strike and maturity.

The public price/Jacobian helpers return columns in the order
`[kappa, vbar, eta, rho, v]`. A scalar strike returns Jacobian shape `(5,)`;
an array strike returns `strike.shape + (5,)`.

This should be checked against finite differences.

---

## 13. Monte Carlo validation implications

Monte Carlo validation should compare discounted Monte Carlo estimates against the semi-analytic Fourier price.

For a vanilla option:

\[
\widehat{V}_{MC}
=
D(0,T)\frac{1}{N}\sum_{n=1}^{N} \mathrm{payoff}(S_T^{(n)}).
\]

A Monte Carlo result should be reported with:

- scheme,
- path count,
- timestep count,
- seed,
- antithetic flag,
- control-variate flag,
- estimate,
- standard error,
- confidence interval,
- Fourier benchmark,
- error versus Fourier in standard-error units.

For validation, avoid using the same vanilla payoff as a control variate when the goal is to test whether MC agrees with Fourier. Doing so can hide simulation bias by anchoring the estimator to the Fourier price.

---

## 14. Recommended permanent tests

The following tests should remain in the project permanently.

### Pricing convention tests

- Call formula uses `forward * P_F - strike * P_K`.
- Put formula uses `strike * (1 - P_K) - forward * (1 - P_F)`.
- Put-call parity holds across representative strikes.
- `x = log(forward / strike)` is used consistently.

### Limit tests

- \(\eta = 0\) and \(\eta\) below the deterministic threshold match the
  deterministic-variance Black-76 limit.
- Near-zero \(\eta\) remains close to Black-76 within a documented tolerance.
- Zero maturity returns intrinsic value.

### Stability tests

- `gauss_legendre` and `quad` agree on representative regimes.
- Dense strike slices produce smooth implied-vol curves.
- Branch-cut regression tests catch discontinuities.
- Tail/cancellation diagnostics trigger under deliberately under-resolved quadrature.

### Calibration tests

- Analytic price Jacobian matches finite differences.
- Objective Jacobian matches finite differences after parameter transform.
- Synthetic clean calibration recovers prices and approximately recovers parameters.
- Multi-start returns consistent fit quality or reports non-identifiability.

### Monte Carlo tests

- QE vanilla prices agree with Fourier within confidence intervals.
- QE discounted forward martingale condition holds.
- Repository MC policy treats Euler as a baseline/debugging scheme, not the
  final validation default; the generated
  [MC convergence artifact](../../assets/generated/heston/heston_mc_vs_fourier_convergence.svg)
  reports QE and Euler separately.
- Control variate reduces variance but is not used as the main Fourier validation proof.

---

## 15. Minimal numerical example

Suppose:

```python
spot = 100.0
rate = 0.02
dividend_yield = 0.00
tau = 1.0
strike = 100.0
params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)
```

Then:

\[
D(0,T) = e^{-0.02},
\qquad
F(0,T) = 100e^{0.02},
\qquad
x = \log(F/K).
\]

The call should be computed as:

```python
forward = ctx.fwd(tau)
df = ctx.df(tau)
x = np.log(forward / strike)

p0 = heston_probability(
    x=x,
    tau=tau,
    params=params,
    probability_index=0,
)

p1 = heston_probability(
    x=x,
    tau=tau,
    params=params,
    probability_index=1,
)

P_K = p0  # legacy alias: probability_index=0
P_F = p1  # legacy alias: probability_index=1

call = df * (forward * P_F - strike * P_K)
```

The put should be computed as:

```python
put = df * (strike * (1.0 - P_K) - forward * (1.0 - P_F))
```

And the parity error should be approximately zero:

```python
parity_error = (call - put) - df * (forward - strike)
```

---

## 16. Resolved implementation policies

The final Capstone 3 policy is:

1. Keep Heston helpers namespaced under submodules such as
   `option_pricing.models.heston`, `option_pricing.pricers.heston`, and
   `option_pricing.diagnostics.heston`; do not add broad root-package exports
   for advanced Heston helpers.
2. Enforce positivity/correlation bounds, report Feller diagnostics by default,
   and allow an optional soft Feller penalty. Do not hard-block Feller
   violations by default.
3. Keep `vega_scaled_price` as the backward-compatible calibration default while
   documenting the objective in calibration metadata.
4. Use `robust` quadrature for final capstone reporting tables and plots, with a
   `diagnostics` rerun for headline fitted parameters. Lighter settings are for
   quick interactive exploration only.
5. Block or quarantine hard integration failures as calibration input, and
   soft-review tail/cancellation/oscillation/near-origin warnings.
6. Keep analytic Jacobians default-on only on the guarded fixed
   Gauss-Legendre, eta-floor, bounded-parameter domain. Unsupported requests
   raise or require explicit finite differences.
7. Use a version-controlled deterministic market-like synthetic fixture for the
   Heston-vs-local-vol comparison target. It is not market data and is not
   generated from the fitted Heston model.
8. Include direct Dupire/PDE local-vol repricing on a small validation grid, with
   the eSSVI implied-surface repricing retained only as a supporting proxy.

---

## 17. Final implementation contract

The repository's Heston pricing convention can be summarized as:

\[
x = \log(F/K),
\]

\[
C = D(FP_F - KP_K),
\]

\[
P = D(K(1-P_K) - F(1-P_F)).
\]

where:

- `probability_index=0` corresponds to \(P_K\), the strike-term probability,
- `probability_index=1` corresponds to \(P_F\), the forward-term probability,
- the Fourier phase \(e^{iux}\) is applied exactly once,
- repository implementation policy uses `gauss_legendre` as the default
  backend for reviewed Heston pricing and calibration paths,
- `quad` is the independent comparison backend,
- analytic parameter Jacobians support only the guarded fixed
  Gauss-Legendre production domain with `eta >= 1e-6`,
- the price path uses the deterministic variance fallback for
  `abs(params.eta) <= 1e-8`,
- diagnostic warnings must be surfaced, not hidden,
- final Heston-vs-local-vol reporting uses the deterministic market-like
  synthetic fixture and includes a direct local-vol PDE repricing audit.

Any future pricing, calibration, diagnostics, or Monte Carlo validation code should preserve this contract.

## References

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327-343.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2006). The Little Heston Trap.
- Cui, Y., del Baño Rollin, S., & Germano, G. (2016). Full and fast calibration of the Heston stochastic volatility model.
- Christoffersen, P., & Jacobs, K. (2003). The importance of the loss function in option valuation. CIRANO Scientific Series 2003s-52.
- Andersen, L. (2007). Efficient simulation of the Heston stochastic volatility model.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Davis, P. J., & Rabinowitz, P. (1984). *Methods of Numerical Integration* (2nd ed.). Academic Press.
- Gautschi, W. (2004). *Orthogonal Polynomials: Computation and Approximation*. Oxford University Press.
- Hale, N., & Townsend, A. Fast and accurate computation of Gauss-Legendre and Gauss-Jacobi quadrature nodes and weights.
