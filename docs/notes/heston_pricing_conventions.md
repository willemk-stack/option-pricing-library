# Heston Pricing Conventions

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

The implementation does **not** require the Feller condition

\[
2\kappa\bar v \ge \eta^2.
\]

This is intentional. The Heston model can be priced outside the Feller region, but calibration diagnostics should report whether a fitted parameter set violates it.

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

The repository uses two Heston probabilities:

\[
P_0(K,T)
\quad\text{and}\quad
P_1(K,T).
\]

The public call formula is

\[
C(K,T)
=
D(0,T)\left(F(0,T)P_1(K,T) - K P_0(K,T)\right).
\]

Therefore:

- `p0` / probability index `0` multiplies the strike term.
- `p1` / probability index `1` multiplies the forward term.

Equivalently:

| Repository probability | Formula role |
|---|---|
| `p0`, `probability_index=0` | strike-discount term |
| `p1`, `probability_index=1` | forward-discount term |

This naming should be treated as part of the public implementation contract.

---

## 6. Call price formula

For a European call with strike \(K\) and maturity \(T\):

\[
C(K,T)
=
D(0,T)\left(F(0,T)P_1(K,T) - K P_0(K,T)\right).
\]

In implementation terms:

```python
call = df * (forward * p1 - strike * p0)
```

where

```python
forward = ctx.fwd(tau)
df = ctx.df(tau)
x = log(forward / strike)
p0 = heston_probability(x=x, tau=tau, params=params, probability_index=0)
p1 = heston_probability(x=x, tau=tau, params=params, probability_index=1)
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
D(0,T)\left(K(1-P_0(K,T)) - F(0,T)(1-P_1(K,T))\right).
\]

In implementation terms:

```python
put = df * (strike * (1.0 - p0) - forward * (1.0 - p1))
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
D\left(FP_1 - KP_0\right)
-
D\left(K(1-P_0) - F(1-P_1)\right) \\
&=
D\left(FP_1 - KP_0 - K + KP_0 + F - FP_1\right) \\
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

The near-deterministic case \(\eta \approx 0\) should also remain numerically close to the Black-76 limit, but tolerances may need to be looser because the generic Heston Fourier formula can become ill-conditioned near singular parameter regimes.

---

## 10. Backend convention

The repository supports two probability integration backends:

| Backend | Role |
|---|---|
| `gauss_legendre` | Production/default fixed-rule backend |
| `quad` | Reference/comparison backend using SciPy adaptive integration |

The intended default is `gauss_legendre`, because it is vectorizable and gives better control over diagnostics. The `quad` backend is useful as an independent comparison path, especially during validation.

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

A pricing result deserves review if diagnostics show:

- non-finite integrand values,
- non-finite total integral,
- probability outside \([0,1]\) beyond tolerance,
- large tail contribution,
- excessive cancellation,
- too many bad panels,
- large disagreement between `gauss_legendre` and `quad`,
- material price movement under denser quadrature settings,
- jagged implied-vol behavior across neighboring strikes.

A severe diagnostic warning does not automatically mean the price is unusable. It means the point should be rerun, compared across backends/configurations, and documented if retained.

---

## 12. Calibration implications

Calibration uses the same pricing convention. This matters because a small sign or index error can still produce smooth prices while corrupting parameter estimates.

Calibration diagnostics should therefore report:

- objective function,
- backend and quadrature settings,
- fitted parameters,
- Feller condition status,
- in-sample residuals,
- held-out residuals when available,
- multi-start dispersion,
- parameter sensitivity,
- whether any quotes were priced with diagnostic warnings.

The price Jacobian convention follows the call formula:

\[
\frac{\partial C}{\partial \theta_i}
=
D(0,T)
\left(
F(0,T)\frac{\partial P_1}{\partial \theta_i}
-
K\frac{\partial P_0}{\partial \theta_i}
\right).
\]

For puts, because the constants vanish under differentiation,

\[
\frac{\partial P}{\partial \theta_i}
=
D(0,T)
\left(
F(0,T)\frac{\partial P_1}{\partial \theta_i}
-
K\frac{\partial P_0}{\partial \theta_i}
\right).
\]

Thus, under this convention, call and put parameter Jacobians have the same Heston-parameter derivative expression for the same strike and maturity.

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

- Call formula uses `forward * p1 - strike * p0`.
- Put formula uses `strike * (1 - p0) - forward * (1 - p1)`.
- Put-call parity holds across representative strikes.
- `x = log(forward / strike)` is used consistently.

### Limit tests

- \(\eta = 0\) deterministic-variance Heston matches Black-76.
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
- Euler is treated as baseline, not production default.
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

call = df * (forward * p1 - strike * p0)
```

The put should be computed as:

```python
put = df * (strike * (1.0 - p0) - forward * (1.0 - p1))
```

And the parity error should be approximately zero:

```python
parity_error = (call - put) - df * (forward - strike)
```

---

## 16. Open implementation decisions

These are allowed to remain open during development, but should be resolved before final Capstone 3 submission.

1. Whether the root package should export Heston helpers directly or keep them under `option_pricing.models.heston` and `option_pricing.pricers.heston`.
2. Whether calibration should enforce the Feller condition, penalize violations, or only report violations.
3. Which objective should be the default calibration objective.
4. Which quadrature quality level should be used in final calibration notebooks.
5. Which warning flags should block calibration quotes automatically.
6. Whether analytic Jacobians should be default-on after finite-difference validation is complete.
7. Whether local-vol comparison should use a synthetic eSSVI surface, a market-like fixture, or a Heston-generated surface.

---

## 17. Final implementation contract

The repository's Heston pricing convention can be summarized as:

\[
x = \log(F/K),
\]

\[
C = D(FP_1 - KP_0),
\]

\[
P = D(K(1-P_0) - F(1-P_1)).
\]

where:

- `probability_index=0` corresponds to \(P_0\), the strike-term probability,
- `probability_index=1` corresponds to \(P_1\), the forward-term probability,
- the Fourier phase \(e^{iux}\) is applied exactly once,
- `gauss_legendre` is the production/default backend,
- `quad` is the independent comparison backend,
- diagnostic warnings must be surfaced, not hidden.

Any future pricing, calibration, diagnostics, or Monte Carlo validation code should preserve this contract.
