# Heston stochastic volatility

!!! note "Status: literature-backed convention"
    This note summarizes the model setup and pricing conventions used by the
    Heston implementation notes. Repository-specific numerical policy and
    validation evidence are separated into the adjacent Heston notes.

The Heston model is a standard extension of Black–Scholes when one constant volatility is no longer enough. It keeps continuous-time diffusion pricing, but lets the variance itself evolve randomly and lets spot and variance shocks be correlated. This is one of the simplest stochastic-volatility models that can generate persistent skew and smile while still leaving European vanilla pricing semi-analytic.

## Context

This note assumes familiarity with:

- [Risk-neutral pricing](../foundations/risk_neutral_pricing.md)
- [Brownian motion and Itô](../foundations/brownian_motion_and_ito.md)
- [Black-Scholes pricing](../pricing/bs_pricing.md)
- [Finite Differences](../local-vol-pde/finite_difference_pde.md)
- [Dupire local vol](../local-vol-pde/dupire_local_vol.md)

A useful progression is:

**Black–Scholes → local volatility → stochastic volatility.**

Local volatility matches the vanilla surface with a deterministic function
\(\sigma_{\mathrm{loc}}(S,t)\). Heston instead treats variance itself as a stochastic state variable.

## Conventions and notation

Use the notation:

- \(S_t\): spot price,
- \(v_t\): instantaneous variance,
- \(r\): continuously compounded risk-free rate,
- \(q\): continuous dividend yield when relevant,
- \(\tau = T-t\): time to expiry,
- \(F_{t,T} = S_t e^{(r-q)\tau}\): forward price in the flat-rate case,
- \(y = \log(K/F_{t,T})\): log-moneyness.

In Heston and Gatheral derivations it is also common to use

\[
x = \log(F_{t,T}/K) = -y.
\]

For parameters, use

- \(\kappa\): variance mean-reversion speed,
- \(\theta\): long-run variance,
- \(\xi\): volatility of variance,
- \(\rho\): spot/variance correlation,
- \(v_0\): initial variance.

This keeps the notation separate from the different uses of \(\lambda\) in the literature.

## 1. Why Black–Scholes is not enough

Under Black–Scholes, the underlying follows GBM with one constant volatility \(\sigma\). That model is analytically elegant, but too rigid to explain the skew and smile patterns seen in market implied volatilities.

Heston keeps diffusion pricing, replaces constant volatility with a stochastic variance process, correlates spot and variance shocks, and preserves an affine structure that keeps European vanilla pricing tractable.

## 2. Model dynamics

### Under the real-world measure

A modern presentation writes

\[
dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_1^{\mathbb{P}},
\]

\[
dv_t = \kappa_{\mathbb{P}}(\theta_{\mathbb{P}} - v_t)dt + \xi\sqrt{v_t}\,dW_2^{\mathbb{P}},
\]

with

\[
d\langle W_1^{\mathbb{P}}, W_2^{\mathbb{P}} \rangle_t = \rho\,dt.
\]

Heston’s original paper reaches the square-root variance process by first specifying an OU process for \(\sqrt{v_t}\). For pricing and implementation, it is cleaner to write the CIR-style variance process directly.

### Under the pricing measure

For valuation, the relevant dynamics are the risk-neutral ones:

\[
dS_t = (r-q) S_t\,dt + \sqrt{v_t}\,S_t\,dW_1^{\mathbb{Q}},
\]

\[
dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}\,dW_2^{\mathbb{Q}},
\]

with

\[
d\langle W_1^{\mathbb{Q}}, W_2^{\mathbb{Q}} \rangle_t = \rho\,dt.
\]

These are the parameters calibrated to option prices.

## 3. Why Heston needs a volatility-risk premium

Once variance is stochastic, the stock is no longer the only state variable relevant for option pricing. The claim value depends on both \(S_t\) and \(v_t\). That creates an extra source of risk, while variance itself is not directly traded in the basic setup.

The market is therefore generally incomplete. No-arbitrage alone does not pin down a unique option price unless the pricing of variance risk is also specified.

That is why Heston’s original PDE contains a volatility-risk term. If \(U(S,v,t)\) denotes the claim value, the PDE contains

\[
\lambda_v(S,v,t)\,U_v,
\]

where \(\lambda_v\) is the market price of volatility risk.

The logic is:

**stochastic variance → extra state variable → incomplete market → volatility-risk premium → pricing PDE.**

Heston motivates a proportional specification such as

\[
\lambda_v(S,v,t) = \bar\lambda v.
\]

In practice, many presentations work directly under \(\mathbb{Q}\), where this pricing adjustment is already absorbed into the variance drift.

## 4. Risk-neutral implementation view

For pricing and coding, the convenient form is the risk-neutral one:

- the stock drift is \(r-q\),
- the variance process already carries its pricing-measure drift,
- and European option pricing can proceed from the characteristic function.

This is the implementation form of the same model.

## 5. The pricing PDE

Under the original Heston setup, a sufficiently smooth claim \(U(S,v,t)\) satisfies a two-factor PDE of the form

\[
U_t + \tfrac12 vS^2 U_{SS} + \rho\xi vS U_{Sv} + \tfrac12 \xi^2 v U_{vv}
+ (r-q)S U_S + \bigl(\kappa(\theta-v)-\lambda_v(S,v,t)\bigr)U_v - rU = 0.
\]

The terms have the interpretation:

- \(\tfrac12 vS^2 U_{SS}\): spot curvature,
- \(\rho\xi vS U_{Sv}\): coupling between spot and variance shocks,
- \(\tfrac12 \xi^2 v U_{vv}\): variance curvature,
- \((r-q)S U_S - rU\): financing and carry,
- \((\kappa(\theta-v)-\lambda_v)U_v\): effective variance drift under pricing,
- \(U_t\): time evolution.

Relative to Black–Scholes, valuation is now a PDE in two state variables rather than one.

## 6. European call decomposition

A European call can still be written in a Black–Scholes-like form:

\[
C_t = e^{-q\tau} S_t P_1 - e^{-r\tau} K P_2.
\]

Equivalently, in forward form,

\[
C_t = D(t,T)\bigl(F_{t,T} P_1 - K P_2\bigr).
\]

The quantities \(P_1\) and \(P_2\) are not physical probabilities. They are pricing probabilities induced by the PDE.

Gatheral often writes

\[
C(x,v,\tau) = K\bigl(e^x P_1(x,v,\tau) - P_0(x,v,\tau)\bigr),
\]

with \(x = \log(F_{t,T}/K)\). In that notation, Gatheral’s \(P_0\) plays the same role as Heston’s \(P_2\).

## 7. Why characteristic functions solve the problem

After substituting the call decomposition into the PDE, the pricing problem reduces to PDEs for the pseudo-probabilities \(P_j\). Those functions are not available in a simple elementary form.

The natural route is to solve for their characteristic functions.

For a random variable \(X\), the characteristic function is

\[
\psi_X(u) = \mathbb{E}[e^{iuX}].
\]

In the Heston model, the affine variance process implies an exponential-affine transform for the relevant log-price characteristic functions:

\[
f_j(x,v,\tau;u) = \exp\bigl(C_j(\tau,u) + D_j(\tau,u)v + iux\bigr).
\]

Once this exponential-affine form is imposed, the PDE reduces to ODEs for \(C_j\) and \(D_j\). The quantities \(P_1\) and \(P_2\) are then recovered by Fourier inversion.

The pricing chain is:

**state dynamics → pricing PDE → pseudo-probabilities → characteristic functions → Fourier inversion → call price.**

## 8. What the parameters do

A useful map is:

- \(\rho\): mainly controls skew direction and strength,
- \(\xi\): mainly controls smile intensity and tail thickness,
- \(\kappa\): controls how quickly variance mean reverts,
- \(\theta\): sets the long-run variance level,
- \(v_0\): sets the short-maturity variance level.

Model-intuition note: in the Heston setup, negative \(\rho\) is the parameter
that tilts equity-style smiles toward higher lower-strike implied volatilities.
Random variance contributes smile and tail effects, but this note treats the
correlation/skew link as practitioner intuition from the Heston/Gatheral model
reading, not as a calibrated-market universal.

## 9. Notation map: Heston vs Gatheral

### Variance-process parameters

Use

\[
(\kappa, \theta, \xi, \rho, v_0).
\]

Gatheral’s Chapter 2 variance process is often written as

\[
dv_t = -\lambda(v_t-\bar v)dt + \eta\sqrt{v_t}\,dZ_2.
\]

The dictionary is

\[
\lambda_{\text{Gatheral}} = \kappa,
\qquad
\bar v = \theta,
\qquad
\eta = \xi.
\]

### The lambda collision

There are two different lambdas in the literature:

1. Heston’s volatility-risk premium \(\lambda_v(S,v,t)\),
2. Gatheral’s mean-reversion speed \(\lambda\).

They are different objects.

It is therefore safer to reserve

- \(\kappa\) for mean reversion,
- \(\lambda_v\) for the volatility-risk premium.

### Forward/log-moneyness notation

Gatheral often uses

\[
x = \log(F/K),
\]

while many volatility notes use

\[
y = \log(K/F).
\]

Hence

\[
x = -y.
\]

That sign convention should be kept explicit when moving between derivations and code.

## 10. Numerical implementation notes

### Stable characteristic-function representation

Equivalent formulas are not always numerically equivalent. The main issue is the complex logarithm inside the Heston characteristic function. A naive principal-branch evaluation can produce discontinuities when the argument crosses the negative real axis.

Typical symptoms are:

- jagged integrands,
- unstable prices,
- noisy implied-vol curves,
- fragile calibration.

This is why the repository follows the Gatheral-style algebraic rearrangement
and branch-handling policy discussed in
[Heston Fourier diagnostics](heston_fourier_diagnostics.md). That is an
implementation-stability choice backed by the branch-stability tests, not a
claim that every equivalent formula has been benchmarked here.

### Narrow scope for a first pricer

A first Heston implementation can focus on European vanillas only:

- one parameter object \((\kappa,\theta,\xi,\rho,v_0)\),
- forward and discount inputs consistent with the rest of the library,
- characteristic-function pricing with stable log handling,
- numerical inversion for \(P_1\) and \(P_2\),
- a Black–Scholes sanity limit where appropriate,
- and a Monte Carlo cross-check on a small vanilla test set.

### Practical parameter checks

Useful checks include:

- \(v_0 \ge 0\), \(\theta \ge 0\), \(\xi \ge 0\), \(|\rho| < 1\), \(\kappa > 0\),
- the Feller condition \(2\kappa\theta \ge \xi^2\).

The Feller condition is an important process-quality diagnostic. The repository
reports violations as review evidence rather than treating them as an automatic
vanilla-pricing failure, because the supported Fourier pricing path is tested
on the configured regimes separately from process-boundary interpretation.

### Calibration parameter transforms

The calibration entrypoint supports two raw-optimizer parameter transforms.

`parameter_transform="unconstrained"` keeps the original admissibility transform: softplus for positive parameters \((\kappa,\theta,\xi,v_0)\) and tanh for \(\rho\). This guarantees mathematical admissibility, but it does not encode a finite practical calibration box.

`parameter_transform="bounded"` maps each raw optimizer variable through a sigmoid into configured finite bounds:

\[
p = p_{\min} + (p_{\max} - p_{\min})\,\sigma(u).
\]

This keeps the optimizer unconstrained in raw space while the transform guarantees the fitted Heston parameters remain inside the practical interval for all five parameters. No SciPy optimizer-side bounds are needed in this mode.

The bounded transform is useful for generic unconstrained optimizers, multi-start raw seed grids, and calibration setups where the permitted domain should live in the transform itself. The trade-off is that sigmoid gradients vanish near the configured bounds, so near-bound diagnostics remain important.

## Summary

The Heston model is an affine stochastic-volatility extension of Black–Scholes in which variance mean reverts, spot and variance shocks are correlated, and European vanilla prices are recovered from characteristic functions through a Black–Scholes-like decomposition.

Heston's original paper emphasizes the economics and PDE structure. Gatheral
emphasizes the risk-neutral implementation route, while Albrecher, Mayer,
Schoutens, and Tistaert document why branch handling in Heston characteristic
functions is a numerical issue rather than cosmetic algebra. These are
complementary views of the same model.

## References

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327-343.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2006). The Little Heston Trap.
