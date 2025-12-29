# Monte Carlo option pricing

Monte Carlo (MC) methods price derivatives by simulating the underlying under the **risk-neutral measure**
and averaging discounted payoffs. MC is flexible: it handles high dimensions and path-dependent payoffs naturally.
Its trade-off is statistical noise (convergence is \(O(1/\sqrt{N})\)).

## Objectives

- Connect Monte Carlo to [risk-neutral pricing](risk_neutral_pricing.md).
- Build the basic MC estimator for European options under GBM.
- Understand standard error / confidence intervals.
- See the main variance-reduction ideas used in practice.

## 1. Risk-neutral pricing recap

Let \(V_t\) be the value at time \(t\) of a claim paying \(H_T\) at \(T\).
Then

\[
V_t = \mathbb{E}^{\mathbb{Q}}\!\left[\left.D(t,T)\,H_T\right|\mathcal{F}_t\right].
\]

In the basic constant-rate case, \(D(0,T)=e^{-rT}\) and

\[
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[H_T].
\]

## 2. Monte Carlo estimator

Suppose we can sample \(H_T^{(i)}\) under \(\mathbb{Q}\) for \(i=1,\dots,N\).
The MC price estimator is

\[
\widehat V_0 = e^{-rT}\,\frac{1}{N}\sum_{i=1}^N H_T^{(i)}.
\]

### Error bars

Let \(\sigma_H^2 = \operatorname{Var}^{\mathbb{Q}}(H_T)\).
Then

\[
\operatorname{SE}(\widehat V_0) \approx e^{-rT}\,\frac{\widehat\sigma_H}{\sqrt{N}},
\]

and (approximately) a 95% confidence interval is

\[
\widehat V_0 \pm 1.96\,\operatorname{SE}(\widehat V_0).
\]

The slow \(1/\sqrt{N}\) rate is why variance reduction matters.

## 3. Example: European call under Black–Scholes

Payoff:

\[
H_T = (S_T-K)^+.
\]

Under \(\mathbb{Q}\) with no dividends,

\[
dS_t = rS_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}},
\]

so we can sample \(S_T\) exactly:

\[
S_T = S_0\exp\!\left((r-\tfrac12\sigma^2)T + \sigma\sqrt{T}\,Z\right),
\qquad Z\sim\mathcal N(0,1).
\]

**Algorithm (exact sampling)**

```text
for i = 1..N:
    Z ~ Normal(0,1)
    ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    payoff[i] = max(ST - K, 0)
price = exp(-r*T) * mean(payoff)
stderr = exp(-r*T) * std(payoff, ddof=1)/sqrt(N)
```

This is the cleanest baseline because there is **no time-discretization error** for European options under GBM.

## 4. Variance reduction (what usually helps first)

### Antithetic variates

Use \(Z\) and \(-Z\) together:

\[
\widehat H = \tfrac12\big(g(Z)+g(-Z)\big).
\]

This often reduces variance for monotone payoffs with almost no extra complexity.

### Control variates (often the biggest win)

Pick a random variable \(Y\) with known expectation \(\mathbb{E}[Y]\), correlated with the payoff.
For example, for a European call under GBM, the Black–Scholes price is known, so you can use the call payoff itself
as a control via conditional expectations, or use \(S_T\) (whose expectation is known under \(\mathbb{Q}\)).

A common control estimator is

\[
\widehat V^{\text{cv}} = \widehat V - \beta\,(\widehat Y - \mathbb{E}[Y]),
\]

with \(\beta\) chosen (empirically) to minimize variance.

### Stratified / quasi-Monte Carlo (QMC)

Replacing pseudo-random draws with low-discrepancy sequences can dramatically reduce error for smooth payoffs,
especially in low/moderate dimension. QMC needs careful scrambling and diagnostics but is widely used.

## 5. Path-dependent payoffs

For payoffs depending on the full path (Asian, barrier, etc.), you typically discretize time and simulate increments:

\[
\Delta W_k \sim \mathcal N(0,\Delta t).
\]

Then you have both **statistical error** and **time-discretization error**. For GBM you can still sample exactly
between grid points (lognormal bridges), but for more complex models you may need Euler / Milstein schemes.

## Where to go next

- Compare MC to the [Black–Scholes benchmark](bs_pricing.md) for European options.
- For discrete-time intuition and early exercise, see [Binomial CRR](binomial_crr.md).
