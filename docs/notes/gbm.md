# Geometric Brownian motion

Geometric Brownian motion (GBM) is the standard baseline model for an equity price:
it produces **lognormal** prices and **normally distributed log-returns**.
It is simple enough to solve in closed form and is the starting point for Black–Scholes, Monte Carlo, and binomial-tree limits.

## Model definition

Under a chosen measure (\(\mathbb{P}\) or \(\mathbb{Q}\)), GBM assumes
\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,\qquad S_0>0,
\]
with constant volatility \(\sigma>0\). Under risk-neutral pricing (no dividends), \(\mu=r\);
with continuous dividend yield \(q\), \(\mu=r-q\).

## Solving the SDE with Itô

Set \(X_t=\ln S_t\). Applying Itô’s lemma to \(X_t\) gives
\[
dX_t = \frac{1}{S_t}dS_t - \frac{1}{2}\frac{1}{S_t^2}(dS_t)^2.
\]
Using \((dW_t)^2=dt\) and \((dS_t)^2 = (\sigma S_t)^2 dt\), we obtain
\[
dX_t = \left(\mu - \tfrac{1}{2}\sigma^2\right)dt + \sigma\,dW_t.
\]
Integrating from \(0\) to \(t\),
\[
\ln S_t
= \ln S_0 + \left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t,
\]
so
\[
\boxed{
S_t = S_0\exp\!\left(\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t\right).
}
\]

## Distributional consequences

Because \(W_t\sim\mathcal N(0,t)\),

- \(\ln S_t\) is normal:
  \[
  \ln S_t \sim \mathcal N\!\left(\ln S_0 + \left(\mu - \tfrac{1}{2}\sigma^2\right)t,\; \sigma^2 t\right).
  \]
- \(S_t\) is **lognormal**.

Moments (useful sanity checks):

\[
\mathbb{E}[S_t] = S_0 e^{\mu t},\qquad
\operatorname{Var}(S_t) = S_0^2 e^{2\mu t}\left(e^{\sigma^2 t}-1\right).
\]

Conditional distribution (key for simulation): for \(0\le s<t\),
\[
\ln\frac{S_t}{S_s}
\sim \mathcal N\!\left(\left(\mu-\tfrac12\sigma^2\right)(t-s),\;\sigma^2(t-s)\right),
\]
independent of the past given \(S_s\).

## Financial interpretation: returns vs log-returns

In finance we describe performance via **returns**.

- The *simple* return over \([0,T]\) is
  \[
  R_T = \frac{S_T-S_0}{S_0}.
  \]
- The *log-return* is
  \[
  r_T = \ln\frac{S_T}{S_0}.
  \]

Log-returns correspond to **continuous compounding** and have a key additivity property:
for a partition \(0=t_0<t_1<\dots<t_n=T\),
\[
\ln\frac{S_T}{S_0}
= \sum_{k=1}^n \ln\frac{S_{t_k}}{S_{t_{k-1}}}.
\]

Under GBM,
\[
\ln\frac{S_T}{S_0}
\sim \mathcal N\!\Big(\big(\mu-\tfrac12\sigma^2\big)T,\;\sigma^2 T\Big).
\]

## Appendix A: Continuous compounding and log-returns (derivation)

This appendix explains why log-returns are the natural “continuously compounded” quantity.

### From discrete to continuous compounding

A rate \(R\) compounded once over \([0,T]\) gives \(S_T=S_0(1+R)\).
If it compounds \(n\) times per year with nominal rate \(R\),
\[
S_T = S_0\left(1+\frac{R}{n}\right)^{nT}.
\]
As \(n\to\infty\), this converges to continuous compounding:
\[
S_T = S_0 e^{RT}.
\]
Taking logs,
\[
\ln\frac{S_T}{S_0} = RT,
\]
which motivates using log-returns as the additive quantity.

### Simple returns vs log-returns

For small returns, \(\ln(1+x)\approx x\), so simple and log-returns are close when moves are small.
For large moves, log-returns behave better analytically and in aggregation.

### Time additivity of log-returns

For any \(a<b<c\),
\[
\ln\frac{S_c}{S_a} = \ln\frac{S_c}{S_b} + \ln\frac{S_b}{S_a}.
\]
This additivity is why models often work in log space.

### Distribution of log-returns under GBM

From the explicit GBM solution,
\[
\ln\frac{S_T}{S_0} = \left(\mu-\tfrac12\sigma^2\right)T + \sigma W_T,
\]
so it is normal with variance \(\sigma^2 T\). Annualized variance over \([0,T]\) is \(\sigma^2\).

## Takeaways

- GBM implies **lognormal prices** and **normal log-returns**.
- Under pricing (risk-neutral) dynamics, the drift is \(r\) (or \(r-q\)).
- The explicit solution makes simulation straightforward and underpins both Black–Scholes and Monte Carlo.

## Where to go next

- Plug \(\mu=r\) into the GBM distribution to derive the [Black–Scholes formula](bs_pricing.md).
- Use the conditional distribution to build a [Monte Carlo pricer](mc.md).
