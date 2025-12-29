# Black–Scholes–Merton pricing (European options)

The Black–Scholes–Merton (BSM) model is the classic closed-form benchmark for pricing **European** calls and puts.
It combines:

1. a simple stock model (GBM with constant \(\sigma\)),
2. a frictionless market with continuous trading,
3. no-arbitrage replication (or equivalently, risk-neutral pricing).

## Objectives

- State the BSM assumptions and understand where they enter the derivation.
- Derive the Black–Scholes PDE via delta hedging.
- Present the closed-form European call/put formulas and interpret their terms.
- Connect the PDE view to the risk-neutral expectation view.
- Use put–call parity as a consistency check.

## Assumptions (what the model *needs*)

A standard version assumes:

- Trading is frictionless (no transaction costs, infinite divisibility, continuous trading).
- A constant risk-free rate \(r\) (continuous compounding).
- The underlying follows GBM with constant volatility \(\sigma\):
  \[
  dS_t = \mu S_t\,dt + \sigma S_t\,dW_t^{\mathbb{P}}.
  \]
- No arbitrage, and enough traded instruments to replicate the claim (market completeness).
- No dividends (add a dividend yield \(q\) by replacing \(r\) with \(r-q\) in the stock drift under \(\mathbb{Q}\)).

## Derivation 1: Replication \(\Rightarrow\) PDE

Let \(V(t,S)\) be the option value. Apply Itô’s lemma to \(V(t,S_t)\):
\[
dV
= V_t\,dt + V_S\,dS + \tfrac12 V_{SS}\,(dS)^2
= \left(V_t + \mu S V_S + \tfrac12\sigma^2 S^2 V_{SS}\right)dt + \sigma S V_S\,dW.
\]

Form a self-financing hedged portfolio
\[
\Pi = V - \Delta S.
\]
Choose \(\Delta = V_S\) (delta hedging) so the \(dW\) term cancels:
\[
d\Pi = \left(V_t + \tfrac12\sigma^2 S^2 V_{SS}\right)dt.
\]

The portfolio is locally riskless, so in an arbitrage-free market it must earn the risk-free rate:
\[
d\Pi = r\Pi\,dt = r(V - S V_S)dt.
\]

Equating the two expressions yields the **Black–Scholes PDE**:
\[
\boxed{
V_t + \tfrac12\sigma^2 S^2 V_{SS} + r S V_S - rV = 0.
}
\]

### Terminal conditions

For a call with strike \(K\) and maturity \(T\),
\[
V(T,S) = (S-K)^+.
\]
For a put,
\[
V(T,S) = (K-S)^+.
\]

Solving the PDE with these terminal conditions gives the closed-form formulas below.

## Derivation 2: Risk-neutral expectation \(\Rightarrow\) same answer

From [risk-neutral pricing](risk_neutral_pricing.md), under \(\mathbb{Q}\) (no dividends),
\[
dS_t = rS_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}}.
\]
GBM implies
\[
\ln S_T \sim \mathcal N\!\left(\ln S_0 + (r-\tfrac12\sigma^2)T,\;\sigma^2 T\right).
\]
So for a European payoff \(g(S_T)\),
\[
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[g(S_T)].
\]
Evaluating this expectation for \(g(S_T)=(S_T-K)^+\) yields the same closed form as the PDE approach.

## Closed-form prices

Let \(\tau=T-t\) and define
\[
d_1 = \frac{\ln(S/K) + (r + \tfrac12\sigma^2)\tau}{\sigma\sqrt{\tau}},
\qquad
d_2 = d_1 - \sigma\sqrt{\tau}.
\]
Let \(N(\cdot)\) be the standard normal CDF.

### European call

\[
\boxed{
C(t,S)= S\,N(d_1) - K e^{-r\tau} N(d_2).
}
\]

### European put

\[
\boxed{
P(t,S)= K e^{-r\tau} N(-d_2) - S\,N(-d_1).
}
\]

## Interpreting the formula

The call price can be read as:

- \(S\,N(d_1)\): “present value of receiving the stock”, weighted by a hedge ratio; \(N(d_1)\) is the delta.
- \(K e^{-r\tau} N(d_2)\): present value of paying the strike, weighted by a risk-neutral exercise probability.

A useful identity: \(N(d_2)\) is the risk-neutral probability that \(S_T>K\) **under the \(T\)-forward measure**;
in the simplest constant-rate setting it is commonly interpreted as the risk-neutral in-the-money probability.

## Put–call parity

For European options on a non-dividend-paying stock,
\[
\boxed{
C - P = S - K e^{-r\tau}.
}
\]
This is a powerful consistency check and is often used for data cleaning.

## Greeks (quick reference)

Under BSM (no dividends), the most common Greeks are:

- Delta: \(\Delta_C = N(d_1)\), \(\Delta_P = N(d_1)-1\).
- Gamma: \(\Gamma = \frac{\varphi(d_1)}{S\sigma\sqrt{\tau}}\), where \(\varphi\) is the standard normal pdf.
- Vega: \(\nu = S\sqrt{\tau}\,\varphi(d_1)\).
- Theta and rho follow by differentiating the closed form.

## Practical implementation notes

- Use **log-moneyness** \(\ln(S/K)\) and \(\tau\) to avoid loss of precision.
- For very small \(\tau\) or extreme strikes, clamp \(\sigma\sqrt{\tau}\) away from 0 when computing \(d_1,d_2\).
- To back out \(\sigma\) from a market price, see [Implied volatility](IV.md).

## References

- Shreve, *Stochastic Calculus for Finance II*, Black–Scholes chapters.
- Hull, *Options, Futures, and Other Derivatives*, chapters on the BS formula and Greeks.
- Black & Scholes (1973); Merton (1973).
