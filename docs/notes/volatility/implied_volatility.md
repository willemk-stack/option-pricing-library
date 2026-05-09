# Implied volatility

!!! note "Status: mathematical background and implementation context"
    This note explains implied volatility as a price inversion. Solver-specific
    guarantees belong to the API docs and focused tests linked from the
    validation matrix.

Market prices are often quoted not as option prices but as **implied volatilities**.
The implied volatility (IV) of an option is the value of \(\sigma\) that, plugged into the Black–Scholes formula,
reproduces the observed market price.

This note focuses on the “practical inversion” problem: given a market price, find \(\sigma\) robustly.

## Why desks quote vol instead of price

- **Comparability:** prices depend strongly on spot, strike, maturity, rates; vol is a more comparable “normalized” number.
- **Risk language:** vega, variance swaps, and hedging discussions are naturally in vol/variance units.
- **Surface building:** traders reason in terms of an implied-vol surface \(\sigma(K,T)\).

## Definition and existence

Let \(C_{\text{BS}}(S,K,r,\tau,\sigma)\) be the Black–Scholes call price (similarly for puts).
Given a market price \(C_{\text{mkt}}\), the implied volatility \(\sigma_{\text{imp}}\) solves

\[
C_{\text{BS}}(\sigma_{\text{imp}}) = C_{\text{mkt}}.
\]

For fixed \((S,K,r,\tau)\), the Black–Scholes call price is:

- **increasing** in \(\sigma\),
- continuous,
- and satisfies simple no-arbitrage bounds.

Therefore, if \(C_{\text{mkt}}\) lies within the no-arbitrage bounds, there is a **unique** solution.

## No-arbitrage bounds (call, no dividends)

Let \(\tau=T-t\).

- Lower bound: \(C \ge \max\{S - K e^{-r\tau},\,0\}\).
- Upper bound: \(C \le S\).

If \(C\) is extremely close to the lower bound, IV is near 0 and numerical inversion becomes ill-conditioned (vega is tiny).

## Numerically solving for IV

The standard Newton–Raphson update is

\[
\sigma_{n+1} = \sigma_n - \frac{C(\sigma_n)-C_{\text{mkt}}}{\text{vega}(\sigma_n)}.
\]

Newton is fast when it works, but it can fail when:

- vega is small (deep ITM/OTM or very short maturity),
- the initial guess is poor,
- or the price is near the no-arbitrage bounds.

In production you typically use a **safeguarded** method:
Newton steps when they stay inside a bracket, otherwise fall back to bisection/Brent.

## Initial guess notes (practical seeds)

A robust solver starts with a good initial guess \(\sigma_0\). The ideas below are designed to avoid regions where vega is tiny.

### Intrinsic value and time value (present-value form)

Write a call price as

\[
C = \underbrace{\max(S - K e^{-r\tau},\,0)}_{\text{intrinsic value}}
  + \underbrace{\text{time value}}_{\ge 0}.
\]

Near the intrinsic bound, time value is small and IV is close to 0.

### Rewriting in forward / undiscounted terms

Let the forward price be \(F = S e^{r\tau}\) (no dividends). Define the undiscounted call

\[
\widetilde C := e^{r\tau} C.
\]

Then \(\widetilde C\) is a function of \(F,K,\tau,\sigma\) and is often numerically better behaved.

### ATM-forward seed (time-value based)

At-the-money-forward (\(K\approx F\)), the option time value is approximately linear in \(\sigma\sqrt{\tau}\).
A time-value approximation commonly used as a heuristic leads to a seed

\[
\sigma_{\text{ATM}} \approx \sqrt{\frac{2\pi}{\tau}}\,\frac{\widetilde C}{F}.
\]

Use this as a safe starting point near ATM.

### Wing / OTM seed (log-moneyness scaling)

In the wings, it is often more stable to work with total volatility \(\sigma\sqrt{\tau}\) and log-moneyness
\(k=\ln(K/F)\). A common heuristic seed is

\[
\sigma_{\text{wing}} \approx \frac{|k|}{\sqrt{\tau}}\,\left(\frac{1}{\sqrt{2\,|\ln(\widetilde C/\text{scale})|}}\right),
\]

where “scale” is chosen so the argument is dimensionless (practitioners use several variants).

The main point is not the exact constant, but the behavior: **far OTM implies larger \(|k|\)** and IV should not start near zero.

### Blending and clamping

Blend the seeds smoothly as a function of \(|k|\):

\[
w(|k|)=\exp\!\left(-\frac{|k|}{k_0}\right),\qquad
\sigma_0 = w\,\sigma_{\text{ATM}}+(1-w)\,\sigma_{\text{MK}}.
\]

A typical choice is \(k_0\approx 0.10\) (roughly \(\pm 10\%\) moneyness).

Finally clamp into a safe domain:

\[
\sigma_0 \leftarrow \min(\sigma_{\max},\,\max(\sigma_{\min},\,\sigma_0)).
\]

### Practical solver notes

- If \(C\) is extremely close to the **lower no-arbitrage bound**, return \(\sigma\approx 0\) (or \(\sigma_{\min}\))
  rather than forcing Newton to “discover” it.
- Always keep a bracket \([\sigma_{\min},\sigma_{\max}]\) and ensure \(C(\sigma)\) brackets the market price.
- Use forward/discounted variables to improve stability (especially for very long \(\tau\) or large \(r\)).

## Interpolation notes


## Conceptual check questions

- Why do desks quote implied vol instead of price?
- What does the smile/skew encode (distributional asymmetry, tail risk, demand/supply), and what should you not over-interpret?
  - **Smile/skew meaning:** for a fixed maturity, the IV smile reflects the market-implied distribution under \(\mathbb{Q}\),
    plus supply/demand effects (crash protection, flow, constraints).
  - **Don’t over-interpret:** IV is not a direct forecast of realized volatility, and the implied distribution is not the
    physical probability distribution.

## References

The note relies on the local `Finance-books` source library:

- *Options Futures Derivatives (2021).pdf*
    in `01_Foundations/01_Derivatives_Overview_(Hall)`.
- *Gatheral - The Volatility Surface.pdf*
    in `03_Volatility_Surface/01_Books_Notes`.
- *Euan Sinclair - Volatility Trading.pdf* in `06_Trading_and_ML`.
