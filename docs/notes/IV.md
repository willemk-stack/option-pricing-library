## Conceptual check questions
- Why do desks quote implied vol instead of price?
    - Scaled parameter, which holds relevance when discussing different Strike prices and experiation dates as opposed to the just the price.
- What does the smile/skew encode (distributional asymmetry, tail risk, demand/supply), and what should you not over-interpret?
    - Smile/skew meaning: For a fixed maturity, the IV “smile” is the market’s way of quoting the shape of option prices across strikes. It reflects the risk-neutral distribution’s asymmetry and tail heaviness: higher IV in the left wing (OTM puts) typically means downside tail risk/crash protection is priced rich; curvature reflects fat tails. The smile also reflects risk premia and supply/demand (insurance demand, dealer hedging flows), so it’s not a pure forecast of realized vol.
    - Don’t over-interpret: Don’t read IV as expected realized vol, and don’t treat the implied distribution as the physical probability distribution.

## Initial Guess Notes (Implied Volatility, Black–Scholes)

A common issue when using Newton–Raphson for implied volatility is choosing an initial value
\(\sigma_0\) in a region where **vega** is very small. Since Newton updates have the form

\[
\sigma_{n+1}=\sigma_n-\frac{C(\sigma_n)-C_{\text{mkt}}}{\text{vega}(\sigma_n)},
\]

small vega makes the iteration ill-conditioned: the update can become excessively large
(overshoot) or dominated by numerical noise.

To mitigate this, we use two approximations:

- an **ATM-forward** approximation based on **time value** (good near \(F\approx K\)),
- a **wing-scaled** approximation based on **log-moneyness** (robust ITM/OTM).

We then **blend** these seeds smoothly as a function of distance-to-ATM in log-moneyness, and
**clamp** the result into a safe volatility domain before passing it to a bracketed solver.

---

### Intrinsic Value and Time Value (Present-Value Form)

Under continuous dividend yield \(q\) and interest rate \(r\), the Black–Scholes European call
price satisfies the no-arbitrage bounds

\[
\max\!\left(S e^{-q\tau}-K e^{-r\tau},\,0\right)\ \le\ C_{BS}\ \le\ S e^{-q\tau},
\]

where \(\tau\) is time to expiry.

The lower bound \(\max(S e^{-q\tau}-K e^{-r\tau},0)\) is the **present value** of exercising the
call under forward carry: \(S e^{-q\tau}\) is the **prepaid forward** (PV of one share delivered
at expiry under continuous yield \(q\)), and \(K e^{-r\tau}\) is the PV of paying the strike at expiry.

Hence we can decompose

\[
C_{BS}
=
\underbrace{\max\!\left(S e^{-q\tau}-K e^{-r\tau},\,0\right)}_{\text{intrinsic value (PV)}}
+
\underbrace{\Big(C_{BS}-\max\!\left(S e^{-q\tau}-K e^{-r\tau},\,0\right)\Big)}_{\text{time value (PV)}}.
\]

Define the **present-value time value**

\[
\mathrm{TV}_{pv}
:=
C_{BS}-\max\!\left(S e^{-q\tau}-K e^{-r\tau},\,0\right).
\]

By no-arbitrage, \(\mathrm{TV}_{pv}\ge 0\). Importantly, the intrinsic term does **not** depend on
\(\sigma\); the dependence on \(\sigma\) is primarily carried by time value. This motivates using a
time-value-based approximation to construct a stable initial guess \(\sigma_0\).

> **Puts:** Analogous bounds/decomposition hold for puts. In practice, you can convert puts to calls
> via put–call parity (in forward/undiscounted units) and reuse the call formulas below.

---

### Rewriting in Forward / Undiscounted Terms

For algebraic convenience, rewrite the problem in forward units.

Let

\[
df := e^{-r\tau},\qquad F := S e^{(r-q)\tau}.
\]

Define the **undiscounted** call price

\[
c := \frac{C_{BS}}{df} = C_{BS}\,e^{r\tau}.
\]

In these units the intrinsic value is \(\max(F-K,0)\) and the **undiscounted time value** is

\[
\mathrm{TV}_u := c-\max(F-K,0).
\]

---

### ATM-Forward Seed (Time-Value Based)

At ATM-forward (\(F=K\)), \(\ln(F/K)=0\) and intrinsic is zero, so \(\mathrm{TV}_u=c\).
Define total volatility \(y:=\sigma\sqrt{\tau}\). In Black’s forward form,

\[
c = F N(d_1)-K N(d_2),\qquad
d_1=\frac{\ln(F/K)+\tfrac12 y^2}{y},\qquad d_2=d_1-y.
\]

Setting \(F=K\) gives \(d_1=y/2=-d_2\), hence

\[
c = F\big(N(y/2)-N(-y/2)\big)=F\big(2N(y/2)-1\big).
\]

Using the first-order expansion \(N(z)\approx \tfrac12+\varphi(0)z\) around \(z=0\), we obtain

\[
c \approx F\,\varphi(0)\,y = F\,\varphi(0)\,\sigma\sqrt{\tau},
\qquad \text{where}\quad \varphi(0)=\frac{1}{\sqrt{2\pi}}.
\]

Solving for \(\sigma\) and replacing \(c\) by \(\mathrm{TV}_u\) yields the **ATM seed**

\[
\sigma_{\text{ATM}} \approx \sqrt{\frac{2\pi}{\tau}}\ \frac{\mathrm{TV}_u}{F}.
\]

This seed is accurate near ATM-forward and tends to place the initial iterate in a region of
non-negligible vega.

---

### Wing / OTM Seed (Manaster–Koehler-Style Scaling)

The ATM seed can become unreliable in the wings because \(\mathrm{TV}_u\) becomes very small (and
vega collapses), even when the true implied volatility is not small.

To obtain a seed that remains well-scaled ITM/OTM, use a log-moneyness-based initializer commonly
associated with Manaster–Koehler. Let

\[
k := \ln(F/K).
\]

Define the wing seed

\[
\sigma_{\text{MK}} \approx \sqrt{\frac{2|k|}{\tau}}
= \sqrt{\frac{2|\ln(F/K)|}{\tau}}.
\]

This sets a total-volatility scale proportional to \(\sqrt{|k|}\), which helps avoid starting in
near-zero-vol regions when \(K\) is far from \(F\).

---

### Blending and Clamping

We blend the seeds smoothly as a function of \(|k|\):

\[
w(|k|)=\exp\!\left(-\frac{|k|}{k_0}\right),\qquad
\sigma_0 = w\,\sigma_{\text{ATM}}+(1-w)\,\sigma_{\text{MK}}.
\]

A typical choice is \(k_0\approx 0.10\), which corresponds roughly to the “near-ATM” region of about
\(\pm 10\%\) moneyness (since \(|\ln(1.10)|\approx 0.095\)).

Finally clamp into a safe domain:

\[
\sigma_0 \leftarrow \min(\sigma_{\max},\,\max(\sigma_{\min},\,\sigma_0)).
\]

---

### Practical Solver Notes

- If the market price is extremely close to the **lower no-arbitrage bound**, implied volatility is
  near 0 and vega is tiny; return \(\sigma\approx 0\) (or \(\sigma_{\min}\)) rather than forcing Newton.
- Use a **bracketed** method (Newton safeguarded by bisection/Brent):
  build an initial \([\sigma_{\min}, \sigma_{\text{hi}}]\) around \(\sigma_0\) and expand \(\sigma_{\text{hi}}\)
  until the root is bracketed.

---

## References

- Manaster, S. & Koehler, G. (1982). *The Calculation of Implied Variances from the Black–Scholes Model: A Note.*
- Brenner, M. & Subrahmanyam, M. (1988). *A Simple Formula to Compute the Implied Standard Deviation.*
- Tehranchi, M. (2015/2016). *Uniform bounds for Black–Scholes implied volatility* (discussion of the Manaster–Koehler landmark scaling in total-vol units).
- Corrado, C. & Miller, T. (1996). *A note on a simple, accurate formula to compute implied volatility* (extends ATM-style approximations away from ATM).
- Jäckel, P. (2015). *Let’s Be Rational* (robust inversion methods and initializations used in practice).
