# Risk-neutral pricing

Derivative pricing in modern quantitative finance is primarily an **arbitrage** story, not a forecasting story.
The key idea is that (under standard assumptions) we can price without ever estimating the real-world drift \(\mu\).

## The martingale pricing principle

Fix a strictly positive traded asset as a **numéraire**. In the simplest single-currency setting we use the
money-market account
\[
B_t = \exp\!\left(\int_0^t r_u\,du\right),
\]
and with constant \(r\),
\[
B_t = e^{rt}.
\]

In an arbitrage-free market, there exists an **equivalent martingale measure** \(\mathbb{Q}\) such that every traded
asset price, when **discounted** by the numéraire, is a \(\mathbb{Q}\)-martingale. For a stock \(S_t\),
\[
\frac{S_t}{B_t} \text{ is a } \mathbb{Q}\text{-martingale.}
\]
Equivalently, for \(0\le t\le T\),
\[
\frac{S_t}{B_t}
= \mathbb{E}^{\mathbb{Q}}\!\left[\left.\frac{S_T}{B_T}\right|\mathcal{F}_t\right].
\]

## Pricing formula

Let a claim pay \(H_T\) at time \(T\). If the market is arbitrage-free (and complete in the classic Black–Scholes setup),
then its price process \(V_t\) satisfies
\[
V_t
= B_t\,\mathbb{E}^{\mathbb{Q}}\!\left[\left.\frac{H_T}{B_T}\right|\mathcal{F}_t\right]
= \mathbb{E}^{\mathbb{Q}}\!\left[\left.D(t,T)\,H_T\right|\mathcal{F}_t\right],
\]
where \(D(t,T)=\frac{B_t}{B_T}=\exp\big(-\int_t^T r_u\,du\big)\) is the stochastic discount factor.

That is the “one-line” recipe behind most models: **simulate / compute under \(\mathbb{Q}\)** and discount.

## Change of measure and the Radon–Nikodym derivative

A change of measure from \(\mathbb{P}\) to \(\mathbb{Q}\) is described by a Radon–Nikodym derivative (density) process
\((Z_t)_{t\ge 0}\) such that
\[
\left.\frac{d\mathbb{Q}}{d\mathbb{P}}\right|_{\mathcal{F}_t} = Z_t,\qquad Z_t>0,\qquad \mathbb{E}^{\mathbb{P}}[Z_t]=1.
\]
Intuitively: \(Z_t\) reweights paths so that the discounted traded assets lose their drift and become martingales.

## From \(\mu\) to \(r\) in Black–Scholes (Girsanov in one page)

Under the real-world measure \(\mathbb{P}\), the Black–Scholes model assumes
\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t^{\mathbb{P}},\qquad S_0>0,
\]
with constant \(\sigma>0\).

Define the (constant) **market price of risk**
\[
\lambda := \frac{\mu-r}{\sigma}.
\]
Then the exponential martingale
\[
Z_t = \exp\!\left(-\lambda W_t^{\mathbb{P}} - \tfrac{1}{2}\lambda^2 t\right)
\]
defines an equivalent measure \(\mathbb{Q}\) via \(d\mathbb{Q}=Z_T\,d\mathbb{P}\) (for each fixed horizon \(T\)).
Under \(\mathbb{Q}\), Girsanov’s theorem implies
\[
dW_t^{\mathbb{Q}} = dW_t^{\mathbb{P}} + \lambda\,dt
\]
is a Brownian motion, and substituting into the SDE yields
\[
dS_t = r S_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}}.
\]
So the drift becomes \(r\) under the pricing measure.

## Practical notes

- **Dividends:** with a continuous dividend yield \(q\), the risk-neutral drift is \(r-q\).
- **Different numeraires:** changing numéraire (e.g., to a forward measure) changes \(\mathbb{Q}\) but keeps prices invariant.
- **Curves:** in real markets you typically use the appropriate discount curve for \(D(t,T)\); the martingale idea remains the same.

## Where to go next

- The change-of-measure step uses Brownian motion; see [Brownian motion and Itô](brownian_motion_and_Ito.md).
- The resulting stock dynamics under \(\mathbb{Q}\) are solved explicitly in [GBM](gbm.md).
