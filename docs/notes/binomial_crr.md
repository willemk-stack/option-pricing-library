# Binomial CRR model

The Cox–Ross–Rubinstein (CRR) binomial tree is the classic discrete-time option-pricing model.
It is valuable because it makes **replication and risk-neutral probabilities** completely explicit,
and because it converges to Black–Scholes as the time step shrinks.

## Objectives

- State the assumptions that make the one-period binomial market arbitrage-free and complete.
- Derive the CRR risk-neutral probability and the pricing recursion.
- See how the tree converges to Black–Scholes for European options.
- Understand why binomial trees are convenient for early-exercise features (American options).

## 1. One-period market recap

Over one time step \(\Delta t\), assume:

- A risk-free asset grows by \(R = e^{r\Delta t}\).
- The stock moves from \(S\) to \(Su\) (up) or \(Sd\) (down), with \(u>d>0\).

### No-arbitrage condition

To avoid arbitrage you need

\[
d < R < u.
\]

If \(R\ge u\), borrow at \(r\) and buy the stock (dominates the bond). If \(R\le d\), short the stock and invest in the bond.

## 2. Replication and risk-neutral probability

Let a derivative pay \(V_u\) in the up state and \(V_d\) in the down state.
Hold \(\Delta\) shares and \(B\) in the bond so that

\[
\Delta Su + BR = V_u,\qquad \Delta Sd + BR = V_d.
\]

Solving gives

\[
\Delta = \frac{V_u - V_d}{S(u-d)},
\qquad
B = \frac{uV_d - dV_u}{R(u-d)}.
\]

The time-0 price is \(V_0=\Delta S + B\), which can be rearranged into a risk-neutral expectation:

\[
\boxed{
V_0 = \frac{1}{R}\left(p^* V_u + (1-p^*)V_d\right),
}
\qquad
p^* := \frac{R-d}{u-d}.
\]

Here \(p^*\in(0,1)\) precisely when \(d<R<u\). This is the **CRR risk-neutral probability**.

## 3. Multi-period tree and backward induction

With \(n\) steps, you build the tree of possible stock prices and compute option values backwards:

1. Set terminal values \(V_{n,j} = g(S_{n,j})\) at maturity.
2. For \(k=n-1,\dots,0\),

\[
V_{k,j} = \frac{1}{R}\left(p^* V_{k+1,j+1} + (1-p^*)V_{k+1,j}\right).
\]

For **American** options, replace step 2 with

\[
V_{k,j} = \max\left\{ g(S_{k,j}),\; \frac{1}{R}\left(p^* V_{k+1,j+1} + (1-p^*)V_{k+1,j}\right)\right\},
\]

which is where trees shine: early exercise is handled naturally by the max operator.

## 4. CRR parameter choice and link to Black–Scholes

CRR chooses

\[
u = e^{\sigma\sqrt{\Delta t}},\qquad d = e^{-\sigma\sqrt{\Delta t}},
\]

so the tree variance matches \(\sigma^2\Delta t\) to first order.
Then

\[
p^* = \frac{e^{r\Delta t}-e^{-\sigma\sqrt{\Delta t}}}{e^{\sigma\sqrt{\Delta t}}-e^{-\sigma\sqrt{\Delta t}}}.
\]

As \(\Delta t\to 0\) (i.e., \(n\to\infty\) for fixed \(T\)), the binomial price of a European call converges to
the Black–Scholes price. Intuitively:

- the log-stock performs a random walk with step size \(\sigma\sqrt{\Delta t}\),
- the central limit theorem turns that walk into Brownian motion,
- and the risk-neutral drift matches the continuous-time \(r\) drift.

## References

- Cox, Ross & Rubinstein (1979), “Option Pricing: A Simplified Approach”.
- Shreve, *Stochastic Calculus for Finance I*, binomial-tree chapters.
- Hull, *Options, Futures, and Other Derivatives*, chapters on binomial trees.
