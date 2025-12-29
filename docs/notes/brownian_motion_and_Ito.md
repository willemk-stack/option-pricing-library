# Brownian motion and Itô calculus

Brownian motion is the canonical continuous-time “randomness” used in finance. Itô calculus is the extension of ordinary
calculus that lets us differentiate and integrate functions of Brownian-driven processes.

## Learning goals

- Know the defining properties of Brownian motion and what they imply.
- Simulate sample paths and verify mean/variance numerically.
- Understand why increments are independent (and why that matters for modelling).
- Apply Itô’s lemma in a simple example and see where the extra \(\tfrac12 f_{xx}\,dt\) term comes from.

## 1. Definition and intuition

### Formal definition

A process \((W_t)_{t\ge 0}\) is a **(standard) Brownian motion** if:

1. \(W_0=0\).
2. \(W_t\) has **continuous** sample paths.
3. For \(0\le s<t\), the increment \(W_t-W_s\) is normally distributed:
   \[
   W_t-W_s \sim \mathcal N(0,\,t-s).
   \]
4. Increments over disjoint intervals are **independent** (and the distribution depends only on the length, so increments are
   stationary).

Immediate consequences:

- \(\mathbb{E}[W_t]=0\), \(\operatorname{Var}(W_t)=t\).
- \(\operatorname{Cov}(W_s,W_t)=\min(s,t)\).
- Scaling: \((W_{ct})_{t\ge 0}\) has the same law as \((\sqrt{c}\,W_t)_{t\ge 0}\).

### Intuition

You can think of Brownian motion as the continuous-time limit of a random walk:
many tiny independent shocks accumulate into a path that is continuous but extremely jagged
(almost surely nowhere differentiable).

### Why finance cares

In the Black–Scholes–Merton model (and many extensions), the stock price is driven by Brownian motion:
\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t.
\]
This captures the idea of many small, approximately independent sources of uncertainty aggregated over time.

## 2. Simulating sample paths

To simulate \(W_t\) on \([0,T]\), choose a grid \(0=t_0<t_1<\dots<t_n=T\) with \(\Delta t_k=t_k-t_{k-1}\).
Generate i.i.d. increments
\[
\Delta W_k := W_{t_k}-W_{t_{k-1}} \sim \mathcal N(0,\,\Delta t_k),
\]
and set
\[
W_{t_k} = \sum_{j=1}^k \Delta W_j.
\]

**Pseudo-code**

```text
W = 0
for k = 1..n:
    dW ~ Normal(0, sqrt(dt))
    W = W + dW
    store W
```

This is also the building block for simulating GBM paths and Monte Carlo option pricing.

## 3. Mean and variance over time

From the definition,
\[
\mathbb{E}[W_t]=0,\qquad \operatorname{Var}(W_t)=t.
\]

A useful check when you simulate \(M\) independent paths is:

- the sample mean of \(W_t\) should be close to 0,
- the sample variance of \(W_t\) should be close to \(t\),
- and the standard error of the sample mean scales like \(1/\sqrt{M}\).

## 4. Independence of increments (and the Markov property)

Independence means that, conditional on the present, the future increment is “fresh noise”:
\[
(W_t-W_s) \perp\!\!\!\perp\; \mathcal F_s.
\]
In particular, Brownian motion is a **Markov process**: given \(W_s\), the future depends on the past only through \(W_s\).

For modelling, this is the simplest way to encode “no memory” at the noise level. Many models introduce memory by replacing
Brownian motion with something richer (e.g., stochastic volatility, rough volatility), but Brownian motion is the baseline.

## 5. Itô calculus: a first example

### The Itô multiplication rules

For Brownian motion increments we treat differentials according to:
\[
(dW_t)^2 = dt,\qquad dW_t\,dt = dt\,dW_t = (dt)^2 = 0.
\]
This encodes Brownian motion’s **quadratic variation**, which is the source of the extra Itô term.

### Itô’s lemma

If \(X_t\) follows
\[
dX_t = a(t,X_t)\,dt + b(t,X_t)\,dW_t,
\]
and \(f(t,x)\) is sufficiently smooth, then
\[
df(t,X_t)
= f_t\,dt + f_x\,dX_t + \tfrac12 f_{xx}\,(dX_t)^2
= \left(f_t + a f_x + \tfrac12 b^2 f_{xx}\right)dt + b f_x\,dW_t.
\]

### Example: \(f(W_t)=W_t^2\)

Take \(X_t=W_t\) so \(a=0\), \(b=1\), and \(f(x)=x^2\). Then \(f_x=2x\), \(f_{xx}=2\), and Itô’s lemma gives
\[
d(W_t^2) = 2W_t\,dW_t + dt.
\]
Taking expectations and using \(\mathbb{E}[\int_0^t W_s\,dW_s]=0\),
\[
\mathbb{E}[W_t^2] = t,
\]
matching \(\operatorname{Var}(W_t)=t\).

## Where to go next

- Brownian-driven stock dynamics are solved in [GBM](gbm.md).
- The Black–Scholes PDE and formula use Itô’s lemma implicitly; see [Black–Scholes pricing](bs_pricing.md).
