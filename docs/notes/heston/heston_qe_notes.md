# Andersen's QE Heston simulation scheme

!!! note "Status"
    This note gives model background and implementation-oriented interpretation for the QE
    scheme used in the repository. It is meant to explain the method and its limitations,
    not to claim a full convergence proof beyond what the cited references support.

## Learning goals

- List implementation notes and failure notes.
- Know what is actually justified for the QE scheme: not a full weak convergence proof in the paper, but a **weak consistency** result under the condition that $\gamma_1 + \gamma_2 \to 1$ as $\Delta \to 0$.
- Understand why naive Euler is not a production scheme here, even if it is useful as a benchmark or sanity check.

## Core question

For an arbitrary increment $\Delta$, how do we generate a random sample of
$$
(\log X_{t+\Delta}, V_{t+\Delta}) \mid (\log X_t, V_t)
$$
with low discretization bias and acceptable computational cost?

That is the real simulation problem for Heston path generation.

## Model setup

The Heston model is
$$
\frac{dX_t}{X_t} = \sqrt{V_t}\, dW_t^X,
$$
$$
dV_t = \kappa(\theta - V_t)\, dt + \varepsilon \sqrt{V_t}\, dW_t^V,
$$
with
$$
dW_t^X\, dW_t^V = \rho\, dt.
$$

Equivalently, for the log-price,
$$
d \log X_t = -\frac12 V_t\, dt + \sqrt{V_t}\, dW_t^X.
$$

## Why this object matters

The whole point is to minimize **discretization bias** in Monte Carlo simulation of the Heston model.

Broadly:

- Monte Carlo error has two pieces:
  1. statistical noise,
  2. bias from approximating the continuous-time diffusion by a discrete-time scheme.

In Heston, the second piece can be large at practical step sizes if the discretization is too naive.

## Big idea behind Andersen's approach

The variance process $V$ is not well-approximated by a plain Gaussian near zero. Its true conditional law is non-central chi-square, so it is:

- nonnegative,
- skewed,
- often heavily concentrated near zero.

So the right strategy is:

1. use the **exact first two conditional moments** of $V_{t+\Delta} \mid V_t$,
2. choose a tractable sampling law that preserves nonnegativity,
3. switch forms depending on how hard the conditional distribution is to approximate.

The exact conditional moments used by QE are
$$
m = \mathbb{E}[V_{t+\Delta} \mid V_t]
  = \theta + (V_t - \theta)e^{-\kappa \Delta},
$$
and

$$
s^2 = \operatorname{Var}(V_{t+\Delta} \mid V_t)
= V_t \frac{\varepsilon^2 e^{-\kappa \Delta}}{\kappa}\left(1 - e^{-\kappa \Delta}\right)
+ \theta \frac{\varepsilon^2}{2\kappa}\left(1 - e^{-\kappa \Delta}\right)^2.
$$

Define the key shape ratio
$$
\psi = \frac{s^2}{m^2}.
$$

This $\psi$ is the switching statistic.

## Sampling variance with the QE scheme

### Quadratic branch

When $\psi$ is small enough, represent the next-step variance as
$$
\widehat V_{t+\Delta} = a(b + Z_V)^2,
\qquad Z_V \sim N(0,1).
$$

Moment matching gives
$$
b^2 = \frac{2}{\psi} - 1 + \sqrt{\frac{2}{\psi}}\sqrt{\frac{2}{\psi} - 1},
$$
and
$$
a = \frac{m}{1 + b^2}.
$$

This branch only works when
$$
\psi \le 2.
$$

### Exponential branch

When $\psi$ is larger, use the atom-at-zero plus exponential-tail approximation:
$$
\widehat V_{t+\Delta} =
\begin{cases}
0, & U_V \le p, \\
\frac{1}{\beta}\log\left(\frac{1-p}{1-U_V}\right), & U_V > p,
\end{cases}
\qquad U_V \sim \mathrm{Unif}(0,1).
$$

The parameters are
$$
p = \frac{\psi - 1}{\psi + 1},
$$
and
$$
\beta = \frac{1-p}{m} = \frac{2}{m(\psi + 1)}.
$$

This branch only works when
$$
\psi \ge 1.
$$

### Switching rule

Because the quadratic branch works for $\psi \le 2$ and the exponential branch works for $\psi \ge 1$, Andersen introduces a threshold
$$
\psi_c \in [1,2]
$$
and uses:

- quadratic branch if $\psi \le \psi_c$,
- exponential branch if $\psi > \psi_c$.

In the paper's numerical tests,
$$
\psi_c = 1.5.
$$

### Summary algorithm for sampling variance

1. Given $\widehat V_t$, compute $m$ and $s^2$.
2. Compute
   $$
   \psi = \frac{s^2}{m^2}.
   $$
3. Draw $U_V \sim \mathrm{Unif}(0,1)$.
4. If $\psi \le \psi_c$:
   - compute $a$ and $b$,
   - compute
     $$
     Z_V = \Phi^{-1}(U_V),
     $$
   - set
     $$
     \widehat V_{t+\Delta} = a(b + Z_V)^2.
     $$
5. If $\psi > \psi_c$:
   - compute $p$ and $\beta$,
   - set
     $$
     \widehat V_{t+\Delta} =
     \begin{cases}
     0, & U_V \le p, \\
     -\frac{1}{\beta}\log\left(\frac{1-p}{1-U_V}\right), & U_V > p.
     \end{cases}
     $$

## What assumption is doing the heavy lifting?

The heavy lifting is **not** just "approximate $V$ by a root function."

The important ingredients are:

1. exact matching of the first two conditional moments of $V_{t+\Delta}\mid V_t$,
2. choosing an approximation family that is compatible with nonnegativity,
3. switching between two approximation families using $\psi$,
4. building the $X$ step from the exact integral representation instead of a naive Euler correlation coupling.

For the asymptotic justification stated in the paper, the explicit assumption is
$$
\gamma_1 + \gamma_2 \to 1
\qquad \text{as } \Delta \to 0,
$$
which is used in the weak consistency proposition.

## Sampling log price and why Euler is too naive

The exact representation Andersen starts from is

$$
\log X_{t+\Delta}
=
\log X_t
+ \frac{\rho}{\varepsilon}\left(V_{t+\Delta} - V_t - \kappa\theta\Delta\right)
+ \left(\frac{\kappa\rho}{\varepsilon} - \frac12\right)\int_t^{t+\Delta} V_u\, du
+ \sqrt{1-\rho^2}\int_t^{t+\Delta} \sqrt{V_u}\, dW_u.
$$

He then approximates the integrated variance by

$$
\int_t^{t+\Delta} V_u\, du
\approx
\Delta\left(\gamma_1 V_t + \gamma_2 V_{t+\Delta}\right).
$$

This gives the practical update

$$
\log \widehat X_{t+\Delta}
=
\log \widehat X_t
+ K_0 + K_1 \widehat V_t + K_2 \widehat V_{t+\Delta}
+ \sqrt{K_3 \widehat V_t + K_4 \widehat V_{t+\Delta}}\, Z,
$$

where $Z \sim N(0,1)$ is **independent** of the random numbers used to generate $\widehat V_{t+\Delta}$.

The coefficients are
$$
K_0 = -\frac{\rho \kappa \theta}{\varepsilon}\Delta,
$$
$$
K_1 = \gamma_1 \Delta\left(\frac{\kappa\rho}{\varepsilon} - \frac12\right) - \frac{\rho}{\varepsilon},
$$
$$
K_2 = \gamma_2 \Delta\left(\frac{\kappa\rho}{\varepsilon} - \frac12\right) + \frac{\rho}{\varepsilon},
$$
$$
K_3 = \gamma_1 \Delta(1-\rho^2),
\qquad
K_4 = \gamma_2 \Delta(1-\rho^2).
$$

## Why naive Euler is naive

There are **two** separate naiveties.

### 1. Naive Euler for variance

A direct Euler step for variance can produce negative values:

$$
\widehat V_{t+\Delta}
=
\widehat V_t + \kappa(\theta - \widehat V_t)\Delta
+ \varepsilon \sqrt{\widehat V_t}\, Z_V \sqrt{\Delta}.
$$

This is already problematic because:

- variance should be nonnegative,
- once it goes negative, $\sqrt{\widehat V}$ is not even defined for the next step,
- truncation repairs are heuristic,
- the scheme ignores the known conditional law structure of the CIR variance process.

So Euler here is not principled enough for production-quality Heston simulation.

### 2. Naive Euler for log price

Even if you repair the $V$ step, it is still tempting to correlate the Gaussian shock in the $\log X$ step directly with the Gaussian used to generate $V$.

That is too naive.

The reason is that once $\widehat V_{t+\Delta}$ is generated through a **nonlinear** transform such as
$$
a(b + Z_V)^2
$$
or through the exponential branch, the effective correlation between $\log \widehat X_{t+\Delta}$ and $\widehat V_{t+\Delta}$ is no longer correctly represented by simply correlating Gaussian shocks.

This produces **correlation leakage**:

- effective correlation becomes too weak,
- simulated tails of $X$ are wrong,
- away-from-the-money option pricing deteriorates.

That is why Andersen abandons naive Euler discretization for $\log X$ and keeps the explicit
$$
\frac{\rho}{\varepsilon}\left(V_{t+\Delta} - V_t - \kappa\theta\Delta\right)
$$
term from the exact representation.

## Martingale correction

The raw $X$ scheme above does **not** automatically satisfy the discrete-time martingale condition
$$
\mathbb{E}[\widehat X_{t+\Delta} \mid \widehat X_t] = \widehat X_t.
$$

So Andersen introduces a corrected constant $K_0^*$.

Let
$$
A = K_2 + \frac12 K_4
= \frac{\rho}{\varepsilon}(1 + \kappa \gamma_2 \Delta) - \frac12 \gamma_2 \Delta \rho^2,
$$
and define
$$
M = \mathbb{E}\left[e^{A \widehat V_{t+\Delta}} \mid \widehat V_t\right].
$$

Then, if $M < \infty$, set
$$
K_0^* = -\log M - \left(K_1 + \frac12 K_3\right)\widehat V_t.
$$

Replacing $K_0$ by $K_0^*$ enforces
$$
\mathbb{E}[\widehat X_{t+\Delta} \mid \widehat X_t] = \widehat X_t.
$$

This is the martingale-corrected version of the scheme.

## Positive correlation caveat

This is an important failure note.

For $\rho \le 0$, the relevant exponential moment is safe.

For $\rho > 0$, regularity conditions must be checked.

In the QE quadratic branch, one needs
$$
A < \frac{1}{2a}.
$$

In the QE exponential branch, one needs
$$
A < \beta.
$$

Andersen notes that, roughly, positive correlation imposes a step-size restriction of the form
$$
\rho \varepsilon \Delta < 2.
$$

So the issue is **not** "positive correlation means QE is wrong."
The real issue is:

- for $\rho > 0$, do not ignore the exponential-moment regularity conditions,
- do not apply the martingale correction blindly.

## What does the paper actually guarantee?

Do **not** overclaim this.

The paper does **not** present a full general weak convergence proof for QE.

Instead, it proves **weak consistency**.

Under the assumption
$$
\gamma_1 + \gamma_2 \to 1
\qquad \text{as } \Delta \to 0,
$$
the schemes are weakly consistent in the sense that

$$
\lim_{\Delta \to 0}
\mathbb{E}\left[
\frac{\log \widehat X_{t+\Delta} - \log \widehat X_t}{\Delta}
\;\middle|\;
\widehat X_t,\widehat V_t
\right]
=
-\frac12 \widehat V_t,
$$

$$
\lim_{\Delta \to 0}
\operatorname{Var}\left[
\frac{\log \widehat X_{t+\Delta} - \log \widehat X_t}{\sqrt{\Delta}}
\;\middle|\;
\widehat X_t,\widehat V_t
\right]
=
\widehat V_t,
$$

$$
\lim_{\Delta \to 0}
\mathbb{E}\left[
\frac{\widehat V_{t+\Delta} - \widehat V_t}{\Delta}
\;\middle|\;
\widehat X_t,\widehat V_t
\right]
=
\kappa(\theta - \widehat V_t),
$$

$$
\lim_{\Delta \to 0}
\operatorname{Var}\left[
\frac{\widehat V_{t+\Delta} - \widehat V_t}{\sqrt{\Delta}}
\;\middle|\;
\widehat X_t,\widehat V_t
\right]
=
\varepsilon^2 \widehat V_t,
$$

and

$$
\lim_{\Delta \to 0}
\operatorname{Cov}\left(
\frac{\widehat V_{t+\Delta} - \widehat V_t}{\sqrt{\Delta}},
\frac{\log \widehat X_{t+\Delta} - \log \widehat X_t}{\sqrt{\Delta}}
\;\middle|\;
\widehat X_t,\widehat V_t
\right)
=
\rho \varepsilon \widehat V_t.
$$

So the right statement is:

> QE is justified in the paper by a weak consistency result, not by a full-blown convergence theorem.

## Implementation notes

- Precompute anything depending only on $\Delta$ outside the Monte Carlo loop, especially
  $$
  e^{-\kappa\Delta}.
  $$
- At each step, compute
  $$
  m, \quad s^2, \quad \psi = \frac{s^2}{m^2}.
  $$
- Switch on $\psi$, not directly on whether $V_t$ is "small."
- Use a fixed threshold such as
  $$
  \psi_c = 1.5.
  $$
- Generate $\widehat V_{t+\Delta}$ first.
- Then draw a **new independent** Gaussian for the $\log X$ step.
- If using the martingale-corrected form, check the regularity conditions for $\rho > 0$.
- Use Broadie--Kaya as an exact-simulation reference in literature discussions,
  not as this repository's default simulation workhorse.
- Treat Euler as a benchmark and debugging baseline, not as the final
  validation discretization.

## Failure notes

### Numerical failure notes

- Negative variance under naive Euler.
- Using a naive correlated-Gaussian $X$ update causes correlation leakage.
- Ignoring the $\psi$ switching rule degrades performance.
- For $\rho > 0$, ignoring regularity conditions can invalidate the martingale correction.
- Using too large a time step can break the intended accuracy advantages of QE.

### Conceptual failure notes

- Saying QE is "just a quadratic transform" is incomplete; it is a **switching** moment-matched scheme.
- Saying the paper proves weak convergence is too strong; the paper proves weak consistency.
- Thinking Euler is "good enough because it is simple" misses the structural near-zero behavior of the CIR variance process.
- Treating Broadie--Kaya as practical production default misses its complexity and bump-and-reprice inconveniences.
- Treating Euler as anything more than a reference scheme in serious Heston MC is usually a mistake.

## One-line mental summary

QE works because it respects the shape of the Heston variance transition much better than Euler, and the Andersen $X$ step preserves the correct variance-price coupling by building from the exact representation rather than from naive correlated Gaussian shocks.

## References

The note relies on the local `Finance-books` source library:

- *AndersenHestonSimulation.pdf*
    in `02_Pricing_Models/01_Classic_Models/Heston`.
- *broadie_kaya_exact_sim_or_2006.pdf*
    in `04_Numerical_Methods/02_Monte_Carlo`.

