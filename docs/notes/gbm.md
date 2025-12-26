## From SDE to GBM using Itô
In order to find a solution, let us consider $X_t = \ln S_t$. Now applying Itô's formule to $dX_t$:

$$
dX_t = \frac{1}{S_t}dS_t - \frac{1}{2}\frac{1}{S_t^2}dS_tdS_t (*)
$$

Computing $dS_tdS_t = (\mu S_tdt+\sigma S_tdW_t)^2$ using Itô's multiplication rules, we find

$$
dS_tdS_t = \sigma^2 S_t^2 dt
$$

Substituting this along with our formula for $dS_t$ in $(*)$

$$
dX_t = \mu dt + \sigma dW_t - \frac{1}{2}\sigma^2 dt
$$

$$
dX_t = (\mu - \frac{1}{2}\sigma^2)dt + \sigma dW_t
$$

Integrating from 0 to t:

$$
X_t = X_0 + (\mu - \frac{1}{2}\sigma^2)t + \sigma W_t \rightarrow \ln S_t = \ln S_0 + (\mu - \frac{1}{2}\sigma^2)t + \sigma W_t
$$

Where taking the exponents gives us the final result:

$$
S_t = S_0 \exp((\mu - \frac{1}{2}\sigma^2)t + \sigma W_t)
$$

## Financial interpretation (returns vs log-returns)

In finance we often describe performance in terms of **returns**.

- The *simple* return over $[0,T]$ is

  $$
  R_T = \frac{S_T - S_0}{S_0}.
  $$

- The **log-return** (continuously compounded return) is

  $$
  x_T = \frac{1}{T}\ln\frac{S_T}{S_0}.
  $$

For small moves, $\ln(1 + R_T) \approx R_T$, so simple returns and
log-returns are numerically close. The key advantage of log-returns is
that they are **additive over time**: if $0 = t_0 < t_1 < \dots < t_n = T$,
then

$$
\ln\frac{S_T}{S_0}
  = \sum_{k=1}^n \ln\frac{S_{t_k}}{S_{t_{k-1}}}.
$$

The total log-return is just the sum of period log-returns.

Under geometric Brownian motion (GBM) we have

$$
\ln\frac{S_T}{S_0} \sim
\mathcal N\!\Big(\big(\mu - \tfrac{1}{2}\sigma^2\big)T,\; \sigma^2 T\Big),
$$

so our scaled log-return satisfies

$$
x_T = \frac{1}{T}\ln\frac{S_T}{S_0}
  \sim \mathcal N\!\left(\mu - \tfrac{1}{2}\sigma^2,\; \frac{\sigma^2}{T}\right).
$$

This is exactly what we see in the simulations of $x_1$ and $x_{20}$:
both distributions are centered near $\mu - \tfrac{1}{2}\sigma^2$, but
the variance shrinks like $\sigma^2/T$ as the horizon $T$ increases.
That variance scaling explains why long-horizon average returns look
more stable than short-horizon ones.

For readers who want the full derivation and the link to continuous
compounding, see **Appendix A** below.

## Appendix A: Continuous compounding and log-returns (derivation)

### From discrete compounding to continuous compounding

Suppose a stock earns a **constant** annual rate of return $r$,
compounded $n$ times per year. After $t$ years the price is

$$
S_t = S_0\left(1 + \frac{r}{n}\right)^{nt}.
$$

If we let the compounding frequency go to infinity ($n \to \infty$),
we obtain **continuous compounding**:

$$
S_t = S_0 e^{rt}.
$$

This motivates the definition of the **continuously compounded rate of
return** $x$ over $[0,t]$ as the (possibly random) number satisfying

$$
S_t = S_0 e^{x t}
\quad\Longleftrightarrow\quad
x = \frac{1}{t}\ln\frac{S_t}{S_0}.
$$

So $x$ is the constant continuously compounded rate that would produce
the same growth as the actual (possibly random) price process.

### Simple returns vs log-returns

The *simple* return over $[0,t]$ is

$$
R_t = \frac{S_t - S_0}{S_0}.
$$

Relating this to $x$, we have

$$
1 + R_t = \frac{S_t}{S_0} = e^{x t}
\quad\Longrightarrow\quad
R_t = e^{x t} - 1.
$$

Conversely,

$$
x = \frac{1}{t}\ln(1 + R_t).
$$

For **small** simple returns $R_t$, we can use the Taylor approximation
$\ln(1 + R_t) \approx R_t$, which gives

$$
x \approx \frac{R_t}{t}.
$$

This explains why, over short horizons and when price moves are small,
simple returns and log-returns are numerically very close.

### Time additivity of log-returns

A crucial property of log-returns is that they are **additive over time**.
Let $0 = t_0 < t_1 < \dots < t_n = T$. Then

$$
\frac{S_T}{S_0}
  = \frac{S_T}{S_{t_{n-1}}}
    \cdot \frac{S_{t_{n-1}}}{S_{t_{n-2}}}
    \cdots
    \frac{S_{t_1}}{S_0}.
$$

Taking logs,

$$
\ln\frac{S_T}{S_0}
  = \sum_{k=1}^n \ln\frac{S_{t_k}}{S_{t_{k-1}}}.
$$

If we define the log-return over $[t_{k-1}, t_k]$ by

$$
x_{t_k, t_{k-1}}
  = \frac{1}{t_k - t_{k-1}}
    \ln\frac{S_{t_k}}{S_{t_{k-1}}},
$$

then the **total** log-return over $[0,T]$ is a time-weighted average of
the period log-returns:

$$
\frac{1}{T}\ln\frac{S_T}{S_0}
  = \sum_{k=1}^n
      \frac{t_k - t_{k-1}}{T}\,
      x_{t_k, t_{k-1}}.
$$

This clean additivity is one of the main reasons log-returns are preferred
in models and in many statistical procedures.

### Distribution of log-returns under GBM

Under geometric Brownian motion (GBM),

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,
$$

the solution is

$$
S_t = S_0 \exp\left(
    \left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t
\right).
$$

Taking logs and subtracting $\ln S_0$,

$$
\ln\frac{S_t}{S_0}
  = \left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t.
$$

Since $W_t \sim \mathcal N(0, t)$, this implies

$$
\ln\frac{S_t}{S_0}
  \sim \mathcal N\!\Big(
      \left(\mu - \tfrac{1}{2}\sigma^2\right)t,\;
      \sigma^2 t
  \Big).
$$

For the **scaled log-return**

$$
x_t = \frac{1}{t}\ln\frac{S_t}{S_0},
$$

we therefore obtain

$$
x_t \sim \mathcal N\!\left(
    \mu - \tfrac{1}{2}\sigma^2,\;
    \frac{\sigma^2}{t}
\right).
$$

So:

- The **mean** of $x_t$ is $\mu - \tfrac{1}{2}\sigma^2$, independent of $t$.
- The **variance** of $x_t$ is $\sigma^2 / t$, which **decreases** as the
  horizon $t$ increases.

In the notebook we verify this numerically by simulating many GBM paths,
computing $x_1$ and $x_{20}$, and plotting their histograms. Both
distributions are centered near the same mean
$\mu - \tfrac{1}{2}\sigma^2$, but the spread of $x_{20}$ is much
smaller than that of $x_1$, illustrating the variance scaling

$$
\operatorname{Var}(x_T) = \frac{\sigma^2}{T}.
$$

### Takeaways

- Log-returns correspond to **continuous compounding** and are naturally
  linked to the exponential solution of GBM.
- They are **additive over time**, which makes them very convenient for
  multi-period modelling.
- Under GBM, the scaled log-return $x_T$ is normal with variance
  $\sigma^2/T$, so average returns become more stable over longer
  horizons.
- These properties explain why log-returns are the standard choice in
  quantitative finance and in models such as Black–Scholes.
