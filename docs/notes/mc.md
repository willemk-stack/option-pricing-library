**Objectives**
- Shortly reflect on MC theory and risk-neutral pricing theory
- Implement a MC pricer and use it for european calls
- Compare to BS and demonstrate convergence and error
- Apply variance reduction techniques

In quantitative finance, \textbf{Monte Carlo (MC)} methods provide a way to price derivatives by approximating payoff expectations taken under a risk-neutral measure. The central idea is that, under standard no-arbitrage assumptions, the discounted value of an option equals the  expectation of its payoff.

### Risk-neutral pricing

Let $V_t$ denote the value at time $t$ of a claim that pays $g(\text{path})$ at maturity $T$, where the payoff may depend on the entire price path (e.g.\ Asian, barrier, lookback options) or only on the terminal value (e.g.\ European options). Then there exists an equivalent  risk-neutral measure (i.e they agree when probabilities are zero) $\mathbb{Q}$ such that

$$
V_t \;=\; \mathbb{E}^{\mathbb{Q}}\!\left[ D(t,T)\, g(\text{path}) \,\middle|\, \mathcal{F}_t \right],
$$

where $D(t,T)$ is the discount factor from $T$ back to $t$. If the instantaneous risk-free interest rate is constant and equal to $r$, then

$$
D(t,T) \;=\; e^{-r(T-t)}.
$$

In particular, at $t=0$,

$$
V_0 \;=\; \mathbb{E}^{\mathbb{Q}}\!\left[ e^{-rT}\, g(\text{path}) \right].
$$

**Example: European call under Black--Scholes**

We will mostly focus on a \textbf{European call} with strike $K$ and maturity $T$, with payoff

$$
g(S_T) \;=\; (S_T-K)^+.
$$

Under the Black--Scholes assumptions (see `03_black_scholes_pricing.ipynb`), the stock price under the risk-neutral measure $\mathbb{Q}$ follows

$$
dS_t \;=\; r S_t\,dt \;+\; \sigma S_t\, dW_t^{\mathbb{Q}}.
$$

Risk-neutral valuation then gives the time-$0$ call price

$$
C_0 \;=\; \mathbb{E}^{\mathbb{Q}}\!\left[e^{-rT}(S_T-K)^+\right].
$$

In this setting, $C_0$ is also available in closed form via the Black--Scholes formula, which we will use as a benchmark for our Monte Carlo estimates.

### Monte Carlo estimator
Define the discounted payoff random variable

$$
X \;:=\; e^{-rT}(S_T-K)^+,
$$

so that the call price can be written as the risk-neutral expectation

$$
C_0 \;=\; \mathbb{E}^{\mathbb{Q}}[X].
$$

By definition, an expectation is the average value of $X$ under $\mathbb{Q}$. Monte Carlo approximates this average by an empirical mean of $N$ i.i.d.\ samples $X^{(1)},\dots,X^{(N)}$:

$$
\widehat{C}_N \;:=\; \frac{1}{N}\sum_{i=1}^N X^{(i)} \;\approx\; \mathbb{E}^{\mathbb{Q}}[X] \;=\; C_0.
$$

Substituting $X^{(i)} = e^{-rT}(S_T^{(i)}-K)^+$ gives

$$
\widehat{C}_N \;=\; e^{-rT}\,\frac{1}{N}\sum_{i=1}^N (S_T^{(i)}-K)^+.
$$

In the Black--Scholes model, the terminal price admits the exact representation

$$
S_T \;=\; S_0 \exp\!\left(\left(r-\tfrac12\sigma^2\right)T + \sigma\sqrt{T}\,Z\right),
\qquad Z \sim \mathcal{N}(0,1).
$$

Thus, the MC workflow for this example is:
  - draw $Z^{(i)} \sim \mathcal{N}(0,1)$ for $i=1,\dots,N$,
  - compute $S_T^{(i)} = S_0 \exp\!\left(\left(r-\tfrac12\sigma^2\right)T + \sigma\sqrt{T}\,Z^{(i)}\right)$,
  - compute payoffs $\left(S_T^{(i)}-K\right)^+$,
  - discount and average to obtain $\widehat{C}_N$.

Finally, Monte Carlo also provides a natural measure of statistical uncertainty. With sample variance

$$
s^2 \;=\; \frac{1}{N-1}\sum_{i=1}^N \left(X^{(i)} - \widehat{C}_N\right)^2,
$$

the standard error is approximately

$$
\mathrm{SE}(\widehat{C}_N) \;\approx\; \frac{s}{\sqrt{N}}.
$$


(We will discuss convergence, error scaling, and confidence intervals in more detail in a dedicated section.)

**Note**
Implementation details
“We use pseudo-random numbers (NumPy) and standard normals to simulate GBM increments, as in Glasserman 2.2–2.3.”
