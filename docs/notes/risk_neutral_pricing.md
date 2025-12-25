## Risk-neutral measure and why the drift becomes $r$

In quantitative finance, we often price derivatives by switching from the real-world probability measure $\mathbb{P}$ to an equivalent martingale (risk-neutral) measure $\mathbb{Q}$. The key idea is **not** that the stock price $S_t$ itself must be a martingale, but that **discounted traded asset prices** (with respect to a chosen numéraire) are martingales under $\mathbb{Q}$.

### Numéraire and discounted martingales

Let the money-market account be the numéraire
$$
B_t = \exp\left(\int_0^t r_u\,du\right),
$$
and in the simplest Black--Scholes setting with constant rate $r$, this reduces to
$$
B_t = e^{rt}.
$$

The defining property of the risk-neutral measure associated with $B_t$ is that the *discounted* stock price is a martingale:
$$
\frac{S_t}{B_t}\ \text{is a } \mathbb{Q}\text{-martingale.}
$$
Equivalently, for $0 \le t \le T$,
$$
\frac{S_t}{B_t} = \mathbb{E}^{\mathbb{Q}}\!\left[\left.\frac{S_T}{B_T}\right|\mathcal{F}_t\right].
$$

### Change of measure and the Radon--Nikodym derivative

A change of measure from $\mathbb{P}$ to $\mathbb{Q}$ is described by the Radon--Nikodym derivative *process*
$$
Z_t=\left.\frac{d\mathbb{Q}}{d\mathbb{P}}\right|_{\mathcal{F}_t},
$$
which is required to be a strictly positive $\mathbb{P}$-martingale with $\mathbb{E}^{\mathbb{P}}[Z_t]=1$. In practice, rather than working with $Z_t$ directly, we use **Girsanov's theorem** to express the measure change as a drift shift of Brownian motion.

### From $\mu$ to $r$ via Girsanov

Under the real-world measure $\mathbb{P}$, the Black--Scholes model assumes the stock follows a geometric Brownian motion:
$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t^{\mathbb{P}},\qquad S_0>0,
$$
where $\mu$ is the real-world drift, $\sigma>0$ is volatility, and $W^{\mathbb{P}}$ is a Brownian motion under $\mathbb{P}$.

Define the (constant) **market price of risk**
$$
\lambda := \frac{\mu - r}{\sigma}.
$$
Girsanov's theorem tells us that we can define a new Brownian motion under $\mathbb{Q}$ by
$$
dW_t^{\mathbb{Q}} = dW_t^{\mathbb{P}} + \lambda\,dt.
$$
Substituting $dW_t^{\mathbb{P}} = dW_t^{\mathbb{Q}} - \lambda\,dt$ into the $\mathbb{P}$-dynamics yields
$$
\begin{aligned}
dS_t
&= \mu S_t\,dt + \sigma S_t\left(dW_t^{\mathbb{Q}} - \lambda\,dt\right) \\
&= (\mu - \sigma\lambda)S_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}} \\
&= r S_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}}.
\end{aligned}
$$
This is the precise sense in which the drift "$\mu$ turns into $r$" under the risk-neutral measure: the drift changes so that discounted prices become martingales.

> With a continuous dividend yield $q$, the same reasoning leads to the risk-neutral drift $r-q$ rather than $r$.

### Why this matters for pricing in practice

Once we work under $\mathbb{Q}$, derivative pricing becomes a discounted expectation. For a payoff $V_T$ at time $T$, the no-arbitrage price at time $t$ is
$$
V_t = B_t\,\mathbb{E}^{\mathbb{Q}}\!\left[\left.\frac{V_T}{B_T}\right|\mathcal{F}_t\right].
$$
This is why implementations of Black--Scholes pricing and implied volatility inversion depend on $(r,\sigma)$ (and possibly $q$), but **not** on the real-world drift $\mu$: $\mu$ is absorbed into the change of measure, and pricing is carried out under $\mathbb{Q}$.
