## Theory recap
A binomial tree is a **discrete-time model** for the evolution of a stock price. Time is divided into steps of length $\Delta t$, and at each step the stock either moves **up by a factor $u$** or **down by a factor $d$**. Starting from $S_0$, this generates all possible stock prices up to the option's maturity $T=n\Delta t$.

We make the following standard assumptions (so that the model is arbitrage-free and complete):

### Market Assumptions

* Trading is **frictionless**: no transaction costs, assets can be traded in any volume, and short-selling is allowed.
* There is a **risk-free money-market account** that earns a constant continuously compounded rate $r$. Over one time step, one unit grows to
    $$R=e^{r\Delta t}.$$

---

### Stock Process Assumptions

* At each step, the stock moves to either $uS_k$ (up) or $dS_k$ (down), with $0<d<u$.
* The parameters satisfy
    $$d<R<u,$$
    which rules out arbitrage and ensures the existence of a risk-neutral probability.

---

### Risk-Neutral Pricing

To price an option, we work under a **risk-neutral measure $\mathbb{Q}$** in which every asset has expected return $r$. In this measure the discounted stock price is a martingale, which in the binomial model implies the **risk-neutral probability** $p$:
$$p = \frac{R - d}{u-d} = \frac{e^{r\Delta t} - d}{u-d},$$
with $0<p<1$ guaranteed by $d<R<u$.

Let $V_k$ be the option value at time $t_k = k\Delta t$, and let $\mathcal{F}_k$ be the information (the sigma-algebra) up to time $t_k$. The **fundamental risk-neutral pricing relation** is:
$$V_k = e^{-r\Delta t}\,\mathbb{E}^{\mathbb{Q}}[\,V_{k+1} \mid \mathcal{F}_k\,].$$
In the one-step binomial setting this becomes the **backward induction formula**:
$$V_k = e^{-r\Delta t}\Big(p\,V_{k+1}^{(u)} + (1-p)\,V_{k+1}^{(d)}\Big),$$
where $V_{k+1}^{(u)}$ and $V_{k+1}^{(d)}$ denote the option values at the up- and down-nodes one step ahead.

---

### Connecting to Black–Scholes (CRR Model)

To connect the binomial model to the Black--Scholes setting, we choose the **Cox–Ross–Rubinstein (CRR) parameters**:
$$u = e^{\sigma\sqrt{\Delta t}}, \qquad d = e^{-\sigma\sqrt{\Delta t}},$$
so that, as $\Delta t \to 0$, the mean and variance of log-returns over a single step match those of a geometric Brownian motion with volatility $\sigma$. With this choice of $u$ and $d$, repeated application of the recursion above yields binomial prices that converge to:
$$V_0 = e^{-rT} \sum_{j=0}^{N} \binom{N}{j} p^{j}(1-p)^{N-j} g(S_0 u^j d^{N-j})$$

---

### Binomial trees beyond European options (context only)

Although we use the CRR tree here only for European calls/puts, the same framework is much more flexible than the closed-form Black–Scholes formula: