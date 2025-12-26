**Objectives**
- State and explain the assumptions of the Black-Scholes-Merton model.
- State the PDE and explain greeks
- Introduce a closed-form solution for European put/call options
- Provide derivation sketches for the PDE aswell as the closed form solutions for easy validation.
- Use `option_pricing.pricing_bs` to price calls and puts.
- Verify the implementation against textbook examples (e.g. Hull).

**References**
- Shreve, *Stochastic Calculus for Finance II*, Black–Scholes chapters.
- Hull, *Options, Futures, and Other Derivatives*, chapters on BS formula and Greeks.


## BSM assumptions and PDE
(1–2 paragraphs: market assumptions, no arbitrage, continuous trading, etc.)
Hull!!!!
**Assumptions:**
- Continuous trading
- No transaction costs
- No payed dividends during the lifetime of the option
- Constant interest rate $r$
- We are in a no arbitrage environment (what does this mean exactly?)
- Stock price follows a GBM as defined in notebook02 with constant drift $\mu$ and constant volatility $\sigma$


**PDE:**
(High-level story: replication → PDE → solution; minimal equations.)
We want to price a European option written on a stock whose price we model as a geometric Brownian motion (GBM). That is, we assume

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,
$$

in a frictionless market with constant volatility $\sigma$ and risk-free rate $r$. Our goal is to find a price for the option that does not create arbitrage opportunities. To do this, we consider an investor who holds a portfolio consisting of $\Delta_t$ units of the stock and some amount in the risk-free asset (bond or bank account). By choosing $\Delta_t$ appropriately, we can construct a self-financing portfolio that replicates the option’s payoff. No-arbitrage then implies that the value of this replicating portfolio must equal the option price at all times. Imposing this condition leads to the Black–Scholes partial differential equation for the option value.

Maybe give a small example to make no arbitrage intuitive??

$$
\frac{\partial V}{\partial t}
+ \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ r S \frac{\partial V}{\partial S}
- r V = 0,
$$

**Greeks**
insert section

## BSM closed form solution
In this section we present the closed form solution for a european call/put option.

1. For the price of european call option $c(t, x)$ with payoff, $(x-K)^+$ the closed form solution is given by:

    $$
    c(t,x) = x N(d_{+}(T-t, x)) - K e^{-r(T-t)} N(d_{-}(T-t, x)), \quad 0 \leq t < T, x > 0
    $$

    Where

    $$
    d_{\pm}(\tau, x) = \frac{1}{\sigma\sqrt{\tau}} \left[ \log \frac{x}{K} + \left(r \pm \frac{\sigma^2}{2}\right) \tau \right]
    $$

    and $N$ is the cumulative standard normal distribution. We also define $\tau=T-t$ as the time to expiration date.

2. For the price of european put option $p(t, x)$, with payoff $(K-x)^+$ the closed form solution is by:

    $$
    p(t, x) = K e^{-r(T-t)} N(-d_{-}(T-t, x)) - x N(-d_{+}(T-t, x))
    $$

    Source: Steve E shreve finance Vol II

# TODO: Complete explanation
**Intuitive explanation of terms:**
1. $K$ Strike price, the time to be paid at time T if the option is exercised
1. $Ke^{-r(T-t)}$ Present value of K at time t
1. $d_{+}(T-t, x)$
1. $d_{-}(T-t, x)$
1. $N(d_{+}(T-t, x))$
1. $N(d_{-}(T-t, x))$

**Poperties of the solution**

Lets take the closed form solution for a european call option $c(t,x)$. We will see that the solution is backed up by our intuition. If we were to let the stock price approach a very large value, the option ends almost surely _in the money_, and the formula reflects this as $N(d_{-})$ approaches 1. The formula then becomes:

$$
c(t, S(t)) = S(t) - Ke^{-r(T-t)}
$$

If the stockprice were to plummet, the option is unlikely to end in the money and we observe $\frac{x}{K}<<1$, then $N(d_\pm) \rightarrow 0$.

If we know look at the same situations but for the put formula, we observe the converse. Let x approach a large value and $p$ goes to:

$$
p(t, S(t)) = Ke^{-r(T-t)} - S(t)
$$

**Put–call parity**

For European options on a non-dividend-paying stock, the call and put prices must satisfy **put–call parity**:

$$
\boxed{
c(t,x) - p(t,x) = x - K e^{-r\tau}.
}
$$

This can be seen by comparing two portfolios:

- Portfolio A: one call and cash $K e^{-r\tau}$.
- Portfolio B: one put and one share of stock.

At maturity $T$, both portfolios produce the same payoff $ \max(S_T, K) $. By no-arbitrage, they must have the same value at time $t$, which leads to the parity relation above.

You can use this relationship as a strong consistency check for your numerical implementation of `bs_call` and `bs_put`.

## Interpretation and Connection to Earlier Notebooks
Earlier we modeled stock using the GBM given by:

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,
$$

Under the risk-neutral probability measure, this formula becomes:

$$
dS_t = r S_t\,dt + \sigma S_t\,dW_t^\mathbb{Q},
$$

Where intuitively the risk-neutral measure can be explained as a measure, where all assets have the same **expected** return $r$, which is the risk-free rate. This is also the reason why the solutions to the BSM are independent of $\mu$. The idea is that in an ideal risk-neutral world, investers don't need compensation if they were to say hold a risky stock vs investing in a money account. This is because they only care for the expected return which is equivalent in both.

In this notebook, we argued our way to the BSM PDE using the replication view. In the complete market, an investor can perfectly hedge an option, by investing in the stock and in the money market. The price to set up this hedge must then be equal to the option price.

Another way to look at it is by a probabilistic view. Where we define the option price to be the discounted expected payoff under the risk neutral measure, given the current underlying stock price.

$$
V(t,S_t)=e^{-r(T-t)}\mathbb{E}^Q[\text{payoff}\midS_t]
$$

As we expect, solving these two problems gives the same solution. An easy way to see this is by using the Feynman-Kac theorem to derive the equality of these two statements.

**Finance intuition**
- Link back to GBM and risk-neutral pricing.
- Brief finance intuition.
