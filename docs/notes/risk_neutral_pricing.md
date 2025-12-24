Under the **risk-neutral measure**, we adjust the drift so that the **discounted price** $e^{-rt}S_t$ becomes a martingale. Intuitively, this means that, once you remove the risk-free rate, there is no predictable way to make excess profit: the “fair” expected return of the asset is just the risk-free rate. Brownian motion is still the source of randomness, but the drift changes, and this is what ultimately leads to the Black–Scholes option pricing formula, which we will see in a later notebook _{insert notebook}_.
# Risk-neutral pricing

These document contains notes justifying some of the assumptions made in this library when it comes to risk-neutral pricing.
## Goals:
- Explain why under the risk neutral measure $\mathbb{Q}$, the drift term in $\mu$ turns to the instanteneous risk-neutral rate of return $r$.
- Reiterate and underline under which assumptions they hold, and what to look out for.
- Illustrate how we use this in practice