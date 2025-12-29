# Notes

These notes are a compact “story” of the standard option-pricing toolkit used throughout the library:
**no-arbitrage → risk-neutral pricing → Brownian motion/Itô → GBM → Black–Scholes → implied vol → numerical methods**.

They are written to be readable linearly, but each note also stands on its own.

## Suggested reading order

1. [Risk-neutral pricing](risk_neutral_pricing.md) — why we price under \(\mathbb{Q}\) and why the drift becomes \(r\).
2. [Brownian motion and Itô](brownian_motion_and_Ito.md) — the noise and calculus behind continuous-time models.
3. [Geometric Brownian motion](gbm.md) — the lognormal stock model and distribution of returns.
4. [Black–Scholes pricing](bs_pricing.md) — PDE and closed-form European option prices, plus intuition.
5. [Implied volatility](IV.md) — “vol as the price”; inversion, bounds, and practical solver notes.
6. [Monte Carlo](mc.md) — simulation pricing, error bars, and variance reduction.
7. [Binomial CRR](binomial_crr.md) — discrete-time replication and convergence to Black–Scholes.

## Conventions and notation

- Time \(t\) is measured in **years**.
- Rates are **continuously compounded** unless stated otherwise.
- \(S_t\): underlying price at time \(t\).
- \(r\): continuously-compounded risk-free rate (constant in basic models).
- \(B_t = e^{rt}\): money-market account (numéraire in the basic setting).
- \(W_t\): Brownian motion.
- \(\mathbb{P}\): real-world (physical) probability measure.
- \(\mathbb{Q}\): risk-neutral measure under which discounted traded prices are martingales.

When a continuous dividend yield \(q\) is relevant, the risk-neutral drift becomes \(r-q\).
