# Notes

These notes provide background and derivations for the models implemented in **option_pricing**. They are more theory-oriented than the user guides and may include math-heavy explanations.

## Core topics

- [Risk neutral pricing](risk_neutral_pricing.md) — the pricing principle behind the library.
- [GBM](gbm.md) — the geometric Brownian motion model assumptions and implications.
- [Brownian motion and Itô](brownian_motion_and_Ito.md) — the calculus underpinning stochastic models.

## Model-specific notes

- [Black–Scholes pricing](bs_pricing.md) — derivation and interpretation of the BS formula.
- [Monte Carlo](mc.md) — estimators, variance reduction ideas, and numerical considerations.
- [Binomial CRR](binomial_crr.md) — discrete-time replication and convergence intuition.
- [Implied volatility](IV.md) — no-arbitrage bounds, inversion, and common pitfalls.