# Diagnostics

This folder contains **notebook-friendly** helpers for generating tables and plots.
The intention is to keep core pricing logic out of diagnostics.

## Structure

- `_mpl.py` shared matplotlib helpers
- `mc_vs_bs/` Monte Carlo vs Black-Scholes tables + plots
- `binom/` binomial convergence series + plots
- `iv/` implied-vol benchmarks + smile plots
- `gbm/` GBM distribution plots
- `greeks/` greek sweep helpers

## Moved modules

Put/Call parity helpers were moved into core: `option_pricing/arbitrage/parity.py`.
