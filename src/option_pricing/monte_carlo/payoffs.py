"""Optional payoff adapters for Monte Carlo path pricing.

This module is intended for reusable payoff-shaping helpers only, especially if
path-dependent payoffs such as Asian, barrier, lookback, or cliquet-style
contracts grow beyond simple instrument methods.

Instrument-specific payoff behavior should generally remain with the instrument
or pricing wrapper unless it becomes broadly reusable.
"""
