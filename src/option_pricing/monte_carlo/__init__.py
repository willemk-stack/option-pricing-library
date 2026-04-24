"""Monte Carlo infrastructure.

Shared Monte Carlo building blocks for option pricing: run configuration, RNG
construction, estimators, results, and optional payoff adapters.

This package should stay model-agnostic. Model-specific path evolution belongs
under ``option_pricing.models``, and user-facing pricing wrappers belong under
``option_pricing.pricers``.
"""

from .config import MCConfig, RandomConfig, RngType
from .engine import (
    mc_price_path_instrument_from_ctx,
    mc_price_terminal_instrument_from_ctx,
)
from .estimators import (
    ControlVariate,
    apply_control_variate,
    estimate_discounted_payoff,
    estimate_mean_stderr,
    pair_antithetic,
)
from .results import MonteCarloResult, monte_carlo_result_from_samples
from .rng import correlated_normals, make_rng, rng_from_random_config, standard_normals
from .samples import (
    as_1d_samples,
    effective_payoff_samples,
    effective_samples_from_payoffs,
)
from .simulators import PathSimulator, TerminalSimulator

__all__ = [
    # cfg objects
    "MCConfig",
    "RandomConfig",
    "RngType",
    # engine orchestration
    "mc_price_terminal_instrument_from_ctx",
    "mc_price_path_instrument_from_ctx",
    # rng/noise generation
    "make_rng",
    "rng_from_random_config",
    "standard_normals",
    "correlated_normals",
    "TerminalSimulator",
    "PathSimulator",
    # sample validation/effective samples
    "as_1d_samples",
    "effective_samples_from_payoffs",
    "effective_payoff_samples",
    # estimators/estimation
    "ControlVariate",
    "apply_control_variate",
    "estimate_discounted_payoff",
    "estimate_mean_stderr",
    "pair_antithetic",
    # result containers
    "MonteCarloResult",
    "monte_carlo_result_from_samples",
]
