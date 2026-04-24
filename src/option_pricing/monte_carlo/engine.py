"""Generic Monte Carlo pricing orchestration."""

from __future__ import annotations

import numpy as np

from ..instruments.base import ExerciseStyle, PathInstrument, TerminalInstrument
from ..market.curves import PricingContext
from ..typing import FloatDType
from .config import MCConfig
from .estimators import ControlVariate
from .results import MonteCarloResult, monte_carlo_result_from_samples
from .samples import (
    as_1d_samples,
    effective_payoff_samples,
    effective_samples_from_payoffs,
)
from .simulators import PathSimulator, TerminalSimulator


def _cfg_or_default(cfg: MCConfig | None) -> MCConfig:
    return MCConfig() if cfg is None else cfg


def _seed_from_cfg(cfg: MCConfig) -> int | None:
    return None if cfg.rng is not None else int(cfg.random.seed)


def mc_price_terminal_instrument_from_ctx(
    *,
    ctx: PricingContext,
    inst: TerminalInstrument,
    simulator: TerminalSimulator,
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """Price a European terminal-payoff instrument from simulator samples."""
    if inst.exercise != ExerciseStyle.EUROPEAN:
        raise ValueError(
            "mc_price_terminal_instrument_from_ctx supports European exercise only"
        )

    cfg = _cfg_or_default(cfg)
    tau = float(inst.expiry)
    terminal = as_1d_samples(
        simulator.simulate_terminal(ctx=ctx, tau=tau, cfg=cfg),
        name="terminal",
        expected_shape=(int(cfg.n_paths),),
    )
    payoff_values = as_1d_samples(
        inst.payoff(terminal),
        name="payoff",
        expected_shape=terminal.shape,
    )
    effective_samples = effective_payoff_samples(
        terminal=terminal,
        payoff_values=payoff_values,
        antithetic=bool(cfg.antithetic),
        control=control,
    )
    return monte_carlo_result_from_samples(
        effective_samples,
        discount=ctx.df(tau),
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        seed=_seed_from_cfg(cfg),
    )


def mc_price_path_instrument_from_ctx(
    *,
    ctx: PricingContext,
    inst: PathInstrument,
    simulator: PathSimulator,
    cfg: MCConfig | None = None,
) -> MonteCarloResult:
    """Price a European path-payoff instrument from simulator paths."""
    if inst.exercise != ExerciseStyle.EUROPEAN:
        raise ValueError(
            "mc_price_path_instrument_from_ctx supports European exercise only"
        )

    cfg = _cfg_or_default(cfg)
    tau = float(inst.expiry)
    paths = np.asarray(
        simulator.simulate_paths(ctx=ctx, tau=tau, cfg=cfg),
        dtype=FloatDType,
    )
    if paths.ndim < 2:
        raise ValueError(
            "path samples must have path dimension first and at least 2 dimensions"
        )
    if paths.shape[0] != int(cfg.n_paths):
        raise ValueError(
            f"path samples must have {int(cfg.n_paths)} paths; got {paths.shape[0]}"
        )

    payoff_values = as_1d_samples(
        inst.payoff(paths),
        name="payoff",
        expected_shape=(paths.shape[0],),
    )
    effective_samples = effective_samples_from_payoffs(
        payoff_values,
        antithetic=bool(cfg.antithetic),
    )
    return monte_carlo_result_from_samples(
        effective_samples,
        discount=ctx.df(tau),
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        seed=_seed_from_cfg(cfg),
    )
