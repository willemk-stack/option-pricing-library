from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..instruments.base import (
    ExerciseStyle,
    PathPayoff,
    TerminalInstrument,
    TerminalPayoff,
)
from ..instruments.factory import from_pricing_inputs
from ..instruments.payoffs import make_vanilla_payoff
from ..market.curves import PricingContext
from ..models.gbm import GBMPathSimulator, GBMTerminalSimulator
from ..monte_carlo import MCConfig
from ..monte_carlo.engine import (
    mc_price_path_instrument_from_ctx as _mc_price_path_instrument_from_ctx,
)
from ..monte_carlo.engine import (
    mc_price_terminal_instrument_from_ctx as _mc_price_terminal_instrument_from_ctx,
)
from ..monte_carlo.estimators import ControlVariate
from ..monte_carlo.results import MonteCarloResult
from ..types import MarketData, OptionType, PricingInputs
from ..typing import FloatArray, FloatDType


@dataclass(frozen=True, slots=True)
class _TerminalPayoffInstrument:
    expiry: float
    _payoff: TerminalPayoff
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    @property
    def payoff(self) -> TerminalPayoff:
        return self._payoff


@dataclass(frozen=True, slots=True)
class _GridBoundPathPayoff:
    base_payoff: PathPayoff
    time_grid: FloatArray

    def __call__(
        self,
        paths: FloatArray,
        *,
        times: FloatArray | None = None,
    ) -> FloatArray:
        return self.base_payoff(paths, times=self.time_grid)


@dataclass(frozen=True, slots=True)
class _PathPayoffInstrument:
    expiry: float
    _payoff: PathPayoff
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    @property
    def payoff(self) -> PathPayoff:
        return self._payoff


def mc_price_from_ctx(
    *,
    ctx: PricingContext,
    kind: OptionType,
    strike: float,
    sigma: float,
    tau: float,
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """Monte Carlo GBM pricer using curves-first inputs."""
    inst = _TerminalPayoffInstrument(
        expiry=float(tau),
        _payoff=make_vanilla_payoff(kind, K=float(strike)),
    )
    return _mc_price_terminal_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=GBMTerminalSimulator(sigma=float(sigma)),
        cfg=cfg,
        control=control,
    )


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    """Normalize flat MarketData or an already-curves-first PricingContext."""
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def mc_price_instrument_from_ctx(
    *,
    ctx: PricingContext,
    inst: TerminalInstrument,
    sigma: float,
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """Monte Carlo GBM pricer for a European terminal-payoff instrument."""
    return _mc_price_terminal_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=GBMTerminalSimulator(sigma=float(sigma)),
        cfg=cfg,
        control=control,
    )


def mc_price_path_payoff_from_ctx(
    *,
    ctx: PricingContext,
    payoff: PathPayoff,
    sigma: float,
    tau: float,
    n_steps: int,
    cfg: MCConfig | None = None,
) -> MonteCarloResult:
    """Price a European path payoff under GBM on an explicit time grid."""
    tau = float(tau)
    time_grid = np.linspace(0.0, tau, int(n_steps) + 1, dtype=FloatDType)
    inst = _PathPayoffInstrument(
        expiry=tau,
        _payoff=_GridBoundPathPayoff(base_payoff=payoff, time_grid=time_grid),
    )
    result = _mc_price_path_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=GBMPathSimulator(sigma=float(sigma), n_steps=int(n_steps)),
        cfg=cfg,
    )
    return replace(
        result,
        metadata={**dict(result.metadata), "n_steps": int(n_steps)},
    )


def mc_price_instrument(
    inst: TerminalInstrument,
    *,
    market: MarketData | PricingContext,
    sigma: float,
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """Convenience wrapper accepting flat `MarketData` or `PricingContext`."""
    return mc_price_instrument_from_ctx(
        ctx=_to_ctx(market),
        inst=inst,
        sigma=sigma,
        cfg=cfg,
        control=control,
    )


def mc_price(
    p: PricingInputs,
    *,
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """
    Price a European vanilla option using Monte Carlo GBM simulation.

    Examples
    --------
    >>> from option_pricing.monte_carlo import MCConfig, RandomConfig
    >>> cfg = MCConfig(n_paths=200_000, antithetic=True, random=RandomConfig(seed=123))
    >>> result = mc_price(p, cfg=cfg)
    >>> result.price, result.stderr
    """
    inst = from_pricing_inputs(p, exercise=ExerciseStyle.EUROPEAN)
    return mc_price_instrument_from_ctx(
        ctx=p.ctx,
        inst=inst,
        sigma=p.sigma,
        cfg=cfg,
        control=control,
    )


def mc_price_call(
    p: PricingInputs,
    *,
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    return mc_price_from_ctx(
        ctx=p.ctx,
        kind=OptionType.CALL,
        strike=p.K,
        sigma=p.sigma,
        tau=p.tau,
        cfg=cfg,
        control=control,
    )


def mc_price_put(
    p: PricingInputs,
    *,
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    return mc_price_from_ctx(
        ctx=p.ctx,
        kind=OptionType.PUT,
        strike=p.K,
        sigma=p.sigma,
        tau=p.tau,
        cfg=cfg,
        control=control,
    )
