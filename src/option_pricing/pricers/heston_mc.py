from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..instruments.base import ExerciseStyle, PathPayoff, TerminalInstrument
from ..instruments.factory import from_pricing_inputs
from ..instruments.vanilla import VanillaOption
from ..market.curves import PricingContext
from ..models.heston import HestonParams
from ..models.heston.simulation import (
    HestonPathSimulator,
    HestonScheme,
    HestonTerminalSimulator,
)
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


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    """Normalize flat MarketData or an already-curves-first PricingContext."""
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def heston_mc_price_from_ctx(
    *,
    ctx: PricingContext,
    kind: OptionType,
    strike: float,
    params: HestonParams,
    tau: float,
    n_steps: int,
    scheme: HestonScheme = "euler_full_truncation",
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """Monte Carlo Heston pricer using curves-first inputs."""
    inst = VanillaOption(
        expiry=float(tau),
        strike=float(strike),
        kind=kind,
        exercise=ExerciseStyle.EUROPEAN,
    )
    return heston_mc_price_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        params=params,
        n_steps=n_steps,
        scheme=scheme,
        cfg=cfg,
        control=control,
    )


def heston_mc_price_instrument_from_ctx(
    *,
    ctx: PricingContext,
    inst: TerminalInstrument,
    params: HestonParams,
    n_steps: int,
    scheme: HestonScheme = "euler_full_truncation",
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """Monte Carlo Heston pricer for a European terminal-payoff instrument."""
    return _mc_price_terminal_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=HestonTerminalSimulator(
            params=params,
            n_steps=int(n_steps),
            scheme=scheme,
        ),
        cfg=cfg,
        control=control,
    )


def heston_mc_price_path_payoff_from_ctx(
    *,
    ctx: PricingContext,
    payoff: PathPayoff,
    params: HestonParams,
    tau: float,
    n_steps: int,
    scheme: HestonScheme = "euler_full_truncation",
    cfg: MCConfig | None = None,
) -> MonteCarloResult:
    """Price a European path payoff under Heston on an explicit time grid."""
    tau = float(tau)
    time_grid = np.linspace(0.0, tau, int(n_steps) + 1, dtype=FloatDType)
    inst = _PathPayoffInstrument(
        expiry=tau,
        _payoff=_GridBoundPathPayoff(base_payoff=payoff, time_grid=time_grid),
    )
    result = _mc_price_path_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=HestonPathSimulator(
            params=params,
            n_steps=int(n_steps),
            scheme=scheme,
        ),
        cfg=cfg,
    )
    return replace(
        result,
        metadata={
            **dict(result.metadata),
            "n_steps": int(n_steps),
            "scheme": str(scheme),
        },
    )


def heston_mc_price_instrument(
    inst: TerminalInstrument,
    *,
    market: MarketData | PricingContext,
    params: HestonParams,
    n_steps: int,
    scheme: HestonScheme = "euler_full_truncation",
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """Convenience wrapper accepting flat `MarketData` or `PricingContext`."""
    return heston_mc_price_instrument_from_ctx(
        ctx=_to_ctx(market),
        inst=inst,
        params=params,
        n_steps=n_steps,
        scheme=scheme,
        cfg=cfg,
        control=control,
    )


def heston_mc_price(
    p: PricingInputs,
    *,
    params: HestonParams,
    n_steps: int,
    scheme: HestonScheme = "euler_full_truncation",
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    """Price a European vanilla option under Heston Monte Carlo simulation.

    The stochastic-volatility model is supplied through ``params``; this helper
    uses ``PricingInputs`` for the market, strike, and time-to-expiry fields.
    """
    inst = from_pricing_inputs(p, exercise=ExerciseStyle.EUROPEAN)
    return heston_mc_price_instrument_from_ctx(
        ctx=p.ctx,
        inst=inst,
        params=params,
        n_steps=n_steps,
        scheme=scheme,
        cfg=cfg,
        control=control,
    )


def heston_mc_price_call(
    p: PricingInputs,
    *,
    params: HestonParams,
    n_steps: int,
    scheme: HestonScheme = "euler_full_truncation",
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    return heston_mc_price_from_ctx(
        ctx=p.ctx,
        kind=OptionType.CALL,
        strike=p.K,
        params=params,
        tau=p.tau,
        n_steps=n_steps,
        scheme=scheme,
        cfg=cfg,
        control=control,
    )


def heston_mc_price_put(
    p: PricingInputs,
    *,
    params: HestonParams,
    n_steps: int,
    scheme: HestonScheme = "euler_full_truncation",
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    return heston_mc_price_from_ctx(
        ctx=p.ctx,
        kind=OptionType.PUT,
        strike=p.K,
        params=params,
        tau=p.tau,
        n_steps=n_steps,
        scheme=scheme,
        cfg=cfg,
        control=control,
    )


__all__ = [
    "heston_mc_price",
    "heston_mc_price_call",
    "heston_mc_price_from_ctx",
    "heston_mc_price_instrument",
    "heston_mc_price_instrument_from_ctx",
    "heston_mc_price_path_payoff_from_ctx",
    "heston_mc_price_put",
]
