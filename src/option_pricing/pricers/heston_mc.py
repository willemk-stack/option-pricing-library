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
from ..numerics.quadrature import CompositeRule, QuadratureConfig
from ..types import MarketData, OptionType, PricingInputs
from ..typing import FloatArray, FloatDType
from .heston import HestonBackend, heston_price_from_ctx


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


def heston_vanilla_control_from_ctx(
    *,
    ctx: PricingContext,
    kind: OptionType,
    strike: float,
    tau: float,
    params: HestonParams,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> ControlVariate:
    """Build a Heston vanilla payoff control variate.

    The generic MC engine applies controls to *undiscounted* payoff samples and
    discounts afterward. Therefore the known control mean is:

        E[Y] = semi_analytic_price / df(tau)

    not the discounted option price itself.
    """
    inst = VanillaOption(
        expiry=float(tau),
        strike=float(strike),
        kind=kind,
        exercise=ExerciseStyle.EUROPEAN,
    )

    analytic_price = float(
        heston_price_from_ctx(
            kind=kind,
            strike=float(strike),
            tau=float(tau),
            ctx=ctx,
            params=params,
            backend=backend,
            quad_cfg=quad_cfg,
            rule=rule,
        )
    )

    df = float(ctx.df(float(tau)))
    if not np.isfinite(df) or df <= 0.0:
        raise ValueError("discount factor must be finite and positive")

    return ControlVariate(
        values=inst.payoff,
        mean=analytic_price / df,
    )


def heston_mc_price_from_ctx(
    *,
    ctx: PricingContext,
    kind: OptionType,
    strike: float,
    params: HestonParams,
    tau: float,
    n_steps: int,
    scheme: HestonScheme = "quadratic_exponential",
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
    scheme: HestonScheme = "quadratic_exponential",
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
    scheme: HestonScheme = "quadratic_exponential",
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
    scheme: HestonScheme = "quadratic_exponential",
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
    scheme: HestonScheme = "quadratic_exponential",
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
    scheme: HestonScheme = "quadratic_exponential",
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
    scheme: HestonScheme = "quadratic_exponential",
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


def heston_mc_price_with_vanilla_control_from_ctx(
    *,
    ctx: PricingContext,
    kind: OptionType,
    strike: float,
    params: HestonParams,
    tau: float,
    n_steps: int,
    scheme: HestonScheme = "quadratic_exponential",
    cfg: MCConfig | None = None,
    control_kind: OptionType | None = None,
    control_strike: float | None = None,
    backend: HestonBackend = "gauss_legendre",
    quad_cfg: QuadratureConfig | None = None,
    rule: CompositeRule | None = None,
) -> MonteCarloResult:
    """Price under Heston MC using a semi-analytic vanilla control variate.

    For validating vanilla MC against Fourier, usually run without this control.
    If the control is exactly the same payoff as the target payoff, the estimator
    collapses almost entirely to the semi-analytic price and stops being a useful
    MC cross-check.
    """
    control = heston_vanilla_control_from_ctx(
        ctx=ctx,
        kind=kind if control_kind is None else control_kind,
        strike=float(strike if control_strike is None else control_strike),
        tau=float(tau),
        params=params,
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
    )

    result = heston_mc_price_from_ctx(
        ctx=ctx,
        kind=kind,
        strike=float(strike),
        params=params,
        tau=float(tau),
        n_steps=int(n_steps),
        scheme=scheme,
        cfg=cfg,
        control=control,
    )

    return replace(
        result,
        metadata={
            **dict(result.metadata),
            "variance_reduction": {
                "antithetic": bool(cfg.antithetic) if cfg is not None else False,
                "control": "heston_semi_analytic_vanilla",
                "control_kind": str(kind if control_kind is None else control_kind),
                "control_strike": float(
                    strike if control_strike is None else control_strike
                ),
            },
        },
    )


__all__ = [
    "heston_vanilla_control_from_ctx",
    "heston_mc_price_with_vanilla_control_from_ctx",
    "heston_mc_price",
    "heston_mc_price_call",
    "heston_mc_price_from_ctx",
    "heston_mc_price_instrument",
    "heston_mc_price_instrument_from_ctx",
    "heston_mc_price_path_payoff_from_ctx",
    "heston_mc_price_put",
]
