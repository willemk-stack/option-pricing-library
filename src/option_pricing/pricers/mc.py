from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..instruments.base import (
    ExerciseStyle,
    PathInstrument,
    PathPayoff,
    TerminalInstrument,
    TerminalPayoff,
)
from ..instruments.factory import from_pricing_inputs
from ..instruments.payoffs import make_vanilla_payoff
from ..market.curves import PricingContext
from ..models.gbm import GBMParams, simulate_gbm_paths, simulate_gbm_terminal
from ..monte_carlo import MCConfig
from ..monte_carlo.estimators import (
    ControlVariate,
    apply_control_variate,
    pair_antithetic,
)
from ..monte_carlo.results import MonteCarloResult, monte_carlo_result_from_samples
from ..monte_carlo.rng import make_rng, standard_normals
from ..monte_carlo.simulators import PathSimulator, TerminalSimulator
from ..types import MarketData, OptionType, PricingInputs
from ..typing import FloatArray, FloatDType


def _cfg_or_default(cfg: MCConfig | None) -> MCConfig:
    return MCConfig() if cfg is None else cfg


def _as_1d_samples(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> FloatArray:
    samples = np.asarray(values, dtype=FloatDType)
    if samples.shape != expected_shape:
        raise ValueError(
            f"{name} samples must have shape {expected_shape}; got {samples.shape}"
        )
    return samples


def _effective_samples_from_payoffs(
    payoff_values: FloatArray,
    *,
    antithetic: bool,
) -> FloatArray:
    if not antithetic:
        return payoff_values

    if payoff_values.shape[0] % 2 != 0:
        raise ValueError("antithetic=True requires an even number of payoff samples")

    n_pairs = payoff_values.shape[0] // 2
    return pair_antithetic(payoff_values[:n_pairs], payoff_values[n_pairs:])


def _simulate_terminal_gbm(
    *,
    spot: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    n_paths: int,
    antithetic: bool,
    rng: np.random.Generator,
) -> FloatArray:
    normals = standard_normals(
        rng,
        n_paths=int(n_paths),
        antithetic=bool(antithetic),
        dtype=np.dtype(FloatDType),
    )
    params = GBMParams(spot=float(spot), drift=float(r) - float(q), sigma=float(sigma))
    return simulate_gbm_terminal(params=params, tau=float(tau), normals=normals)


def _simulate_paths_gbm(
    *,
    spot: float,
    r: float,
    q: float,
    sigma: float,
    time_grid: FloatArray,
    n_paths: int,
    antithetic: bool,
    rng: np.random.Generator,
) -> FloatArray:
    normals = standard_normals(
        rng,
        n_paths=int(n_paths),
        sample_shape=(len(time_grid) - 1,),
        antithetic=bool(antithetic),
        dtype=np.dtype(FloatDType),
    )
    params = GBMParams(spot=float(spot), drift=float(r) - float(q), sigma=float(sigma))
    return simulate_gbm_paths(params=params, time_grid=time_grid, normals=normals)


def _effective_payoff_samples(
    *,
    terminal: FloatArray,
    payoff_values: FloatArray,
    antithetic: bool,
    control: ControlVariate | None,
) -> FloatArray:
    if not antithetic:
        samples = payoff_values
        if control is not None:
            control_values = _as_1d_samples(
                control.values(terminal),
                name="control",
                expected_shape=terminal.shape,
            )
            samples = apply_control_variate(samples, control_values, control.mean)
        return samples

    samples = _effective_samples_from_payoffs(
        payoff_values,
        antithetic=antithetic,
    )

    if control is not None:
        n_pairs = terminal.shape[0] // 2
        control_values = _as_1d_samples(
            control.values(terminal),
            name="control",
            expected_shape=terminal.shape,
        )
        control_pairs = pair_antithetic(
            control_values[:n_pairs],
            control_values[n_pairs:],
        )
        samples = apply_control_variate(samples, control_pairs, control.mean)

    return samples


def _price_terminal_gbm(
    *,
    spot: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    payoff: TerminalPayoff | Callable[[FloatArray], FloatArray],
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    cfg = _cfg_or_default(cfg)
    tau = float(tau)
    sigma = float(sigma)

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    terminal = _simulate_terminal_gbm(
        spot=float(spot),
        r=float(r),
        q=float(q),
        sigma=sigma,
        tau=tau,
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        rng=make_rng(cfg),
    )
    payoff_values = _as_1d_samples(
        payoff(terminal),
        name="payoff",
        expected_shape=terminal.shape,
    )
    effective_samples = _effective_payoff_samples(
        terminal=terminal,
        payoff_values=payoff_values,
        antithetic=bool(cfg.antithetic),
        control=control,
    )
    discount = float(np.exp(-float(r) * tau))
    return monte_carlo_result_from_samples(
        effective_samples,
        discount=discount,
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        seed=None if cfg.rng is not None else int(cfg.random.seed),
    )


def _price_terminal_gbm_from_ctx(
    *,
    ctx: PricingContext,
    tau: float,
    sigma: float,
    payoff: TerminalPayoff | Callable[[FloatArray], FloatArray],
    cfg: MCConfig | None = None,
    control: ControlVariate | None = None,
) -> MonteCarloResult:
    tau = float(tau)
    return _price_terminal_gbm(
        spot=ctx.spot,
        r=ctx.r_avg(tau),
        q=ctx.q_avg(tau),
        sigma=float(sigma),
        tau=tau,
        payoff=payoff,
        cfg=cfg,
        control=control,
    )


@dataclass(frozen=True, slots=True)
class _GBMTerminalSimulator:
    sigma: float

    def simulate_terminal(
        self,
        *,
        ctx: PricingContext,
        tau: float,
        cfg: MCConfig,
    ) -> FloatArray:
        tau = float(tau)
        return _simulate_terminal_gbm(
            spot=ctx.spot,
            r=ctx.r_avg(tau),
            q=ctx.q_avg(tau),
            sigma=float(self.sigma),
            tau=tau,
            n_paths=int(cfg.n_paths),
            antithetic=bool(cfg.antithetic),
            rng=make_rng(cfg),
        )


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
    payoff = make_vanilla_payoff(kind, K=float(strike))
    return _price_terminal_gbm_from_ctx(
        ctx=ctx,
        tau=float(tau),
        sigma=float(sigma),
        payoff=payoff,
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
    return mc_price_terminal_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=_GBMTerminalSimulator(sigma=float(sigma)),
        cfg=cfg,
        control=control,
    )


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
    terminal = _as_1d_samples(
        simulator.simulate_terminal(ctx=ctx, tau=tau, cfg=cfg),
        name="terminal",
        expected_shape=(int(cfg.n_paths),),
    )
    payoff_values = _as_1d_samples(
        inst.payoff(terminal),
        name="payoff",
        expected_shape=terminal.shape,
    )
    effective_samples = _effective_payoff_samples(
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
        seed=None if cfg.rng is not None else int(cfg.random.seed),
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

    payoff_values = _as_1d_samples(
        inst.payoff(paths),
        name="payoff",
        expected_shape=(paths.shape[0],),
    )
    effective_samples = _effective_samples_from_payoffs(
        payoff_values,
        antithetic=bool(cfg.antithetic),
    )
    return monte_carlo_result_from_samples(
        effective_samples,
        discount=ctx.df(tau),
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        seed=None if cfg.rng is not None else int(cfg.random.seed),
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
    cfg = _cfg_or_default(cfg)
    tau = float(tau)
    sigma = float(sigma)

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    time_grid = np.linspace(0.0, tau, int(n_steps) + 1, dtype=FloatDType)
    paths = _simulate_paths_gbm(
        spot=ctx.spot,
        r=ctx.r_avg(tau),
        q=ctx.q_avg(tau),
        sigma=sigma,
        time_grid=time_grid,
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        rng=make_rng(cfg),
    )
    payoff_values = _as_1d_samples(
        payoff(paths, times=time_grid),
        name="payoff",
        expected_shape=(paths.shape[0],),
    )
    effective_samples = _effective_samples_from_payoffs(
        payoff_values,
        antithetic=bool(cfg.antithetic),
    )
    return monte_carlo_result_from_samples(
        effective_samples,
        discount=ctx.df(tau),
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        seed=None if cfg.rng is not None else int(cfg.random.seed),
        metadata={"n_steps": int(n_steps)},
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
