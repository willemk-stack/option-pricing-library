import math
from dataclasses import dataclass

import numpy as np

from option_pricing.instruments import ExerciseStyle, VanillaOption
from option_pricing.market.curves import (
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    PricingContext,
)
from option_pricing.monte_carlo import (
    MCConfig,
    RandomConfig,
    mc_price_path_instrument_from_ctx,
    mc_price_terminal_instrument_from_ctx,
)
from option_pricing.monte_carlo.estimators import ControlVariate
from option_pricing.types import OptionType


@dataclass(frozen=True)
class _FixedTerminalSimulator:
    samples: np.ndarray

    def simulate_terminal(self, *, ctx, tau, cfg):
        return self.samples


@dataclass(frozen=True)
class _FixedPathSimulator:
    paths: np.ndarray

    def simulate_paths(self, *, ctx, tau, cfg):
        return self.paths


@dataclass(frozen=True)
class _AsianCallInstrument:
    expiry: float
    strike: float
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN

    @property
    def payoff(self):
        return lambda paths: np.maximum(np.mean(paths, axis=1) - self.strike, 0.0)


def _pricing_context(
    *, spot: float = 100.0, r: float = 0.05, q: float = 0.0
) -> PricingContext:
    return PricingContext(
        spot=spot,
        discount=FlatDiscountCurve(r),
        forward=FlatCarryForwardCurve(spot=spot, r=r, q=q),
    )


def test_terminal_engine_uses_simulator_samples_and_ctx_discount() -> None:
    ctx = _pricing_context(r=0.05)
    inst = VanillaOption(
        expiry=1.0,
        strike=100.0,
        kind=OptionType.CALL,
        exercise=ExerciseStyle.EUROPEAN,
    )
    samples = np.array([90.0, 110.0, 120.0, 80.0])

    result = mc_price_terminal_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=_FixedTerminalSimulator(samples=samples),
        cfg=MCConfig(n_paths=4),
    )

    expected_payoffs = inst.payoff(samples)
    expected_price = ctx.df(inst.expiry) * float(np.mean(expected_payoffs))
    expected_stderr = (
        ctx.df(inst.expiry)
        * float(np.std(expected_payoffs, ddof=1))
        / math.sqrt(len(expected_payoffs))
    )

    assert math.isclose(result.price, expected_price, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.stderr, expected_stderr, rel_tol=0.0, abs_tol=1e-12)
    assert result.effective_n == len(expected_payoffs)


def test_terminal_engine_antithetic_reports_half_effective_sample_count() -> None:
    ctx = _pricing_context(r=0.05)

    class _IdentityInstrument:
        expiry = 1.0
        exercise = ExerciseStyle.EUROPEAN

        @property
        def payoff(self):
            return lambda terminal: terminal

    terminal = np.array([90.0, 110.0, 130.0, 150.0])
    result = mc_price_terminal_instrument_from_ctx(
        ctx=ctx,
        inst=_IdentityInstrument(),
        simulator=_FixedTerminalSimulator(samples=terminal),
        cfg=MCConfig(n_paths=4, antithetic=True, random=RandomConfig(seed=3)),
    )

    paired = 0.5 * (terminal[:2] + terminal[2:])
    expected_stderr = (
        ctx.df(1.0) * float(np.std(paired, ddof=1)) / math.sqrt(len(paired))
    )

    assert result.effective_n == 2
    assert math.isclose(result.stderr, expected_stderr, rel_tol=0.0, abs_tol=1e-12)


def test_terminal_engine_supports_control_variate() -> None:
    ctx = _pricing_context(spot=100.0, r=0.05, q=0.02)

    class _IdentityInstrument:
        expiry = 1.5
        exercise = ExerciseStyle.EUROPEAN

        @property
        def payoff(self):
            return lambda terminal: terminal

    terminal = np.array([90.0, 100.0, 110.0, 120.0])
    control = ControlVariate(
        values=lambda samples: samples, mean=float(np.mean(terminal))
    )

    result = mc_price_terminal_instrument_from_ctx(
        ctx=ctx,
        inst=_IdentityInstrument(),
        simulator=_FixedTerminalSimulator(samples=terminal),
        cfg=MCConfig(n_paths=4),
        control=control,
    )

    assert math.isclose(result.sample_std or 0.0, 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert result.stderr == 0.0


def test_path_engine_uses_simulator_samples_and_ctx_discount() -> None:
    ctx = _pricing_context(r=0.05)
    inst = _AsianCallInstrument(expiry=1.0, strike=100.0)
    paths = np.array(
        [
            [100.0, 110.0, 120.0],
            [100.0, 100.0, 100.0],
            [100.0, 90.0, 80.0],
        ]
    )

    result = mc_price_path_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=_FixedPathSimulator(paths=paths),
        cfg=MCConfig(n_paths=3),
    )

    payoff_values = inst.payoff(paths)
    expected_price = ctx.df(inst.expiry) * float(np.mean(payoff_values))
    expected_stderr = (
        ctx.df(inst.expiry)
        * float(np.std(payoff_values, ddof=1))
        / math.sqrt(len(payoff_values))
    )

    assert math.isclose(result.price, expected_price, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.stderr, expected_stderr, rel_tol=0.0, abs_tol=1e-12)
    assert result.effective_n == len(payoff_values)


def test_path_engine_antithetic_reports_half_effective_sample_count() -> None:
    ctx = _pricing_context(r=0.05)
    inst = _AsianCallInstrument(expiry=1.0, strike=100.0)
    paths = np.array(
        [
            [100.0, 110.0, 120.0],
            [100.0, 100.0, 100.0],
            [100.0, 120.0, 140.0],
            [100.0, 80.0, 60.0],
        ]
    )

    result = mc_price_path_instrument_from_ctx(
        ctx=ctx,
        inst=inst,
        simulator=_FixedPathSimulator(paths=paths),
        cfg=MCConfig(n_paths=4, antithetic=True, random=RandomConfig(seed=9)),
    )

    payoff_values = inst.payoff(paths)
    paired = 0.5 * (payoff_values[:2] + payoff_values[2:])
    expected_stderr = (
        ctx.df(inst.expiry) * float(np.std(paired, ddof=1)) / math.sqrt(len(paired))
    )

    assert result.effective_n == 2
    assert math.isclose(result.stderr, expected_stderr, rel_tol=0.0, abs_tol=1e-12)
