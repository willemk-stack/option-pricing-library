import math
from dataclasses import dataclass

import numpy as np
import pytest

from option_pricing.instruments import ExerciseStyle, VanillaOption
from option_pricing.models.gbm import GBMParams, simulate_gbm_terminal
from option_pricing.monte_carlo import (
    ControlVariate,
    MCConfig,
    MonteCarloResult,
    RandomConfig,
    pair_antithetic,
    standard_normals,
)
from option_pricing.pricers.black_scholes import bs_price_call
from option_pricing.pricers.mc import (
    mc_price,
    mc_price_call,
    mc_price_from_ctx,
    mc_price_instrument,
    mc_price_path_instrument_from_ctx,
    mc_price_path_payoff_from_ctx,
    mc_price_terminal_instrument_from_ctx,
)
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


def test_mc_matches_bs_within_a_few_standard_errors(make_inputs):
    """MC price should agree with BS within a few reported SEs."""
    p = make_inputs(
        S=100.0,
        K=110.0,
        r=0.03,
        q=0.0,
        sigma=0.25,
        T=1.0,
        kind=OptionType.CALL,
    )

    bs = float(bs_price_call(p))
    cfg = MCConfig(n_paths=40_000, random=RandomConfig(seed=7))
    result = mc_price(p, cfg=cfg)

    assert isinstance(result, MonteCarloResult)
    assert result.stderr > 0.0
    assert abs(result.price - bs) <= (3.0 * result.stderr + 2e-3)


def test_mc_standard_error_scales_like_inverse_sqrt_n(make_inputs):
    """SE should scale ~ 1/sqrt(N) (approx; allow generous tolerance)."""
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )

    n1 = 5_000
    n2 = 20_000  # 4x more paths => SE should be ~ half

    se1 = mc_price_call(
        p,
        cfg=MCConfig(n_paths=n1, random=RandomConfig(seed=11)),
    ).stderr
    se2 = mc_price_call(
        p,
        cfg=MCConfig(n_paths=n2, random=RandomConfig(seed=12)),
    ).stderr

    ratio = se1 / se2
    assert 1.4 <= ratio <= 2.8


def test_mc_seed_is_deterministic_across_public_wrappers(make_inputs):
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.02,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    inst = VanillaOption(
        expiry=p.tau,
        strike=p.K,
        kind=p.spec.kind,
        exercise=ExerciseStyle.EUROPEAN,
    )
    cfg = MCConfig(n_paths=8_000, antithetic=True, random=RandomConfig(seed=7))

    result_inputs = mc_price(p, cfg=cfg)
    result_ctx = mc_price_from_ctx(
        ctx=p.ctx,
        kind=p.spec.kind,
        strike=p.K,
        sigma=p.sigma,
        tau=p.tau,
        cfg=cfg,
    )
    result_inst = mc_price_instrument(inst, market=p.ctx, sigma=p.sigma, cfg=cfg)

    assert math.isclose(
        result_inputs.price, result_ctx.price, rel_tol=0.0, abs_tol=1e-12
    )
    assert math.isclose(
        result_inputs.price, result_inst.price, rel_tol=0.0, abs_tol=1e-12
    )
    assert math.isclose(
        result_inputs.stderr, result_ctx.stderr, rel_tol=0.0, abs_tol=1e-12
    )
    assert math.isclose(
        result_inputs.stderr, result_inst.stderr, rel_tol=0.0, abs_tol=1e-12
    )
    assert (
        result_inputs.effective_n == result_ctx.effective_n == result_inst.effective_n
    )


def test_antithetic_pricing_uses_pair_averages_for_effective_stderr():
    spot = 100.0
    r = 0.05
    q = 0.01
    sigma = 0.2
    tau = 1.0
    n_paths = 4

    class _IdentityInstrument:
        expiry = tau
        exercise = ExerciseStyle.EUROPEAN

        @property
        def payoff(self):
            return lambda terminal: terminal

    class _DeterministicTerminalSimulator:
        def simulate_terminal(self, *, ctx, tau, cfg):
            return simulate_gbm_terminal(
                params=GBMParams(spot=spot, drift=r - q, sigma=sigma),
                tau=tau,
                normals=standard_normals(
                    np.random.default_rng(123),
                    n_paths=n_paths,
                    antithetic=True,
                ),
            )

    from option_pricing.market.curves import (
        FlatCarryForwardCurve,
        FlatDiscountCurve,
        PricingContext,
    )

    ctx = PricingContext(
        spot=spot,
        discount=FlatDiscountCurve(r),
        forward=FlatCarryForwardCurve(spot=spot, r=r, q=q),
    )
    result = mc_price_terminal_instrument_from_ctx(
        ctx=ctx,
        inst=_IdentityInstrument(),
        simulator=_DeterministicTerminalSimulator(),
        cfg=MCConfig(n_paths=n_paths, antithetic=True, random=RandomConfig(seed=123)),
    )

    normals = standard_normals(
        np.random.default_rng(123),
        n_paths=n_paths,
        antithetic=True,
    )
    terminal = simulate_gbm_terminal(
        params=GBMParams(spot=spot, drift=r - q, sigma=sigma),
        tau=tau,
        normals=normals,
    )
    paired = pair_antithetic(terminal[:2], terminal[2:])
    expected_mean = float(np.mean(paired))
    expected_std = float(np.std(paired, ddof=1))
    expected_price = math.exp(-r * tau) * expected_mean
    expected_stderr = math.exp(-r * tau) * expected_std / math.sqrt(len(paired))

    assert math.isclose(result.price, expected_price, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.stderr, expected_stderr, rel_tol=0.0, abs_tol=1e-12)
    assert result.effective_n == len(paired)


def test_control_variate_flows_through_canonical_estimator():
    spot = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.25
    tau = 1.5
    expected_terminal_mean = spot * math.exp((r - q) * tau)

    class _IdentityInstrument:
        expiry = tau
        exercise = ExerciseStyle.EUROPEAN

        @property
        def payoff(self):
            return lambda terminal: terminal

    from option_pricing.market.curves import (
        FlatCarryForwardCurve,
        FlatDiscountCurve,
        PricingContext,
    )

    ctx = PricingContext(
        spot=spot,
        discount=FlatDiscountCurve(r),
        forward=FlatCarryForwardCurve(spot=spot, r=r, q=q),
    )
    control = ControlVariate(
        values=lambda terminal: terminal,
        mean=expected_terminal_mean,
    )

    result = mc_price_instrument(
        _IdentityInstrument(),
        market=ctx,
        sigma=sigma,
        cfg=MCConfig(n_paths=1_000, rng=np.random.default_rng(321)),
        control=control,
    )

    assert math.isclose(
        result.price, spot * math.exp(-q * tau), rel_tol=0.0, abs_tol=1e-12
    )
    assert result.stderr <= 1e-12


def test_antithetic_rejects_odd_path_count(make_inputs):
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    cfg = MCConfig(n_paths=101, antithetic=True, random=RandomConfig(seed=1))

    with pytest.raises(ValueError, match="antithetic=True requires an even n_paths"):
        mc_price(p, cfg=cfg)


def test_terminal_instrument_pricer_uses_simulator_samples_and_discount(make_inputs):
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    inst = VanillaOption(
        expiry=p.tau,
        strike=p.K,
        kind=p.spec.kind,
        exercise=ExerciseStyle.EUROPEAN,
    )
    samples = np.array([90.0, 110.0, 120.0, 80.0])
    cfg = MCConfig(n_paths=4)

    result = mc_price_terminal_instrument_from_ctx(
        ctx=p.ctx,
        inst=inst,
        simulator=_FixedTerminalSimulator(samples=samples),
        cfg=cfg,
    )

    expected_payoffs = inst.payoff(samples)
    expected_price = p.ctx.df(p.tau) * float(np.mean(expected_payoffs))
    expected_stderr = (
        p.ctx.df(p.tau)
        * float(np.std(expected_payoffs, ddof=1))
        / math.sqrt(len(expected_payoffs))
    )
    assert math.isclose(result.price, expected_price, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.stderr, expected_stderr, rel_tol=0.0, abs_tol=1e-12)
    assert result.effective_n == len(expected_payoffs)


def test_path_instrument_pricer_uses_path_simulator_samples(make_inputs):
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    inst = _AsianCallInstrument(expiry=p.tau, strike=100.0)
    paths = np.array(
        [
            [100.0, 110.0, 120.0],
            [100.0, 100.0, 100.0],
            [100.0, 90.0, 80.0],
        ]
    )
    cfg = MCConfig(n_paths=3)

    result = mc_price_path_instrument_from_ctx(
        ctx=p.ctx,
        inst=inst,
        simulator=_FixedPathSimulator(paths=paths),
        cfg=cfg,
    )

    payoff_values = inst.payoff(paths)
    expected_price = p.ctx.df(p.tau) * float(np.mean(payoff_values))
    expected_stderr = (
        p.ctx.df(p.tau)
        * float(np.std(payoff_values, ddof=1))
        / math.sqrt(len(payoff_values))
    )
    assert math.isclose(result.price, expected_price, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.stderr, expected_stderr, rel_tol=0.0, abs_tol=1e-12)
    assert result.effective_n == len(payoff_values)


def test_path_payoff_pricer_passes_time_grid_and_returns_result(make_inputs):
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )

    result = mc_price_path_payoff_from_ctx(
        ctx=p.ctx,
        payoff=lambda paths, *, times=None: np.full(paths.shape[0], times[-1]),
        sigma=p.sigma,
        tau=p.tau,
        n_steps=8,
        cfg=MCConfig(n_paths=6, antithetic=True, random=RandomConfig(seed=9)),
    )

    assert isinstance(result, MonteCarloResult)
    assert math.isclose(
        result.price, p.ctx.df(p.tau) * p.tau, rel_tol=0.0, abs_tol=1e-12
    )
    assert result.stderr == 0.0
    assert result.effective_n == 3
    assert result.metadata["n_steps"] == 8
