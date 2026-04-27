import math

import numpy as np
import pytest

from option_pricing.instruments import ExerciseStyle, VanillaOption
from option_pricing.models.heston import HestonParams
from option_pricing.monte_carlo import MCConfig, MonteCarloResult, RandomConfig
from option_pricing.pricers.black_scholes import bs_price_call
from option_pricing.pricers.heston import (
    heston_price_call_from_ctx,
    heston_price_from_ctx,
)
from option_pricing.pricers.heston_mc import (
    heston_mc_price,
    heston_mc_price_call,
    heston_mc_price_from_ctx,
    heston_mc_price_instrument,
    heston_mc_price_path_payoff_from_ctx,
    heston_mc_price_put,
    heston_mc_price_with_vanilla_control_from_ctx,
    heston_vanilla_control_from_ctx,
)
from option_pricing.types import OptionType


def _constant_variance_heston_params(sigma: float) -> HestonParams:
    variance = float(sigma) ** 2
    return HestonParams(
        kappa=1.5,
        vbar=variance,
        eta=0.0,
        rho=0.0,
        v=variance,
    )


def _near_constant_variance_heston_params(sigma: float) -> HestonParams:
    variance = float(sigma) ** 2
    return HestonParams(
        kappa=40.0,
        vbar=variance,
        eta=0.05,
        rho=0.0,
        v=variance,
    )


@pytest.mark.parametrize(
    ("kind", "specialized_wrapper"),
    [
        (OptionType.CALL, heston_mc_price_call),
        (OptionType.PUT, heston_mc_price_put),
    ],
)
def test_heston_mc_seed_is_deterministic_across_public_wrappers(
    make_inputs,
    kind: OptionType,
    specialized_wrapper,
) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.02,
        sigma=0.2,
        T=1.0,
        kind=kind,
    )
    params = _constant_variance_heston_params(p.sigma)
    inst = VanillaOption(
        expiry=p.tau,
        strike=p.K,
        kind=p.spec.kind,
        exercise=ExerciseStyle.EUROPEAN,
    )
    cfg = MCConfig(n_paths=8_000, antithetic=True, random=RandomConfig(seed=7))

    result_inputs = heston_mc_price(p, params=params, n_steps=24, cfg=cfg)
    result_ctx = heston_mc_price_from_ctx(
        ctx=p.ctx,
        kind=p.spec.kind,
        strike=p.K,
        params=params,
        tau=p.tau,
        n_steps=24,
        cfg=cfg,
    )
    result_inst = heston_mc_price_instrument(
        inst,
        market=p.ctx,
        params=params,
        n_steps=24,
        cfg=cfg,
    )
    result_specialized = specialized_wrapper(
        p,
        params=params,
        n_steps=24,
        cfg=cfg,
    )

    for lhs, rhs in [
        (result_inputs, result_ctx),
        (result_inputs, result_inst),
        (result_inputs, result_specialized),
    ]:
        assert math.isclose(lhs.price, rhs.price, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(lhs.stderr, rhs.stderr, rel_tol=0.0, abs_tol=1e-12)
        assert lhs.effective_n == rhs.effective_n


def test_heston_mc_is_reproducible_with_same_seed(make_inputs) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)
    cfg = MCConfig(n_paths=20_000, antithetic=True, random=RandomConfig(seed=41))

    result_a = heston_mc_price_call(p, params=params, n_steps=64, cfg=cfg)
    result_b = heston_mc_price_call(p, params=params, n_steps=64, cfg=cfg)

    assert math.isclose(result_a.price, result_b.price, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result_a.stderr, result_b.stderr, rel_tol=0.0, abs_tol=1e-12)
    assert result_a.effective_n == result_b.effective_n
    assert result_a.seed == result_b.seed == 41


def test_heston_mc_antithetic_reports_half_effective_sample_count(make_inputs) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)

    plain = heston_mc_price_call(
        p,
        params=params,
        n_steps=64,
        cfg=MCConfig(n_paths=4_000, antithetic=False, random=RandomConfig(seed=13)),
    )
    antithetic = heston_mc_price_call(
        p,
        params=params,
        n_steps=64,
        cfg=MCConfig(n_paths=4_000, antithetic=True, random=RandomConfig(seed=13)),
    )

    assert plain.effective_n == 4_000
    assert antithetic.effective_n == 2_000
    assert antithetic.stderr > 0.0


def test_heston_mc_matches_heston_fourier_within_mc_error(make_inputs) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)
    cfg = MCConfig(n_paths=40_000, antithetic=True, random=RandomConfig(seed=123))

    result = heston_mc_price_call(p, params=params, n_steps=128, cfg=cfg)
    fourier_price = float(
        heston_price_call_from_ctx(
            strike=p.K,
            ctx=p.ctx,
            tau=p.tau,
            params=params,
        )
    )

    assert isinstance(result, MonteCarloResult)
    assert result.stderr > 0.0
    assert abs(result.price - fourier_price) <= 4.0 * result.stderr


def test_heston_mc_returns_monte_carlo_result_and_tracks_bs_in_near_constant_variance_limit(
    make_inputs,
) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    params = _near_constant_variance_heston_params(p.sigma)

    bs = float(bs_price_call(p))
    cfg = MCConfig(n_paths=80_000, antithetic=True, random=RandomConfig(seed=321))
    result = heston_mc_price_call(p, params=params, n_steps=128, cfg=cfg)

    assert isinstance(result, MonteCarloResult)
    assert result.stderr > 0.0
    assert abs(result.price - bs) <= 4.0 * result.stderr


def test_heston_vanilla_control_uses_undiscounted_heston_mean(make_inputs) -> None:
    p = make_inputs(
        S=100.0,
        K=95.0,
        r=0.02,
        q=0.01,
        sigma=0.2,
        T=1.25,
        kind=OptionType.CALL,
    )
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)

    control = heston_vanilla_control_from_ctx(
        ctx=p.ctx,
        kind=p.spec.kind,
        strike=p.K,
        tau=p.tau,
        params=params,
    )
    analytic_price = float(
        heston_price_from_ctx(
            kind=p.spec.kind,
            strike=p.K,
            tau=p.tau,
            ctx=p.ctx,
            params=params,
        )
    )
    df = float(p.ctx.df(p.tau))

    terminal = np.array([80.0, 95.0, 105.0, 120.0])
    expected_payoff = np.array([0.0, 0.0, 10.0, 25.0])

    assert np.allclose(control.values(terminal), expected_payoff)
    assert math.isclose(control.mean, analytic_price / df, rel_tol=0.0, abs_tol=1e-12)


def test_heston_mc_vanilla_control_reduces_error_and_sets_metadata(make_inputs) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)
    cfg = MCConfig(n_paths=20_000, antithetic=True, random=RandomConfig(seed=123))

    plain = heston_mc_price_from_ctx(
        ctx=p.ctx,
        kind=p.spec.kind,
        strike=p.K,
        params=params,
        tau=p.tau,
        n_steps=64,
        cfg=cfg,
    )
    controlled = heston_mc_price_with_vanilla_control_from_ctx(
        ctx=p.ctx,
        kind=p.spec.kind,
        strike=p.K,
        params=params,
        tau=p.tau,
        n_steps=64,
        cfg=cfg,
        control_kind=OptionType.PUT,
        control_strike=95.0,
    )
    analytic_price = float(
        heston_price_from_ctx(
            kind=p.spec.kind,
            strike=p.K,
            tau=p.tau,
            ctx=p.ctx,
            params=params,
        )
    )

    assert controlled.stderr < plain.stderr
    assert abs(controlled.price - analytic_price) <= abs(plain.price - analytic_price)
    assert controlled.effective_n == plain.effective_n == 10_000
    assert controlled.metadata["variance_reduction"] == {
        "antithetic": True,
        "control": "heston_semi_analytic_vanilla",
        "control_kind": str(OptionType.PUT),
        "control_strike": 95.0,
    }


def test_heston_mc_path_payoff_pricer_passes_time_grid_and_returns_result(
    make_inputs,
) -> None:
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )

    result = heston_mc_price_path_payoff_from_ctx(
        ctx=p.ctx,
        payoff=lambda paths, *, times=None: np.full(paths.shape[0], times[-1]),
        params=HestonParams(kappa=1.0, vbar=0.0, eta=0.0, rho=0.0, v=0.0),
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
    assert result.metadata["scheme"] == "euler_full_truncation"
