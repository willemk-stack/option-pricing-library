import math
from statistics import mean

import numpy as np
import pytest

from option_pricing.instruments import ExerciseStyle, VanillaOption
from option_pricing.models.heston import HestonParams
from option_pricing.models.heston.simulation import simulate_heston_terminal
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
from option_pricing.types import MarketData, OptionType


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


_QE_FOURIER_VALIDATION_CASES = (
    pytest.param(
        {
            "S": 100.0,
            "K": 100.0,
            "r": 0.02,
            "q": 0.0,
            "sigma": 0.2,
            "T": 1.0,
            "kind": OptionType.CALL,
        },
        HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05),
        128,
        24_000,
        101,
        id="atm_1y_negative_rho",
    ),
    pytest.param(
        {
            "S": 100.0,
            "K": 110.0,
            "r": 0.02,
            "q": 0.0,
            "sigma": 0.2,
            "T": 0.25,
            "kind": OptionType.CALL,
        },
        HestonParams(kappa=2.0, vbar=0.04, eta=0.65, rho=-0.75, v=0.05),
        96,
        24_000,
        102,
        id="otm_3m_negative_rho",
    ),
    pytest.param(
        {
            "S": 100.0,
            "K": 90.0,
            "r": 0.02,
            "q": 0.0,
            "sigma": 0.2,
            "T": 2.0,
            "kind": OptionType.CALL,
        },
        HestonParams(kappa=1.25, vbar=0.05, eta=0.70, rho=-0.55, v=0.04),
        192,
        24_000,
        103,
        id="itm_2y_negative_rho",
    ),
    pytest.param(
        {
            "S": 100.0,
            "K": 95.0,
            "r": 0.01,
            "q": 0.0,
            "sigma": 0.2,
            "T": 0.10,
            "kind": OptionType.CALL,
        },
        HestonParams(kappa=1.5, vbar=0.04, eta=1.20, rho=-0.92, v=0.04),
        96,
        24_000,
        104,
        id="short_high_xi_negative_rho",
    ),
    pytest.param(
        {
            "S": 100.0,
            "K": 85.0,
            "r": 0.015,
            "q": 0.0,
            "sigma": 0.2,
            "T": 0.75,
            "kind": OptionType.CALL,
        },
        HestonParams(kappa=1.8, vbar=0.035, eta=0.80, rho=0.45, v=0.03),
        128,
        24_000,
        105,
        id="itm_positive_rho",
    ),
)


def _qe_call_and_fourier_price(
    make_inputs,
    input_kwargs,
    params: HestonParams,
    *,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> tuple[MonteCarloResult, float]:
    p = make_inputs(**input_kwargs)
    result = heston_mc_price_call(
        p,
        params=params,
        n_steps=n_steps,
        scheme="quadratic_exponential",
        cfg=MCConfig(
            n_paths=n_paths,
            antithetic=True,
            random=RandomConfig(seed=seed),
        ),
    )
    fourier_price = float(
        heston_price_call_from_ctx(
            strike=p.K,
            ctx=p.ctx,
            tau=p.tau,
            params=params,
        )
    )
    return result, fourier_price


def _assert_qe_matches_fourier(
    make_inputs,
    input_kwargs,
    params: HestonParams,
    *,
    n_steps: int,
    n_paths: int,
    seed: int,
    stderr_multiple: float = 5.0,
) -> MonteCarloResult:
    result, fourier_price = _qe_call_and_fourier_price(
        make_inputs,
        input_kwargs,
        params,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.stderr > 0.0
    assert abs(result.price - fourier_price) <= stderr_multiple * result.stderr

    return result


def _mean_qe_call_price(
    make_inputs,
    input_kwargs,
    params: HestonParams,
    *,
    n_steps: int,
    n_paths: int,
    seeds: tuple[int, ...],
) -> float:
    prices = []
    for seed in seeds:
        result, _ = _qe_call_and_fourier_price(
            make_inputs,
            input_kwargs,
            params,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=seed,
        )
        prices.append(result.price)
    return float(mean(prices))


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


def test_heston_mc_public_wrapper_defaults_to_quadratic_exponential(
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
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)
    cfg = MCConfig(n_paths=20_000, antithetic=True, random=RandomConfig(seed=41))

    default_result = heston_mc_price_call(p, params=params, n_steps=64, cfg=cfg)
    explicit_qe_result = heston_mc_price_call(
        p,
        params=params,
        n_steps=64,
        scheme="quadratic_exponential",
        cfg=cfg,
    )

    assert math.isclose(
        default_result.price,
        explicit_qe_result.price,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        default_result.stderr,
        explicit_qe_result.stderr,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert default_result.effective_n == explicit_qe_result.effective_n


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

    result = heston_mc_price_call(
        p,
        params=params,
        n_steps=128,
        scheme="quadratic_exponential",
        cfg=cfg,
    )
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


@pytest.mark.parametrize(
    ("input_kwargs", "params", "n_steps", "n_paths", "seed"),
    _QE_FOURIER_VALIDATION_CASES,
)
def test_qe_vanilla_matches_fourier_across_grid(
    make_inputs,
    input_kwargs,
    params: HestonParams,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> None:
    _assert_qe_matches_fourier(
        make_inputs,
        input_kwargs,
        params,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )


@pytest.mark.slow
def test_qe_bias_decreases_with_timestep(make_inputs) -> None:
    input_kwargs = {
        "S": 100.0,
        "K": 100.0,
        "r": 0.01,
        "q": 0.0,
        "sigma": 0.2,
        "T": 2.0,
        "kind": OptionType.CALL,
    }
    params = HestonParams(kappa=1.0, vbar=0.04, eta=1.25, rho=-0.90, v=0.08)
    seeds = (30, 31, 32, 33, 34, 35)

    p = make_inputs(**input_kwargs)
    fourier_price = float(
        heston_price_call_from_ctx(
            strike=p.K,
            ctx=p.ctx,
            tau=p.tau,
            params=params,
        )
    )
    coarse_bias = abs(
        _mean_qe_call_price(
            make_inputs,
            input_kwargs,
            params,
            n_steps=4,
            n_paths=20_000,
            seeds=seeds,
        )
        - fourier_price
    )
    fine_bias = abs(
        _mean_qe_call_price(
            make_inputs,
            input_kwargs,
            params,
            n_steps=32,
            n_paths=20_000,
            seeds=seeds,
        )
        - fourier_price
    )

    assert fine_bias < coarse_bias


def test_qe_preserves_discounted_forward_martingale() -> None:
    tau = 1.5
    market = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
    ctx = market.to_context()
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)

    terminal = simulate_heston_terminal(
        ctx=ctx,
        tau=tau,
        params=params,
        n_steps=128,
        cfg=MCConfig(
            n_paths=60_000,
            antithetic=True,
            random=RandomConfig(seed=1234),
        ),
        scheme="quadratic_exponential",
    )

    discounted_mean = float(ctx.df(tau) * terminal.mean())
    discounted_stderr = float(
        ctx.df(tau) * terminal.std(ddof=1) / math.sqrt(terminal.size)
    )
    target = market.spot * math.exp(-market.dividend_yield * tau)

    assert discounted_stderr > 0.0
    assert abs(discounted_mean - target) <= 4.0 * discounted_stderr


def test_qe_handles_high_xi_negative_rho_short_maturity(make_inputs) -> None:
    input_kwargs = {
        "S": 100.0,
        "K": 95.0,
        "r": 0.01,
        "q": 0.0,
        "sigma": 0.2,
        "T": 0.10,
        "kind": OptionType.CALL,
    }
    params = HestonParams(kappa=1.5, vbar=0.04, eta=1.20, rho=-0.92, v=0.04)

    result = _assert_qe_matches_fourier(
        make_inputs,
        input_kwargs,
        params,
        n_steps=96,
        n_paths=24_000,
        seed=1_204,
    )

    assert result.price > 0.0


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
    assert result.metadata["scheme"] == "quadratic_exponential"
