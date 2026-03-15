from __future__ import annotations

import math

from option_pricing import (
    ExerciseStyle,
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    MarketData,
    OptionSpec,
    OptionType,
    PricingContext,
    PricingInputs,
    VanillaOption,
    binom_price,
    binom_price_from_ctx,
    binom_price_instrument,
    bs_price,
    bs_price_from_ctx,
    bs_price_instrument,
    mc_price,
    mc_price_from_ctx,
    mc_price_instrument,
)
from option_pricing.config import MCConfig, RandomConfig


def _build_inputs() -> PricingInputs:
    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.02)
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    return PricingInputs(spec=spec, market=market, sigma=0.2, t=0.0)


def test_put_call_parity_contract() -> None:
    market = MarketData(spot=100.0, rate=0.04, dividend_yield=0.01)
    call = PricingInputs(
        spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.5),
        market=market,
        sigma=0.25,
        t=0.0,
    )
    put = PricingInputs(
        spec=OptionSpec(kind=OptionType.PUT, strike=100.0, expiry=1.5),
        market=market,
        sigma=0.25,
        t=0.0,
    )

    tau = call.tau
    df_r = math.exp(-market.rate * tau)
    df_q = math.exp(-market.dividend_yield * tau)

    price_call = float(bs_price(call))
    price_put = float(bs_price(put))

    parity = call.market.spot * df_q - call.spec.strike * df_r
    assert abs((price_call - price_put) - parity) < 1e-10


def test_cross_api_consistency_bs_pricing() -> None:
    p = _build_inputs()
    ctx = p.market.to_context()
    inst = VanillaOption(
        expiry=p.tau,
        strike=p.spec.strike,
        kind=p.spec.kind,
        exercise=ExerciseStyle.EUROPEAN,
    )

    price_inputs = float(bs_price(p))
    price_ctx = float(
        bs_price_from_ctx(
            kind=p.spec.kind,
            strike=p.spec.strike,
            sigma=p.sigma,
            tau=p.tau,
            ctx=ctx,
        )
    )
    price_inst_market = float(bs_price_instrument(inst, market=p.market, sigma=p.sigma))
    price_inst_ctx = float(bs_price_instrument(inst, market=ctx, sigma=p.sigma))

    assert math.isclose(price_inputs, price_ctx, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(price_inputs, price_inst_market, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(price_inputs, price_inst_ctx, rel_tol=0.0, abs_tol=1e-12)


def test_pricing_inputs_uses_absolute_expiry_time() -> None:
    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.02)
    p = PricingInputs(
        spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.5),
        market=market,
        sigma=0.2,
        t=0.5,
    )

    assert math.isclose(p.T, 1.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(p.tau, 1.0, rel_tol=0.0, abs_tol=1e-12)

    price_inputs = float(bs_price(p))
    price_ctx = float(
        bs_price_from_ctx(
            kind=p.spec.kind,
            strike=p.spec.strike,
            sigma=p.sigma,
            tau=1.0,
            ctx=p.market.to_context(),
        )
    )

    assert math.isclose(price_inputs, price_ctx, rel_tol=0.0, abs_tol=1e-12)


def test_golden_value_bs_call_pricinginputs() -> None:
    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    p = PricingInputs(spec=spec, market=market, sigma=0.2, t=0.0)

    expected = 10.450583572185565
    price = float(bs_price(p))
    assert math.isclose(price, expected, rel_tol=0.0, abs_tol=1e-8)


def test_golden_value_bs_call_instrument_and_ctx() -> None:
    expected = 10.450583572185565

    inst = VanillaOption(
        expiry=1.0,
        strike=100.0,
        kind=OptionType.CALL,
        exercise=ExerciseStyle.EUROPEAN,
    )
    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)

    price_inst = float(bs_price_instrument(inst, market=market, sigma=0.2))
    assert math.isclose(price_inst, expected, rel_tol=0.0, abs_tol=1e-8)

    ctx = PricingContext(
        spot=100.0,
        discount=FlatDiscountCurve(0.05),
        forward=FlatCarryForwardCurve(spot=100.0, r=0.05, q=0.0),
    )
    price_ctx = float(
        bs_price_from_ctx(
            kind=OptionType.CALL, strike=100.0, sigma=0.2, tau=1.0, ctx=ctx
        )
    )
    assert math.isclose(price_ctx, expected, rel_tol=0.0, abs_tol=1e-8)


def test_cross_api_consistency_binomial_pricing() -> None:
    p = _build_inputs()
    ctx = p.market.to_context()
    inst = VanillaOption(
        expiry=p.tau,
        strike=p.spec.strike,
        kind=p.spec.kind,
        exercise=ExerciseStyle.EUROPEAN,
    )
    n_steps = 100

    price_inputs = float(binom_price(p, n_steps=n_steps))
    price_ctx = float(
        binom_price_from_ctx(
            kind=p.spec.kind,
            strike=p.spec.strike,
            sigma=p.sigma,
            tau=p.tau,
            ctx=ctx,
            n_steps=n_steps,
        )
    )
    price_inst_market = float(
        binom_price_instrument(inst, market=p.market, sigma=p.sigma, n_steps=n_steps)
    )
    price_inst_ctx = float(
        binom_price_instrument(inst, market=ctx, sigma=p.sigma, n_steps=n_steps)
    )

    assert math.isfinite(price_inputs)
    assert math.isclose(price_inputs, price_ctx, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(price_inputs, price_inst_market, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(price_inputs, price_inst_ctx, rel_tol=0.0, abs_tol=1e-10)


def test_cross_api_consistency_monte_carlo_pricing() -> None:
    p = _build_inputs()
    ctx = p.market.to_context()
    inst = VanillaOption(
        expiry=p.tau,
        strike=p.spec.strike,
        kind=p.spec.kind,
        exercise=ExerciseStyle.EUROPEAN,
    )
    cfg = MCConfig(n_paths=8_000, antithetic=True, random=RandomConfig(seed=7))

    price_inputs, se_inputs = mc_price(p, cfg=cfg)
    price_ctx, se_ctx = mc_price_from_ctx(
        kind=p.spec.kind,
        strike=p.spec.strike,
        sigma=p.sigma,
        tau=p.tau,
        ctx=ctx,
        cfg=cfg,
    )
    price_inst_market, se_inst_market = mc_price_instrument(
        inst, market=p.market, sigma=p.sigma, cfg=cfg
    )
    price_inst_ctx, se_inst_ctx = mc_price_instrument(
        inst, market=ctx, sigma=p.sigma, cfg=cfg
    )

    assert math.isfinite(price_inputs)
    assert se_inputs > 0.0
    assert math.isclose(price_inputs, price_ctx, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(price_inputs, price_inst_market, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(price_inputs, price_inst_ctx, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(se_inputs, se_ctx, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(se_inputs, se_inst_market, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(se_inputs, se_inst_ctx, rel_tol=0.0, abs_tol=1e-12)
