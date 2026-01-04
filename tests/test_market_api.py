from __future__ import annotations

import math

from option_pricing import (
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    MarketData,
    OptionSpec,
    OptionType,
    PricingContext,
    PricingInputs,
    binom_price,
    binom_price_from_ctx,
    bs_price,
    bs_price_from_ctx,
    mc_price,
    mc_price_from_ctx,
)
from option_pricing.config import MCConfig, RandomConfig


def test_marketdata_fwd_alias() -> None:
    m = MarketData(spot=100.0, rate=0.03, dividend_yield=0.01)
    assert m.fwd(1.0) == m.forward(1.0)


def test_flat_market_to_context_consistency() -> None:
    m = MarketData(spot=100.0, rate=0.05, dividend_yield=0.02)
    ctx = m.to_context()

    # MarketData uses (t, T); PricingContext uses tau = T - t.
    df = m.df(1.25, t=0.25)
    fwd = m.fwd(1.25, t=0.25)

    tau = 1.0
    assert math.isclose(ctx.df(tau), df, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(ctx.fwd(tau), fwd, rel_tol=0.0, abs_tol=1e-12)


def test_pricers_from_ctx_match_pricinginputs_wrappers() -> None:
    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    p = PricingInputs(spec=spec, market=market, sigma=0.20, t=0.0)

    ctx = PricingContext(
        spot=100.0,
        discount=FlatDiscountCurve(0.05),
        forward=FlatCarryForwardCurve(spot=100.0, r=0.05, q=0.0),
    )

    # Black-Scholes
    px_inputs = bs_price(p)
    px_ctx = bs_price_from_ctx(
        kind=spec.kind, strike=spec.strike, sigma=p.sigma, tau=p.tau, ctx=ctx
    )
    assert math.isclose(px_inputs, px_ctx, rel_tol=0.0, abs_tol=1e-12)

    # Binomial
    n_steps = 400
    px_inputs = binom_price(p, n_steps=n_steps)
    px_ctx = binom_price_from_ctx(
        kind=spec.kind,
        strike=spec.strike,
        sigma=p.sigma,
        tau=p.tau,
        ctx=ctx,
        n_steps=n_steps,
    )
    assert math.isclose(px_inputs, px_ctx, rel_tol=0.0, abs_tol=1e-12)

    # Monte Carlo (use fixed RNG)
    cfg = MCConfig(n_paths=50_000, antithetic=True, random=RandomConfig(seed=123))
    px_inputs, se_inputs = mc_price(p, cfg=cfg)
    px_ctx, se_ctx = mc_price_from_ctx(
        kind=spec.kind, strike=spec.strike, sigma=p.sigma, tau=p.tau, ctx=ctx, cfg=cfg
    )
    assert math.isclose(px_inputs, px_ctx, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(se_inputs, se_ctx, rel_tol=0.0, abs_tol=1e-12)
