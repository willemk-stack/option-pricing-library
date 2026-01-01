from __future__ import annotations

import math

import pytest

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price
from option_pricing.exceptions import InvalidOptionPriceError
from option_pricing.numerics.root_finding import bracketed_newton
from option_pricing.vol.implied_vol import implied_vol_bs_result


def price_for(
    *,
    kind: OptionType,
    S: float,
    K: float,
    r: float,
    q: float,
    tau: float,
    sigma: float,
) -> float:
    p = PricingInputs(
        spec=OptionSpec(kind=kind, strike=K, expiry=tau),
        market=MarketData(spot=S, rate=r, dividend_yield=q),
        sigma=sigma,
        t=0.0,
    )
    return float(bs_price(p))


def iv_for(
    *,
    kind: OptionType,
    S: float,
    K: float,
    r: float,
    q: float,
    tau: float,
    mkt_price: float,
    sigma0: float = 0.30,
) -> float:
    spec = OptionSpec(kind=kind, strike=K, expiry=tau)
    market = MarketData(spot=S, rate=r, dividend_yield=q)

    res = implied_vol_bs_result(
        mkt_price=float(mkt_price),
        spec=spec,
        market=market,
        root_method=bracketed_newton,
        t=0.0,
        sigma0=float(sigma0),
        sigma_lo=1e-12,
        sigma_hi=8.0,
        tol_f=1e-12,
        tol_x=1e-12,
        max_iter=200,
    )
    assert bool(res.root_result.converged) is True, getattr(
        res.root_result, "message", ""
    )
    return float(res.vol)


@pytest.mark.parametrize("tau", [1e-4, 1 / 3650], ids=lambda t: f"tau={t:g}")
def test_iv_short_expiry_recovers_sigma(tau: float):
    S, r, q = 100.0, 0.01, 0.0
    K = 100.0
    sigma_true = 0.25

    mkt = price_for(kind=OptionType.CALL, S=S, K=K, r=r, q=q, tau=tau, sigma=sigma_true)
    iv = iv_for(
        kind=OptionType.CALL, S=S, K=K, r=r, q=q, tau=tau, mkt_price=mkt, sigma0=0.30
    )

    # short-expiry can be ill-conditioned; keep tolerance slightly looser
    assert iv == pytest.approx(sigma_true, abs=5e-3)


@pytest.mark.parametrize("K_mult", [0.3, 3.0], ids=lambda m: f"K={m}S")
def test_iv_deep_itm_otm_recovers_sigma(K_mult: float):
    S, r, q, tau = 100.0, 0.00, 0.00, 1.0
    K = K_mult * S
    sigma_true = 0.40

    mkt = price_for(kind=OptionType.CALL, S=S, K=K, r=r, q=q, tau=tau, sigma=sigma_true)
    iv = iv_for(
        kind=OptionType.CALL, S=S, K=K, r=r, q=q, tau=tau, mkt_price=mkt, sigma0=0.50
    )

    assert iv == pytest.approx(sigma_true, abs=1e-3)


def test_iv_handles_negative_rates():
    S, r, q, tau = 100.0, -0.02, 0.00, 0.75
    K = 100.0
    sigma_true = 0.30

    mkt = price_for(kind=OptionType.PUT, S=S, K=K, r=r, q=q, tau=tau, sigma=sigma_true)
    iv = iv_for(
        kind=OptionType.PUT, S=S, K=K, r=r, q=q, tau=tau, mkt_price=mkt, sigma0=0.25
    )

    assert iv == pytest.approx(sigma_true, abs=1e-3)


@pytest.mark.parametrize(
    "kind,S,K,r,q,tau",
    [
        (
            OptionType.CALL,
            100.0,
            50.0,
            0.00,
            0.00,
            1.0,
        ),  # call has positive lower bound
        (OptionType.PUT, 100.0, 150.0, 0.00, 0.00, 1.0),  # put has positive lower bound
    ],
)
def test_iv_raises_when_price_below_noarb_lower_bound(kind, S, K, r, q, tau):
    df = math.exp(-r * tau)
    Fp = S * math.exp(-q * tau)

    if kind == OptionType.CALL:
        lb = max(Fp - K * df, 0.0)
    else:
        lb = max(K * df - Fp, 0.0)

    spec = OptionSpec(kind=kind, strike=K, expiry=tau)
    market = MarketData(spot=S, rate=r, dividend_yield=q)

    with pytest.raises(InvalidOptionPriceError):
        implied_vol_bs_result(
            mkt_price=float(lb - 1e-6),
            spec=spec,
            market=market,
            root_method=bracketed_newton,
            t=0.0,
            sigma0=0.20,
        )


@pytest.mark.parametrize(
    "kind,S,K,r,q,tau",
    [
        (OptionType.CALL, 100.0, 100.0, 0.00, 0.00, 1.0),
        (OptionType.PUT, 100.0, 100.0, 0.00, 0.00, 1.0),
    ],
)
def test_iv_raises_when_price_above_noarb_upper_bound(kind, S, K, r, q, tau):
    df = math.exp(-r * tau)
    Fp = S * math.exp(-q * tau)

    if kind == OptionType.CALL:
        ub = Fp
    else:
        ub = K * df

    spec = OptionSpec(kind=kind, strike=K, expiry=tau)
    market = MarketData(spot=S, rate=r, dividend_yield=q)

    with pytest.raises(InvalidOptionPriceError):
        implied_vol_bs_result(
            mkt_price=float(ub + 1e-6),
            spec=spec,
            market=market,
            root_method=bracketed_newton,
            t=0.0,
            sigma0=0.20,
        )


def test_iv_raises_on_negative_price():
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    market = MarketData(spot=100.0, rate=0.0, dividend_yield=0.0)

    with pytest.raises(InvalidOptionPriceError):
        implied_vol_bs_result(
            mkt_price=-1e-6,
            spec=spec,
            market=market,
            root_method=bracketed_newton,
            t=0.0,
            sigma0=0.20,
        )


def test_iv_tiny_time_value_just_above_intrinsic_atm_converges_to_tiny_vol():
    # ATM call: intrinsic is 0, and tiny price implies tiny vol.
    kind = OptionType.CALL
    S, K = 100.0, 100.0
    r, q = 0.0, 0.0
    tau = 1.0

    intrinsic = 0.0
    mkt_price = intrinsic + 1e-6  # tiny time value

    spec = OptionSpec(kind=kind, strike=K, expiry=tau)
    market = MarketData(spot=S, rate=r, dividend_yield=q)

    res = implied_vol_bs_result(
        mkt_price=float(mkt_price),
        spec=spec,
        market=market,
        root_method=bracketed_newton,
        t=0.0,
        sigma0=None,  # exercise heuristic seed
        sigma_lo=1e-12,
        sigma_hi=5.0,
        tol_f=1e-12,
        tol_x=1e-12,
        max_iter=200,
    )

    assert bool(res.root_result.converged) is True, getattr(
        res.root_result, "message", ""
    )

    # ATM: this should be extremely small
    assert 0.0 <= float(res.vol) < 1e-4

    # Optional: reprice and confirm inversion accuracy
    p = PricingInputs(spec=spec, market=market, sigma=float(res.vol), t=0.0)
    assert float(bs_price(p)) == pytest.approx(float(mkt_price), abs=1e-10)
