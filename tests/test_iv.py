from __future__ import annotations

from dataclasses import replace

import pytest

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price
from option_pricing.numerics.root_finding import bracketed_newton
from option_pricing.vol.implied_vol import implied_vol_bs_result

BENCHMARKS = [
    dict(
        name="ATM no-div call",
        kind=OptionType.CALL,
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        tau=1.0,
        sigma_true=0.20,
        mkt_price=10.4506,
        price_tol=5e-4,
        iv_tol=5e-4,
    ),
    dict(
        name="ATM no-div put",
        kind=OptionType.PUT,
        S=100.0,
        K=100.0,
        r=0.05,
        q=0.0,
        tau=1.0,
        sigma_true=0.20,
        mkt_price=5.57352,
        price_tol=5e-4,
        iv_tol=5e-4,
    ),
    dict(
        name="ITM 3M call",
        kind=OptionType.CALL,
        S=100.0,
        K=95.0,
        r=0.01,
        q=0.0,
        tau=3 / 12,
        sigma_true=0.50,
        mkt_price=12.5279,
        price_tol=5e-4,
        iv_tol=5e-4,
    ),
    dict(
        name="Hull-style call",
        kind=OptionType.CALL,
        S=42.0,
        K=40.0,
        r=0.10,
        q=0.0,
        tau=0.5,
        sigma_true=0.20,
        mkt_price=4.7594223929,
        price_tol=5e-4,
        iv_tol=5e-4,
    ),
    dict(
        name="Hull-style put",
        kind=OptionType.PUT,
        S=42.0,
        K=40.0,
        r=0.10,
        q=0.0,
        tau=0.5,
        sigma_true=0.20,
        mkt_price=0.8085993729,
        price_tol=5e-4,
        iv_tol=5e-4,
    ),
]


def make_inputs(case: dict, sigma_guess: float) -> PricingInputs:
    return PricingInputs(
        spec=OptionSpec(kind=case["kind"], strike=case["K"], expiry=case["tau"]),
        market=MarketData(spot=case["S"], rate=case["r"], dividend_yield=case["q"]),
        sigma=sigma_guess,
        t=0.0,
    )


@pytest.mark.parametrize("case", BENCHMARKS, ids=lambda c: c["name"])
def test_bs_price_matches_published(case):
    p_guess = make_inputs(case, sigma_guess=0.30)
    p_true = replace(p_guess, sigma=float(case["sigma_true"]))

    model_price = bs_price(p_true)

    assert model_price == pytest.approx(
        float(case["mkt_price"]),
        abs=float(case["price_tol"]),
    )


@pytest.mark.parametrize("case", BENCHMARKS, ids=lambda c: c["name"])
def test_implied_vol_recovers_sigma_true(case):
    sigma_guess = 0.30 if abs(case["sigma_true"] - 0.30) > 1e-12 else 0.20
    p_guess = make_inputs(case, sigma_guess=sigma_guess)

    ivres = implied_vol_bs_result(
        mkt_price=float(case["mkt_price"]),
        spec=p_guess.spec,
        market=p_guess.market,
        root_method=bracketed_newton,
        t=float(p_guess.t),
        sigma0=float(p_guess.sigma),
        sigma_lo=1e-8,
        sigma_hi=5.0,
        tol_f=1e-10,
        tol_x=1e-10,
    )

    assert bool(ivres.root_result.converged) is True
    assert int(ivres.root_result.iterations) >= 0

    assert ivres.vol == pytest.approx(
        float(case["sigma_true"]),
        abs=float(case["iv_tol"]),
    )
