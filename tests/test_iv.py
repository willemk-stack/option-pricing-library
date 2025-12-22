from dataclasses import replace

import pytest

from option_pricing import MarketData, OptionSpec, OptionType, PricingInputs, bs_price
from option_pricing.numerics.root_finding import bracketed_newton
from option_pricing.vol.implied_vol import IV_solver

# One place to define “golden” cases.
# Important: per-case tolerances can vary because published prices are often rounded.
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
    # Hull-style: these are only 2 decimals in your script, so price_tol needs to be looser
    # OR replace mkt_price with a more precise value.
    dict(
        name="Hull-style call",
        kind=OptionType.CALL,
        S=42.0,
        K=40.0,
        r=0.10,
        q=0.0,
        tau=0.5,
        sigma_true=0.20,
        mkt_price=4.76,
        price_tol=2e-3,  # <-- loosened due to rounding
        iv_tol=2e-3,
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
        mkt_price=0.81,
        price_tol=2e-3,  # <-- loosened due to rounding
        iv_tol=2e-3,
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
    # price at sigma_true
    p_guess = make_inputs(case, sigma_guess=0.30)
    p_true = replace(p_guess, sigma=float(case["sigma_true"]))

    model_price = bs_price(p_true)

    assert model_price == pytest.approx(
        float(case["mkt_price"]),
        abs=float(case["price_tol"]),
    )


@pytest.mark.parametrize("case", BENCHMARKS, ids=lambda c: c["name"])
def test_implied_vol_recovers_sigma_true(case):
    # recover IV from published price
    # choose a guess that isn't always equal to sigma_true
    sigma_guess = 0.30 if abs(case["sigma_true"] - 0.30) > 1e-12 else 0.20
    p_guess = make_inputs(case, sigma_guess=sigma_guess)

    iv = IV_solver(
        p_guess,
        float(case["mkt_price"]),
        root_method=bracketed_newton,
        sigma0=p_guess.sigma,
        sigma_lo=1e-8,
        sigma_hi=5.0,
    )

    assert iv == pytest.approx(
        float(case["sigma_true"]),
        abs=float(case["iv_tol"]),
    )
