from option_pricing.instruments.vanilla import VanillaOption
from option_pricing.pricers.black_scholes import (
    bs_call_greeks,
    bs_greeks_instrument,
    bs_price_call,
)
from option_pricing.pricers.finite_diff import (
    finite_diff_greeks,
    finite_diff_greeks_instrument,
)
from option_pricing.types import MarketData, OptionType


def test_finite_diff_greeks_matches_analytic_mid_time(make_inputs):
    # t > 0 hits the "central difference" theta branch
    p = make_inputs(S=100.0, K=100.0, r=0.02, q=0.01, sigma=0.25, T=1.0, t=0.4)
    fd = finite_diff_greeks(p, price_fn=bs_price_call, h_x=0.05, h_sigma=1e-3, h_t=1e-3)
    an = bs_call_greeks(p)

    assert abs(fd["delta"] - an["delta"]) < 5e-4
    assert abs(fd["gamma"] - an["gamma"]) < 5e-4
    assert abs(fd["vega"] - an["vega"]) < 5e-3
    assert abs(fd["theta"] - an["theta"]) < 5e-3


def test_finite_diff_theta_uses_forward_difference_at_t0(make_inputs):
    # t=0 hits the "forward difference" theta branch
    p = make_inputs(S=100.0, K=100.0, r=0.02, q=0.0, sigma=0.2, T=1.0, t=0.0)
    fd = finite_diff_greeks(p, price_fn=bs_price_call, h_x=0.05, h_sigma=1e-3, h_t=1e-3)
    an = bs_call_greeks(p)

    assert abs(fd["theta"] - an["theta"]) < 1e-2  # forward diff is less accurate


def test_finite_diff_greeks_instrument_theta_tau_matches_negative_theta():
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    inst = VanillaOption(expiry=1.0, strike=100.0, kind=OptionType.CALL)
    sigma = 0.25

    fd = finite_diff_greeks_instrument(
        inst, market=market, sigma=sigma, h_x=0.05, h_sigma=1e-3, h_tau=1e-3
    )
    an = bs_greeks_instrument(inst, market=market, sigma=sigma)

    # theta_tau = dV/dtau, while analytic theta is dV/dt holding expiry fixed => theta_tau ≈ -theta
    assert abs(fd["delta"] - an["delta"]) < 5e-4
    assert abs(fd["gamma"] - an["gamma"]) < 5e-4
    assert abs(fd["vega"] - an["vega"]) < 5e-3
    assert abs(fd["theta_tau"] + an["theta"]) < 5e-3
