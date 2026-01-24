import math

from option_pricing.pricers.black_scholes import bs_price_call
from option_pricing.pricers.tree import (
    binom_price,
    binom_price_call,
    binom_price_put,
)
from option_pricing.types import OptionType


def test_binomial_converges_toward_bs_as_steps_increase(make_inputs):
    """CRR binomial call should approach BS call as steps increase."""
    p = make_inputs(
        S=100.0, K=105.0, r=0.04, q=0.0, sigma=0.22, T=1.0, kind=OptionType.CALL
    )

    bs = float(bs_price_call(p))

    steps = [25, 50, 100, 200, 400]
    errs = [abs(float(binom_price_call(p, n_steps=n)) - bs) for n in steps]

    # not strictly monotone, but high-N should be better than low-N
    assert errs[-1] <= errs[0]
    # keep this loose; tighten once you know typical accuracy
    assert errs[-1] <= 2e-2


def test_binomial_put_call_parity_approximately(make_inputs):
    """Binomial should satisfy parity approximately for sufficiently many steps."""
    p_call = make_inputs(
        S=100.0, K=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0, kind=OptionType.CALL
    )
    p_put = make_inputs(
        S=p_call.S,
        K=p_call.K,
        r=p_call.market.rate,
        q=p_call.market.dividend_yield,
        sigma=p_call.sigma,
        T=p_call.T,
        t=p_call.t,
        kind=OptionType.PUT,
    )
    n_steps = 400

    C = float(binom_price_call(p_call, n_steps=n_steps))
    P = float(binom_price_put(p_put, n_steps=n_steps))

    tau = p_call.tau
    df_r = math.exp(-p_call.market.rate * tau)
    df_q = math.exp(-p_call.market.dividend_yield * tau)
    rhs = p_call.S * df_q - p_call.K * df_r

    assert abs((C - P) - rhs) <= 5e-3


def test_binomial_generic_dispatch_uses_kind(make_inputs):
    """binom_price should dispatch on p.spec.kind consistently."""
    p_call = make_inputs(
        S=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0, kind=OptionType.CALL
    )
    p_put = make_inputs(
        S=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0, kind=OptionType.PUT
    )

    n_steps = 200

    c1 = float(binom_price(p_call, n_steps=n_steps))
    c2 = float(binom_price_call(p_call, n_steps=n_steps))
    p1 = float(binom_price(p_put, n_steps=n_steps))
    p2 = float(binom_price_put(p_put, n_steps=n_steps))

    assert abs(c1 - c2) <= 1e-12
    assert abs(p1 - p2) <= 1e-12
