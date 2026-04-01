from option_pricing.market.parity import put_call_parity_residual
from option_pricing.pricers.black_scholes import bs_price_call, bs_price_put


def test_put_call_parity_residual_near_zero(make_inputs):
    p = make_inputs(S=100.0, K=105.0, r=0.03, q=0.01, sigma=0.2, T=1.0, t=0.0)
    call = bs_price_call(p)
    put = bs_price_put(p)

    resid = put_call_parity_residual(call=call, put=put, p=p)
    assert abs(resid) < 1e-10
