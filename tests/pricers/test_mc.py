from option_pricing.config import MCConfig, RandomConfig
from option_pricing.pricers.black_scholes import bs_price_call
from option_pricing.pricers.mc import mc_price, mc_price_call
from option_pricing.types import OptionType


def test_mc_matches_bs_within_a_few_standard_errors(make_inputs):
    """MC price should agree with BS within a few reported SEs."""
    p = make_inputs(
        S=100.0,
        K=110.0,
        r=0.03,
        q=0.0,
        sigma=0.25,
        T=1.0,
        kind=OptionType.CALL,
    )

    bs = float(bs_price_call(p))
    cfg = MCConfig(n_paths=40_000, random=RandomConfig(seed=7))
    mc, se = mc_price(p, cfg=cfg)

    assert se > 0.0
    assert abs(mc - bs) <= (3.0 * se + 2e-3)


def test_mc_standard_error_scales_like_inverse_sqrt_n(make_inputs):
    """SE should scale ~ 1/sqrt(N) (approx; allow generous tolerance)."""
    p = make_inputs(
        S=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0, kind=OptionType.CALL
    )

    N1 = 5_000
    N2 = 20_000  # 4x more paths => SE should be ~ half

    _, se1 = mc_price_call(p, cfg=MCConfig(n_paths=N1, random=RandomConfig(seed=11)))
    _, se2 = mc_price_call(p, cfg=MCConfig(n_paths=N2, random=RandomConfig(seed=12)))

    ratio = se1 / se2
    # expected ratio ~ sqrt(N2/N1) = 2
    assert 1.4 <= ratio <= 2.8
