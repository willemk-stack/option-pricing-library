from __future__ import annotations


def main() -> None:
    # [START README_CURVES_FIRST]
    from option_pricing import (
        FlatCarryForwardCurve,
        FlatDiscountCurve,
        OptionType,
        PricingContext,
        binom_price_from_ctx,
        bs_greeks_from_ctx,
        bs_price_from_ctx,
        mc_price_from_ctx,
    )
    from option_pricing.config import MCConfig, RandomConfig

    spot = 100.0
    r = 0.05
    q = 0.00
    sigma = 0.20
    tau = 1.0
    K = 100.0

    discount = FlatDiscountCurve(r)
    forward = FlatCarryForwardCurve(spot=spot, r=r, q=q)
    ctx = PricingContext(spot=spot, discount=discount, forward=forward)

    print(
        "BS:",
        bs_price_from_ctx(
            kind=OptionType.CALL, strike=K, sigma=sigma, tau=tau, ctx=ctx
        ),
    )
    print(
        "Greeks:",
        bs_greeks_from_ctx(
            kind=OptionType.CALL, strike=K, sigma=sigma, tau=tau, ctx=ctx
        ),
    )

    cfg_mc = MCConfig(n_paths=200_000, antithetic=True, random=RandomConfig(seed=0))
    price_mc, se = mc_price_from_ctx(
        kind=OptionType.CALL, strike=K, sigma=sigma, tau=tau, ctx=ctx, cfg=cfg_mc
    )
    print("MC:", price_mc, "(SE=", se, ")")

    print(
        "CRR:",
        binom_price_from_ctx(
            kind=OptionType.CALL, strike=K, sigma=sigma, tau=tau, ctx=ctx, n_steps=500
        ),
    )
    # [END README_CURVES_FIRST]


if __name__ == "__main__":
    main()
