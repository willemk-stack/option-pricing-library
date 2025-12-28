from __future__ import annotations


def main() -> None:
    # [START README_QUICKSTART]
    from option_pricing import (
        MarketData,
        OptionSpec,
        OptionType,
        PricingInputs,
        binom_price,
        bs_greeks,
        bs_price,
        mc_price,
    )

    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    p = PricingInputs(spec=spec, market=market, sigma=0.20, t=0.0)

    print("BS:", bs_price(p))
    print("Greeks:", bs_greeks(p))

    price_mc, se = mc_price(p, n_paths=200_000, antithetic=True, seed=0)
    print("MC:", price_mc, "(SE=", se, ")")

    print("CRR:", binom_price(p, n_steps=500))
    # [END README_QUICKSTART]


if __name__ == "__main__":
    main()
