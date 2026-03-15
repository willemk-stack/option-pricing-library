from __future__ import annotations

import os


def _mc_paths(default: int) -> int:
    if os.getenv("OP_FAST_EXAMPLES"):
        return min(default, 10_000)
    return default


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
    from option_pricing.config import MCConfig, RandomConfig

    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
    # In PricingInputs, expiry is the absolute expiry T; with t=0 it equals tau numerically.
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)
    p = PricingInputs(spec=spec, market=market, sigma=0.20, t=0.0)

    print("BS:", bs_price(p))
    print("Greeks:", bs_greeks(p))

    cfg_mc = MCConfig(
        n_paths=_mc_paths(200_000), antithetic=True, random=RandomConfig(seed=0)
    )
    price_mc, se = mc_price(p, cfg=cfg_mc)
    print("MC:", price_mc, "(SE=", se, ")")

    print("CRR:", binom_price(p, n_steps=500))
    # [END README_QUICKSTART]


if __name__ == "__main__":
    main()
