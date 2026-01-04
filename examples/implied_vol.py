from __future__ import annotations


def main() -> None:
    # [START README_IMPLIED_VOL]
    from option_pricing import (
        ImpliedVolConfig,
        MarketData,
        OptionSpec,
        OptionType,
        RootMethod,
        implied_vol_bs_result,
    )

    market = MarketData(spot=100.0, rate=0.05, dividend_yield=0.0)
    spec = OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0)

    cfg = ImpliedVolConfig(
        root_method=RootMethod.BRACKETED_NEWTON, sigma_lo=1e-8, sigma_hi=5.0
    )

    res = implied_vol_bs_result(mkt_price=10.0, spec=spec, market=market, cfg=cfg)

    rr = res.root_result
    print(f"IV: {res.vol:.6f}")
    print(f"Converged: {rr.converged}  iters={rr.iterations}  method={rr.method}")
    print(f"f(root)={rr.f_at_root:.3e}  bracket={rr.bracket}  bounds={res.bounds}")
    # [END README_IMPLIED_VOL]


if __name__ == "__main__":
    main()
