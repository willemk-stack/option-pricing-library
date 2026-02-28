from __future__ import annotations

import pytest

from option_pricing.numerics.pde.ic_remedies import ICRemedy
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from option_pricing.pricers.pde_pricer import (
    bs_digital_price_pde,
    bs_price_pde_european,
)
from option_pricing.types import (
    DigitalSpec,
    MarketData,
    OptionSpec,
    OptionType,
    PricingInputs,
)


def _domain_cfg() -> BSDomainConfig:
    return BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA, n_sigma=6.0, center="strike"
    )


@pytest.mark.parametrize("Nx,Nt", [(201, 201), (401, 401), (801, 801)])
@pytest.mark.parametrize("method", ["cn", "rannacher"])
def test_bench_pde_core_scaling(benchmark, Nx: int, Nt: int, method: str) -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    p = PricingInputs(
        spec=OptionSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0),
        market=market,
        sigma=0.2,
        t=0.0,
    )

    # Use the public PDE wrapper that already wires Black-Scholes coefficients.
    benchmark(
        bs_price_pde_european,
        p,
        coord="logS",
        domain_cfg=_domain_cfg(),
        Nx=Nx,
        Nt=Nt,
        method=method,
    )


@pytest.mark.parametrize("Nx,Nt", [(201, 201), (401, 401)])
@pytest.mark.parametrize("ic_remedy", [ICRemedy.NONE, ICRemedy.L2_PROJ])
def test_bench_pde_digital_ic_remedies(
    benchmark, Nx: int, Nt: int, ic_remedy: ICRemedy
) -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    p = PricingInputs(
        spec=DigitalSpec(kind=OptionType.CALL, strike=100.0, expiry=1.0, payout=1.0),
        market=market,
        sigma=0.2,
        t=0.0,
    )

    # Use the public digital PDE pricer to exercise IC remedy variants.
    benchmark(
        bs_digital_price_pde,
        p,
        coord="logS",
        domain_cfg=_domain_cfg(),
        Nx=Nx,
        Nt=Nt,
        method="rannacher",
        ic_remedy=ic_remedy,
    )
