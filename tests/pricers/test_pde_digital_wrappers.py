import pytest

from option_pricing.instruments.base import ExerciseStyle
from option_pricing.instruments.digital import DigitalOption
from option_pricing.models.black_scholes.bs import digital_call_price
from option_pricing.numerics.pde.ic_remedies import ICRemedy
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from option_pricing.pricers.pde_pricer import (
    bs_digital_price_pde,
    bs_digital_price_pde_from_ctx,
)
from option_pricing.types import DigitalSpec, MarketData, OptionType, PricingInputs


def _digital_inputs(
    *, S=100.0, K=100.0, r=0.02, q=0.01, sigma=0.2, tau=1.0, payout=1.0
):
    market = MarketData(spot=S, rate=r, dividend_yield=q)
    spec = DigitalSpec(kind=OptionType.CALL, strike=K, expiry=tau, payout=payout)
    return PricingInputs(spec=spec, market=market, sigma=sigma, t=0.0)


def test_bs_digital_price_pde_matches_analytic_cell_avg_return_solution():
    p = _digital_inputs()
    domain_cfg = BSDomainConfig(policy=BSDomainPolicy.LOG_NSIGMA, n_sigma=5.0)

    price, sol = bs_digital_price_pde(
        p,
        domain_cfg=domain_cfg,
        Nx=120,
        Nt=120,
        ic_remedy=ICRemedy.CELL_AVG,
        return_solution=True,
    )

    inst = DigitalOption(
        expiry=p.tau, strike=p.K, payout=p.spec.payout, kind=p.spec.kind
    )
    ref = float(digital_call_price(inst, p.ctx, p.sigma))

    assert abs(price - ref) < 2e-3
    assert sol is not None


def test_bs_digital_price_pde_l2_proj_smoke():
    p = _digital_inputs()
    domain_cfg = BSDomainConfig(policy=BSDomainPolicy.LOG_NSIGMA, n_sigma=5.0)

    price = bs_digital_price_pde(
        p,
        domain_cfg=domain_cfg,
        Nx=120,
        Nt=120,
        ic_remedy=ICRemedy.L2_PROJ,
        return_solution=False,
    )

    inst = DigitalOption(
        expiry=p.tau, strike=p.K, payout=p.spec.payout, kind=p.spec.kind
    )
    ref = float(digital_call_price(inst, p.ctx, p.sigma))
    assert abs(price - ref) < 3e-3


def test_bs_digital_price_pde_rejects_non_european():
    p = _digital_inputs()
    domain_cfg = BSDomainConfig(policy=BSDomainPolicy.LOG_NSIGMA, n_sigma=5.0)

    inst = DigitalOption(
        expiry=p.tau,
        strike=p.K,
        payout=p.spec.payout,
        kind=p.spec.kind,
        exercise=ExerciseStyle.AMERICAN,
    )

    with pytest.raises(ValueError):
        bs_digital_price_pde_from_ctx(
            inst, ctx=p.ctx, sigma=p.sigma, domain_cfg=domain_cfg, Nx=50, Nt=50
        )
