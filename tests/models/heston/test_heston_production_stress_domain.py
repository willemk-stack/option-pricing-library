from __future__ import annotations

import math

import numpy as np
import pytest

from option_pricing.models.heston import (
    HestonParams,
    recommend_heston_quadrature_config,
)
from option_pricing.pricers.heston import heston_price_from_ctx
from option_pricing.types import MarketData, OptionType

_BOUND_TOLERANCE = 1.0e-4


@pytest.mark.parametrize(
    "params",
    [
        HestonParams(
            kappa=0.065,
            vbar=0.95,
            eta=2.30,
            rho=0.02,
            v=0.045,
        ),
        HestonParams(
            kappa=0.30,
            vbar=0.20,
            eta=2.40,
            rho=-0.12,
            v=0.04,
        ),
        HestonParams(
            kappa=1.00,
            vbar=0.06,
            eta=1.50,
            rho=-0.85,
            v=0.03,
        ),
        HestonParams(
            kappa=3.00,
            vbar=0.05,
            eta=1.80,
            rho=-0.70,
            v=0.03,
        ),
        HestonParams(
            kappa=5.10,
            vbar=0.08,
            eta=1.40,
            rho=-0.45,
            v=0.06,
        ),
    ],
)
@pytest.mark.parametrize("tau", [2.0 / 365.0, 0.10])
@pytest.mark.parametrize("discount_factor", [0.8, 1.0])
def test_robust_production_stress_prices_are_finite_and_respect_bounds(
    params: HestonParams,
    tau: float,
    discount_factor: float,
) -> None:
    forward = 100.0
    log_forward_moneyness = np.asarray([-1.3, 0.0, 1.3])
    strikes = forward / np.exp(log_forward_moneyness)
    rate = -math.log(discount_factor) / tau
    context = MarketData(
        spot=forward,
        rate=rate,
        dividend_yield=rate,
    ).to_context()
    cfg = recommend_heston_quadrature_config(
        x=float(np.max(np.abs(log_forward_moneyness))),
        tau=tau,
        params=params,
        quality="robust",
    )

    call = np.asarray(
        heston_price_from_ctx(
            kind=OptionType.CALL,
            strike=strikes,
            tau=tau,
            ctx=context,
            params=params,
            backend="gauss_legendre",
            quad_cfg=cfg,
        ),
        dtype=float,
    )
    put = np.asarray(
        heston_price_from_ctx(
            kind=OptionType.PUT,
            strike=strikes,
            tau=tau,
            ctx=context,
            params=params,
            backend="gauss_legendre",
            quad_cfg=cfg,
        ),
        dtype=float,
    )

    assert np.all(np.isfinite(call))
    assert np.all(np.isfinite(put))
    call_lower_bound = discount_factor * np.maximum(forward - strikes, 0.0)
    put_lower_bound = discount_factor * np.maximum(strikes - forward, 0.0)
    assert np.all(call >= call_lower_bound - _BOUND_TOLERANCE)
    assert np.all(put >= put_lower_bound - _BOUND_TOLERANCE)
