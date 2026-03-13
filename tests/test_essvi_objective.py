from __future__ import annotations

import numpy as np
import pytest

from option_pricing.types import MarketData, OptionType
from option_pricing.vol.ssvi import (
    ESSVIPriceObjective,
    ESSVITermStructures,
    EtaTermStructure,
    PsiTermStructure,
    ThetaTermStructure,
    essvi_implied_price,
)


def _const_like(arg, value: float) -> np.ndarray:
    arg_arr = np.asarray(arg, dtype=np.float64)
    return np.full_like(arg_arr, float(value), dtype=np.float64)


def _quadratic_params(*, eps: float = 1e-12) -> ESSVITermStructures:
    return ESSVITermStructures(
        theta_term=ThetaTermStructure(
            value=lambda T: (
                0.06
                + 0.025 * np.asarray(T, dtype=np.float64)
                + 0.004 * np.asarray(T, dtype=np.float64) ** 2
            ),
            first_derivative=lambda T: 0.025 + 0.008 * np.asarray(T, dtype=np.float64),
            second_derivative=lambda T: _const_like(T, 0.008),
        ),
        psi_term=PsiTermStructure(
            value=lambda T: (
                0.20
                + 0.03 * np.asarray(T, dtype=np.float64)
                + 0.005 * np.asarray(T, dtype=np.float64) ** 2
            ),
            first_derivative=lambda T: 0.03 + 0.01 * np.asarray(T, dtype=np.float64),
            second_derivative=lambda T: _const_like(T, 0.01),
        ),
        eta_term=EtaTermStructure(
            value=lambda T: (
                -0.04
                + 0.008 * np.asarray(T, dtype=np.float64)
                - 0.001 * np.asarray(T, dtype=np.float64) ** 2
            ),
            first_derivative=lambda T: 0.008 - 0.002 * np.asarray(T, dtype=np.float64),
            second_derivative=lambda T: _const_like(T, -0.002),
        ),
        eps=eps,
    )


def test_essvi_price_objective_uses_explicit_option_side() -> None:
    params = _quadratic_params()
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    T = np.array([0.5, 0.75, 1.0], dtype=np.float64)
    y = np.array([-0.2, 0.0, 0.15], dtype=np.float64)
    weights = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    is_call = np.array([False, True, True], dtype=np.bool_)

    objective = ESSVIPriceObjective(
        y=y,
        T=T,
        price_mkt=np.zeros_like(y),
        weights=weights,
        market=market,
        is_call=is_call,
    )

    forward, df = objective.forward_and_df()
    strike = objective.strikes()
    call_model = essvi_implied_price(
        kind=OptionType.CALL,
        strike=strike,
        forward=forward,
        df=df,
        params=params,
        T=T,
    )
    put_model = call_model - df * (forward - strike)
    expected = np.where(is_call, call_model, put_model)

    np.testing.assert_allclose(objective.model_prices(params), expected, atol=1e-12)


def test_essvi_price_objective_defaults_to_otm_side_from_y() -> None:
    params = _quadratic_params()
    market = MarketData(spot=100.0, rate=0.015, dividend_yield=0.0)
    T = np.array([0.4, 0.4, 0.4], dtype=np.float64)
    y = np.array([-0.25, 0.0, 0.30], dtype=np.float64)

    objective = ESSVIPriceObjective(
        y=y,
        T=T,
        price_mkt=np.zeros_like(y),
        weights=np.ones_like(y),
        market=market,
    )

    forward, df = objective.forward_and_df()
    strike = objective.strikes()
    call_model = essvi_implied_price(
        kind=OptionType.CALL,
        strike=strike,
        forward=forward,
        df=df,
        params=params,
        T=T,
    )
    put_model = call_model - df * (forward - strike)
    expected = np.array([put_model[0], call_model[1], call_model[2]], dtype=np.float64)

    np.testing.assert_allclose(objective.model_prices(params), expected, atol=1e-12)
    np.testing.assert_array_equal(objective.call_mask(), np.array([False, True, True]))


def test_essvi_price_objective_residual_applies_weights() -> None:
    params = _quadratic_params()
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    T = np.array([0.35, 0.7], dtype=np.float64)
    y = np.array([-0.1, 0.2], dtype=np.float64)
    weights = np.array([2.0, 0.5], dtype=np.float64)

    base_objective = ESSVIPriceObjective(
        y=y,
        T=T,
        price_mkt=np.zeros_like(y),
        weights=np.ones_like(y),
        market=market,
    )
    model = base_objective.model_prices(params)
    price_mkt = model - np.array([0.01, -0.02], dtype=np.float64)

    objective = ESSVIPriceObjective(
        y=y,
        T=T,
        price_mkt=price_mkt,
        weights=weights,
        market=market,
    )

    expected = weights * (model - price_mkt)
    np.testing.assert_allclose(objective.residual(params), expected, atol=1e-12)


def test_essvi_price_objective_validates_inputs() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)

    with pytest.raises(ValueError, match="same size"):
        ESSVIPriceObjective(
            y=np.array([0.0, 0.1], dtype=np.float64),
            T=np.array([0.5], dtype=np.float64),
            price_mkt=np.array([1.0, 2.0], dtype=np.float64),
            weights=np.array([1.0, 1.0], dtype=np.float64),
            market=market,
        )

    with pytest.raises(ValueError, match="weights must be >= 0"):
        ESSVIPriceObjective(
            y=np.array([0.0], dtype=np.float64),
            T=np.array([0.5], dtype=np.float64),
            price_mkt=np.array([1.0], dtype=np.float64),
            weights=np.array([-1.0], dtype=np.float64),
            market=market,
        )
