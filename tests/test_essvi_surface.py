from __future__ import annotations

import numpy as np
import pytest

from option_pricing.market.curves import PricingContext
from option_pricing.models.black_scholes.bs import (
    black76_call_price_vec,
    black76_put_price_vec,
)
from option_pricing.types import MarketData, OptionType
from option_pricing.vol import LocalVolSurface
from option_pricing.vol.ssvi import (
    ESSVIImpliedSurface,
    ESSVITermStructures,
    EtaTermStructure,
    PsiTermStructure,
    ThetaTermStructure,
    essvi_implied_price,
    essvi_total_variance,
    essvi_total_variance_dk,
    essvi_total_variance_dk_dT,
    essvi_total_variance_dkk,
    essvi_total_variance_dT,
    essvi_total_variance_dTT,
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


def test_essvi_matches_classic_theta_phi_rho_formula() -> None:
    params = _quadratic_params()
    T = 0.85
    y = np.array([-1.2, -0.5, 0.0, 0.4, 1.1], dtype=np.float64)

    theta = params.theta(T)
    phi = params.phi(T)
    rho = params.rho(T)

    classic = (
        0.5
        * theta
        * (1.0 + rho * phi * y + np.sqrt((phi * y + rho) ** 2 + 1.0 - rho * rho))
    )
    new = essvi_total_variance(y=y, params=params, T=T)

    np.testing.assert_allclose(new, classic, rtol=0.0, atol=1e-12)


def test_derived_parameters_and_maturity_derivatives_match_ratio_formulas() -> None:
    params = _quadratic_params()
    T = np.array([0.2, 0.7, 1.4], dtype=np.float64)

    theta = params.theta(T)
    psi = params.psi(T)
    eta = params.eta(T)
    dtheta = params.dtheta_dT(T)
    dpsi = params.dpsi_dT(T)
    deta = params.deta_dT(T)
    d2theta = params.d2theta_dT2(T)
    d2psi = params.d2psi_dT2(T)
    d2eta = params.d2eta_dT2(T)

    expected_rho = eta / psi
    expected_phi = psi / theta
    expected_drho = (deta * psi - eta * dpsi) / (psi * psi)
    expected_dphi = (dpsi * theta - psi * dtheta) / (theta * theta)
    expected_d2rho = (d2eta * psi - eta * d2psi) / (psi * psi) - 2.0 * (
        (deta * psi - eta * dpsi) * dpsi
    ) / (psi * psi * psi)
    expected_d2phi = (d2psi * theta - psi * d2theta) / (theta * theta) - 2.0 * (
        (dpsi * theta - psi * dtheta) * dtheta
    ) / (theta * theta * theta)

    np.testing.assert_allclose(params.rho(T), expected_rho, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(params.phi(T), expected_phi, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(params.drho_dT(T), expected_drho, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(params.dphi_dT(T), expected_dphi, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        params.d2rho_dT2(T), expected_d2rho, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        params.d2phi_dT2(T), expected_d2phi, rtol=0.0, atol=1e-12
    )


def test_log_maturity_term_structure_reports_T_derivatives() -> None:
    theta_term = ThetaTermStructure(
        value=lambda x: (
            0.08
            + 0.02 * np.asarray(x, dtype=np.float64)
            + 0.03 * np.asarray(x, dtype=np.float64) ** 2
        ),
        first_derivative=lambda x: 0.02 + 0.06 * np.asarray(x, dtype=np.float64),
        second_derivative=lambda x: _const_like(x, 0.06),
        input_variable="log_T",
    )

    T = np.array([0.35, 1.0, 1.7], dtype=np.float64)
    x = np.log(T)

    expected_value = 0.08 + 0.02 * x + 0.03 * x * x
    expected_first = (0.02 + 0.06 * x) / T
    expected_second = (0.06 - (0.02 + 0.06 * x)) / (T * T)

    np.testing.assert_allclose(theta_term(T), expected_value, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(theta_term.dT(T), expected_first, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        theta_term.d2T2(T), expected_second, rtol=0.0, atol=1e-12
    )


def test_analytic_essvi_derivatives_match_finite_differences() -> None:
    params = _quadratic_params()
    T = 0.9
    y = np.array([-0.8, -0.25, 0.1, 0.75], dtype=np.float64)
    h_y = 1e-6
    h_y2 = 1e-5
    h_T = 1e-6
    h_T2 = 1e-5

    fd_dk = (
        essvi_total_variance(y=y + h_y, params=params, T=T)
        - essvi_total_variance(y=y - h_y, params=params, T=T)
    ) / (2.0 * h_y)
    fd_dkk = (
        essvi_total_variance(y=y + h_y2, params=params, T=T)
        - 2.0 * essvi_total_variance(y=y, params=params, T=T)
        + essvi_total_variance(y=y - h_y2, params=params, T=T)
    ) / (h_y2 * h_y2)
    fd_dT = (
        essvi_total_variance(y=y, params=params, T=T + h_T)
        - essvi_total_variance(y=y, params=params, T=T - h_T)
    ) / (2.0 * h_T)
    fd_dk_dT = (
        essvi_total_variance_dk(y=y, params=params, T=T + h_T)
        - essvi_total_variance_dk(y=y, params=params, T=T - h_T)
    ) / (2.0 * h_T)
    fd_dTT = (
        essvi_total_variance_dT(y=y, params=params, T=T + h_T2)
        - essvi_total_variance_dT(y=y, params=params, T=T - h_T2)
    ) / (2.0 * h_T2)

    np.testing.assert_allclose(
        essvi_total_variance_dk(y=y, params=params, T=T),
        fd_dk,
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        essvi_total_variance_dkk(y=y, params=params, T=T),
        fd_dkk,
        rtol=0.0,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        essvi_total_variance_dT(y=y, params=params, T=T),
        fd_dT,
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        essvi_total_variance_dk_dT(y=y, params=params, T=T),
        fd_dk_dT,
        rtol=0.0,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        essvi_total_variance_dTT(y=y, params=params, T=T),
        fd_dTT,
        rtol=0.0,
        atol=2e-5,
    )


def test_short_maturity_and_small_theta_remain_finite() -> None:
    params = ESSVITermStructures(
        theta_term=ThetaTermStructure(
            value=lambda T: 1e-8 + 1e-4 * np.asarray(T, dtype=np.float64),
            first_derivative=lambda T: _const_like(T, 1e-4),
            second_derivative=lambda T: _const_like(T, 0.0),
        ),
        psi_term=PsiTermStructure.constant(2e-4),
        eta_term=EtaTermStructure.constant(0.0),
        eps=1e-14,
    )
    T = 1e-6
    y = np.array([-0.6, 0.0, 0.6], dtype=np.float64)

    w = essvi_total_variance(y=y, params=params, T=T)
    dw = essvi_total_variance_dk(y=y, params=params, T=T)
    d2w = essvi_total_variance_dkk(y=y, params=params, T=T)
    dT = essvi_total_variance_dT(y=y, params=params, T=T)

    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(dw))
    assert np.all(np.isfinite(d2w))
    assert np.all(np.isfinite(dT))
    assert np.all(np.isfinite(params.phi(T)))


def test_essvi_implied_price_matches_black76_for_scalar_term_inputs() -> None:
    params = _quadratic_params()
    T = 0.8
    forward = 102.0
    df = 0.97
    strike = np.array([80.0, 95.0, 102.0, 115.0], dtype=np.float64)

    y = np.log(strike / forward)
    sigma = np.sqrt(essvi_total_variance(y=y, params=params, T=T) / T)

    call_expected = black76_call_price_vec(
        forward=forward,
        strikes=strike,
        sigma=sigma,
        tau=T,
        df=df,
    )
    put_expected = black76_put_price_vec(
        forward=forward,
        strikes=strike,
        sigma=sigma,
        tau=T,
        df=df,
    )

    call_price = essvi_implied_price(
        kind=OptionType.CALL,
        strike=strike,
        forward=forward,
        df=df,
        params=params,
        T=T,
    )
    put_price = essvi_implied_price(
        kind=OptionType.PUT,
        strike=strike,
        forward=forward,
        df=df,
        params=params,
        T=T,
    )

    np.testing.assert_allclose(call_price, call_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(put_price, put_expected, rtol=0.0, atol=1e-12)


def test_essvi_implied_price_broadcasts_across_surface_inputs() -> None:
    params = _quadratic_params()
    T = np.array([0.35, 0.9, 1.6], dtype=np.float64)
    forward = np.array([101.0, 103.0, 106.0], dtype=np.float64)
    df = np.array([0.995, 0.982, 0.955], dtype=np.float64)
    strike = np.array([90.0, 100.0, 112.0], dtype=np.float64)

    price = essvi_implied_price(
        kind=OptionType.CALL,
        strike=strike,
        forward=forward,
        df=df,
        params=params,
        T=T,
    )

    assert price.shape == strike.shape
    assert np.all(np.isfinite(price))

    parity_gap = price - essvi_implied_price(
        kind=OptionType.PUT,
        strike=strike,
        forward=forward,
        df=df,
        params=params,
        T=T,
    )
    np.testing.assert_allclose(parity_gap, df * (forward - strike), atol=1e-12)


def test_essvi_implied_price_accepts_market_context() -> None:
    params = _quadratic_params()
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
    T = np.array([0.25, 0.75, 1.25], dtype=np.float64)
    strike = np.array([92.0, 100.0, 110.0], dtype=np.float64)

    expected = essvi_implied_price(
        kind=OptionType.CALL,
        strike=strike,
        forward=np.array([market.fwd(float(tau)) for tau in T], dtype=np.float64),
        df=np.array([market.df(float(tau)) for tau in T], dtype=np.float64),
        params=params,
        T=T,
    )
    from_market = essvi_implied_price(
        kind=OptionType.CALL,
        strike=strike,
        params=params,
        T=T,
        market=market,
    )

    np.testing.assert_allclose(from_market, expected, rtol=0.0, atol=1e-12)


def test_essvi_implied_price_rejects_mixed_market_and_forward_inputs() -> None:
    params = _quadratic_params()
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)

    with pytest.raises(ValueError, match="either market or forward/df"):
        essvi_implied_price(
            kind=OptionType.CALL,
            strike=100.0,
            forward=101.0,
            df=0.99,
            params=params,
            T=1.0,
            market=market,
        )


def test_essvi_surface_price_methods_delegate_to_function() -> None:
    params = _quadratic_params()
    surface = ESSVIImpliedSurface(params=params)
    smile = surface.slice(0.75)

    strike = np.array([92.0, 100.0, 111.0], dtype=np.float64)
    forward = 104.0
    df = 0.98

    np.testing.assert_allclose(
        smile.price_at(
            kind=OptionType.CALL,
            strike=strike,
            forward=forward,
            df=df,
        ),
        essvi_implied_price(
            kind=OptionType.CALL,
            strike=strike,
            forward=forward,
            df=df,
            params=params,
            T=0.75,
        ),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        surface.price(
            kind=OptionType.PUT,
            strike=strike,
            forward=forward,
            df=df,
            T=0.75,
        ),
        essvi_implied_price(
            kind=OptionType.PUT,
            strike=strike,
            forward=forward,
            df=df,
            params=params,
            T=0.75,
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_essvi_surface_price_methods_accept_market_context() -> None:
    params = _quadratic_params()
    surface = ESSVIImpliedSurface(params=params)
    smile = surface.slice(0.75)
    market = PricingContext(
        spot=100.0,
        discount=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
        .to_context()
        .discount,
        forward=MarketData(spot=100.0, rate=0.02, dividend_yield=0.01)
        .to_context()
        .forward,
    )

    strike = np.array([92.0, 100.0, 111.0], dtype=np.float64)

    np.testing.assert_allclose(
        smile.price_at(kind=OptionType.CALL, strike=strike, market=market),
        essvi_implied_price(
            kind=OptionType.CALL,
            strike=strike,
            params=params,
            T=0.75,
            market=market,
        ),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        surface.price(kind=OptionType.PUT, strike=strike, T=0.75, market=market),
        essvi_implied_price(
            kind=OptionType.PUT,
            strike=strike,
            params=params,
            T=0.75,
            market=market,
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_nearly_flat_smile_is_even_when_eta_is_zero() -> None:
    params = ESSVITermStructures(
        theta_term=ThetaTermStructure.constant(0.04),
        psi_term=PsiTermStructure.constant(1e-4),
        eta_term=EtaTermStructure.constant(0.0),
    )
    T = 1.0
    y = np.array([0.2, 0.7, 1.1], dtype=np.float64)

    np.testing.assert_allclose(
        essvi_total_variance(y=y, params=params, T=T),
        essvi_total_variance(y=-y, params=params, T=T),
        rtol=0.0,
        atol=1e-14,
    )


def test_validation_and_division_guards_raise_on_invalid_inputs() -> None:
    invalid = ESSVITermStructures(
        theta_term=ThetaTermStructure.constant(0.04),
        psi_term=PsiTermStructure.constant(0.10),
        eta_term=EtaTermStructure.constant(0.10),
    )
    with pytest.raises(ValueError, match=r"\|eta\(T\)\| < psi\(T\)"):
        invalid.validate(1.0)

    tiny_theta = ESSVITermStructures(
        theta_term=ThetaTermStructure.constant(1e-15),
        psi_term=PsiTermStructure.constant(0.20),
        eta_term=EtaTermStructure.constant(0.01),
        eps=1e-12,
    )
    with pytest.raises(ValueError, match=r"phi\(T\)"):
        tiny_theta.phi(1.0)

    with pytest.raises(ValueError, match="T must be finite and > 0"):
        invalid.validate(0.0)


def test_local_vol_surface_accepts_analytic_essvi_surface() -> None:
    params = _quadratic_params(eps=1e-14)
    implied = ESSVIImpliedSurface(params=params)

    def fwd(_T: float) -> float:
        return 100.0

    lv = LocalVolSurface.from_implied(implied, forward=fwd)
    out = lv.local_var(np.array([80.0, 100.0, 120.0], dtype=np.float64), 0.9)

    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
