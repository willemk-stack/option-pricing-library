from __future__ import annotations

import numpy as np
import pytest

from option_pricing.types import MarketData, OptionType
from option_pricing.vol.ssvi import (
    ESSVICalibrationConfig,
    ESSVINodalSurface,
    ESSVINodeSet,
    ESSVITermStructures,
    EtaTermStructure,
    MingoneGlobalParams,
    PsiTermStructure,
    ThetaTermStructure,
    build_theta_term_from_quotes,
    calibrate_essvi,
    compute_Apsi_Cpsi,
    essvi_implied_price,
    essvi_total_variance,
    evaluate_essvi_constraints,
    interpolate_nodes,
    project_essvi_nodes,
    reconstruct_nodes_from_global_params,
    validate_essvi_nodes,
    validate_essvi_surface,
)
from option_pricing.vol.svi.transforms import softplus


def _log_term_params(*, T_pivot: float) -> ESSVITermStructures:
    log_pivot = float(np.log(T_pivot))
    eps = 1e-12

    def theta_value(T):
        T_arr = np.asarray(T, dtype=np.float64)
        return 0.04 + 0.03 * T_arr

    def theta_first(T):
        T_arr = np.asarray(T, dtype=np.float64)
        return np.full_like(T_arr, 0.03, dtype=np.float64)

    def psi_value(log_T):
        x = np.asarray(log_T, dtype=np.float64) - log_pivot
        return np.asarray(softplus(-1.5 + 0.15 * x) + eps, dtype=np.float64)

    def psi_first(log_T):
        x = np.asarray(log_T, dtype=np.float64) - log_pivot
        z = -1.5 + 0.15 * x
        return np.asarray(0.15 / (1.0 + np.exp(-z)), dtype=np.float64)

    def eta_value(log_T):
        x = np.asarray(log_T, dtype=np.float64) - log_pivot
        return np.asarray(-0.03 + 0.02 * x, dtype=np.float64)

    def eta_first(log_T):
        log_T_arr = np.asarray(log_T, dtype=np.float64)
        return np.full_like(log_T_arr, 0.02, dtype=np.float64)

    return ESSVITermStructures(
        theta_term=ThetaTermStructure(
            value=theta_value,
            first_derivative=theta_first,
        ),
        psi_term=PsiTermStructure(
            value=psi_value,
            first_derivative=psi_first,
            input_variable="log_T",
        ),
        eta_term=EtaTermStructure(
            value=eta_value,
            first_derivative=eta_first,
            input_variable="log_T",
        ),
        eps=eps,
    )


def _synthetic_quotes(
    *,
    params: ESSVITermStructures,
    market: MarketData,
    expiries: np.ndarray,
    y_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_all: list[float] = []
    T_all: list[float] = []
    price_all: list[float] = []
    is_call_all: list[bool] = []
    for tau in expiries:
        forward = market.fwd(float(tau))
        strike = forward * np.exp(y_grid)
        call_mask = y_grid >= 0.0
        call_price = essvi_implied_price(
            kind=OptionType.CALL,
            strike=strike,
            forward=forward,
            df=market.df(float(tau)),
            params=params,
            T=float(tau),
        )
        put_price = np.asarray(
            call_price - market.df(float(tau)) * (forward - strike),
            dtype=np.float64,
        )
        price = np.where(call_mask, call_price, put_price)
        y_all.extend(y_grid.tolist())
        T_all.extend(np.full_like(y_grid, float(tau)).tolist())
        price_all.extend(price.tolist())
        is_call_all.extend(call_mask.tolist())
    return (
        np.asarray(y_all, dtype=np.float64),
        np.asarray(T_all, dtype=np.float64),
        np.asarray(price_all, dtype=np.float64),
        np.asarray(is_call_all, dtype=np.bool_),
    )


def test_build_theta_term_from_quotes_uses_exact_atm_quotes() -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    expiries = np.array([0.25, 0.75, 1.5], dtype=np.float64)
    y_grid = np.array([-0.25, 0.0, 0.25], dtype=np.float64)
    params = _log_term_params(T_pivot=float(np.exp(np.mean(np.log(expiries)))))
    y, T, price_mkt, is_call = _synthetic_quotes(
        params=params,
        market=market,
        expiries=expiries,
        y_grid=y_grid,
    )

    theta_term, diag = build_theta_term_from_quotes(
        y=y,
        T=T,
        price_mkt=price_mkt,
        market=market,
        is_call=is_call,
    )

    np.testing.assert_array_equal(diag.extraction_methods, ("exact", "exact", "exact"))
    np.testing.assert_allclose(
        diag.theta_raw,
        params.theta(expiries),
        atol=5e-8,
    )
    np.testing.assert_allclose(theta_term(expiries), diag.theta_isotonic, atol=1e-12)
    assert np.all(theta_term.dT(expiries) >= 0.0)


def test_build_theta_term_from_quotes_interpolates_bracketing_quotes() -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    expiries = np.array([0.4, 1.0], dtype=np.float64)
    y_grid = np.array([-0.2, 0.2], dtype=np.float64)
    params = _log_term_params(T_pivot=float(np.exp(np.mean(np.log(expiries)))))
    y, T, price_mkt, is_call = _synthetic_quotes(
        params=params,
        market=market,
        expiries=expiries,
        y_grid=y_grid,
    )

    _, diag = build_theta_term_from_quotes(
        y=y,
        T=T,
        price_mkt=price_mkt,
        market=market,
        is_call=is_call,
    )

    np.testing.assert_array_equal(diag.extraction_methods, ("bracket", "bracket"))
    for i, tau in enumerate(expiries):
        w_pair = essvi_total_variance(y_grid, params=params, T=float(tau))
        expected = w_pair[0] + (-y_grid[0]) * (w_pair[1] - w_pair[0]) / (
            y_grid[1] - y_grid[0]
        )
        assert float(diag.theta_raw[i]) == pytest.approx(float(expected), abs=5e-8)


def test_build_theta_term_from_quotes_falls_back_to_nearest_quote() -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    expiries = np.array([0.5, 1.0], dtype=np.float64)
    y_grid = np.array([0.1, 0.4], dtype=np.float64)
    params = _log_term_params(T_pivot=float(np.exp(np.mean(np.log(expiries)))))
    y, T, price_mkt, is_call = _synthetic_quotes(
        params=params,
        market=market,
        expiries=expiries,
        y_grid=y_grid,
    )

    _, diag = build_theta_term_from_quotes(
        y=y,
        T=T,
        price_mkt=price_mkt,
        market=market,
        is_call=is_call,
    )

    np.testing.assert_array_equal(diag.extraction_methods, ("nearest", "nearest"))
    for i, tau in enumerate(expiries):
        expected = essvi_total_variance(y_grid[0], params=params, T=float(tau))
        assert float(diag.theta_raw[i]) == pytest.approx(float(expected), abs=5e-8)


def test_evaluate_essvi_constraints_reports_valid_and_invalid_cases() -> None:
    expiries = np.array([0.3, 0.8, 1.4], dtype=np.float64)
    valid = _log_term_params(T_pivot=float(np.exp(np.mean(np.log(expiries)))))
    report_ok = evaluate_essvi_constraints(valid, expiries)
    assert report_ok.ok

    invalid = ESSVITermStructures(
        theta_term=ThetaTermStructure.constant(0.04),
        psi_term=PsiTermStructure.constant(3.95),
        eta_term=EtaTermStructure.constant(0.10),
    )
    report_bad = evaluate_essvi_constraints(invalid, expiries)
    assert not report_bad.ok
    assert "lee_constraint" in report_bad.violations


def test_validate_essvi_surface_reports_and_strict_mode() -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    expiries = np.array([0.25, 0.5, 1.0], dtype=np.float64)
    valid = _log_term_params(T_pivot=float(np.exp(np.mean(np.log(expiries)))))

    with pytest.deprecated_call(
        match="validate_essvi_surface is deprecated; use validate_essvi_continuous."
    ):
        report = validate_essvi_surface(valid, market, expiries=expiries)
    assert report.ok

    invalid = ESSVITermStructures(
        theta_term=ThetaTermStructure.constant(0.04),
        psi_term=PsiTermStructure.constant(3.95),
        eta_term=EtaTermStructure.constant(0.10),
    )
    with pytest.deprecated_call(
        match="validate_essvi_surface is deprecated; use validate_essvi_continuous."
    ):
        report_bad = validate_essvi_surface(invalid, market, expiries=expiries)
    assert not report_bad.ok

    with pytest.deprecated_call(
        match="validate_essvi_surface is deprecated; use validate_essvi_continuous."
    ):
        with pytest.raises(ValueError):
            validate_essvi_surface(invalid, market, expiries=expiries, strict=True)


def test_calibrate_essvi_recovers_synthetic_price_surface() -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    expiries = np.array([0.25, 0.5, 1.0, 1.5], dtype=np.float64)
    y_grid = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float64)
    true_params = _log_term_params(T_pivot=float(np.exp(np.mean(np.log(expiries)))))
    y, T, price_mkt, is_call = _synthetic_quotes(
        params=true_params,
        market=market,
        expiries=expiries,
        y_grid=y_grid,
    )

    result = calibrate_essvi(
        y=y,
        T=T,
        price_mkt=price_mkt,
        market=market,
        is_call=is_call,
        cfg=ESSVICalibrationConfig(strict_validation=True),
    )

    assert result.diag.node_validation.ok
    fitted_surface = ESSVINodalSurface(result.nodes)
    objective_price = fitted_surface.price(
        kind=OptionType.CALL,
        strike=np.asarray(
            [
                market.fwd(float(tau)) * np.exp(y_i)
                for y_i, tau in zip(y, T, strict=True)
            ],
            dtype=np.float64,
        ),
        forward=np.asarray([market.fwd(float(tau)) for tau in T], dtype=np.float64),
        df=np.asarray([market.df(float(tau)) for tau in T], dtype=np.float64),
        T=T,
    )
    put_price = objective_price - np.asarray(
        [
            market.df(float(tau)) * (market.fwd(float(tau)) - strike)
            for tau, strike in zip(
                T,
                np.asarray(
                    [
                        market.fwd(float(tau)) * np.exp(y_i)
                        for y_i, tau in zip(y, T, strict=True)
                    ],
                    dtype=np.float64,
                ),
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    fitted_price = np.where(is_call, objective_price, put_price)
    np.testing.assert_allclose(fitted_price, price_mkt, atol=5e-4)

    projection = project_essvi_nodes(result.nodes)
    assert projection.success
    assert projection.surface is not None
    y_probe = np.array([-0.2, 0.0, 0.2], dtype=np.float64)
    w, w_y, w_yy, w_T = projection.surface.w_and_derivs(y_probe, 0.75)
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(w_y))
    assert np.all(np.isfinite(w_yy))
    assert np.all(np.isfinite(w_T))


def test_reconstruct_nodes_from_global_params_produces_admissible_nodes() -> None:
    params = MingoneGlobalParams(
        expiries=np.array([0.25, 0.5, 1.0], dtype=np.float64),
        rho_raw=np.array([0.1, -0.2, 0.15], dtype=np.float64),
        theta1_raw=-2.5,
        a_raw=np.array([-2.0, -1.7], dtype=np.float64),
        c_raw=np.array([0.0, -0.4, 0.6], dtype=np.float64),
        rho_cap=0.95,
    )

    nodes = reconstruct_nodes_from_global_params(params)
    report = validate_essvi_nodes(nodes, strict=True)
    A, C = compute_Apsi_Cpsi(nodes.theta, nodes.rho, nodes.psi)

    assert report.ok
    assert np.all(nodes.theta > 0.0)
    assert np.all(nodes.psi > 0.0)
    assert np.all(np.abs(nodes.rho) < 1.0)
    assert np.all(A[1:] < nodes.psi[1:])
    assert np.all(nodes.psi < C + 1e-12)


def test_interpolate_nodes_is_linear_in_theta_psi_and_chi() -> None:
    nodes = ESSVINodeSet(
        expiries=np.array([0.5, 1.5], dtype=np.float64),
        theta=np.array([0.04, 0.10], dtype=np.float64),
        psi=np.array([0.18, 0.24], dtype=np.float64),
        rho=np.array([-0.25, 0.10], dtype=np.float64),
    )

    state_mid = interpolate_nodes(nodes, 1.0)
    chi = nodes.psi * nodes.rho
    np.testing.assert_allclose(np.asarray(state_mid.theta), 0.07, atol=1e-12)
    np.testing.assert_allclose(np.asarray(state_mid.psi), 0.21, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(state_mid.eta),
        0.5 * (chi[0] + chi[1]),
        atol=1e-12,
    )

    state_left = interpolate_nodes(nodes, 0.25)
    np.testing.assert_allclose(
        np.asarray(state_left.theta), 0.5 * nodes.theta[0], atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(state_left.psi), 0.5 * nodes.psi[0], atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(state_left.rho), nodes.rho[0], atol=1e-12)

    state_right = interpolate_nodes(nodes, 2.0)
    expected_theta = nodes.theta[-1] + (
        (nodes.theta[-1] - nodes.theta[-2]) / (nodes.expiries[-1] - nodes.expiries[-2])
    ) * (2.0 - nodes.expiries[-1])
    np.testing.assert_allclose(
        np.asarray(state_right.theta), expected_theta, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(state_right.psi), nodes.psi[-1], atol=1e-12)
    np.testing.assert_allclose(np.asarray(state_right.rho), nodes.rho[-1], atol=1e-12)
