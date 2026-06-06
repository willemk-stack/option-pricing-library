from __future__ import annotations

import numpy as np

from option_pricing.diagnostics.vol_surface import (
    fixed_strikes_from_forward_y,
    localvol_compare_gatheral_vs_dupire_on_forward_y,
)
from option_pricing.models.black_scholes import bs
from option_pricing.types import MarketData
from option_pricing.vol.local_vol_dupire import local_vol_from_call_grid_diagnostics
from option_pricing.vol.local_vol_surface import LocalVolSurface
from option_pricing.vol.ssvi import (
    ESSVIImpliedSurface,
    ESSVITermStructures,
    EtaTermStructure,
    PsiTermStructure,
    ThetaTermStructure,
)


def _const_like(arg, value: float) -> np.ndarray:
    arr = np.asarray(arg, dtype=np.float64)
    return np.full_like(arr, float(value), dtype=np.float64)


def _smooth_essvi_params() -> ESSVITermStructures:
    return ESSVITermStructures(
        theta_term=ThetaTermStructure(
            value=lambda T: (
                0.04
                + 0.025 * np.asarray(T, dtype=np.float64)
                + 0.003 * np.asarray(T, dtype=np.float64) ** 2
            ),
            first_derivative=lambda T: 0.025 + 0.006 * np.asarray(T, dtype=np.float64),
            second_derivative=lambda T: _const_like(T, 0.006),
        ),
        psi_term=PsiTermStructure(
            value=lambda T: 0.08 + 0.01 * np.asarray(T, dtype=np.float64),
            first_derivative=lambda T: _const_like(T, 0.01),
            second_derivative=lambda T: _const_like(T, 0.0),
        ),
        eta_term=EtaTermStructure(
            value=lambda T: -0.02 + 0.002 * np.asarray(T, dtype=np.float64),
            first_derivative=lambda T: _const_like(T, 0.002),
            second_derivative=lambda T: _const_like(T, 0.0),
        ),
        eps=1.0e-14,
    )


def test_forward_y_fixed_strike_dupire_agrees_with_analytic_gatheral() -> None:
    market = MarketData(spot=100.0, rate=0.04, dividend_yield=0.01)
    ctx = market.to_context()
    surface = ESSVIImpliedSurface(params=_smooth_essvi_params())
    localvol = LocalVolSurface.from_implied(
        surface,
        forward=ctx.fwd,
        discount=ctx.df,
    )

    expiries = np.linspace(0.30, 2.00, 31, dtype=np.float64)
    y_grid = np.linspace(-0.20, 0.20, 101, dtype=np.float64)

    report = localvol_compare_gatheral_vs_dupire_on_forward_y(
        localvol,
        expiries=expiries,
        y_grid=y_grid,
        market=market,
        reference_expiry=1.0,
        price_convention="discounted",
        strike_coordinate="logK",
        trim_t=1,
        trim_k=1,
    )

    assert report.summary["gatheral_invalid_frac"] == 0.0
    assert report.boundary_summary["interior_invalid_count"] == 0
    assert report.boundary_summary["invalids_are_boundary_only"] is True
    assert report.summary["diff_sigma_max_abs"] < 2.0e-5
    assert report.invalid_points["is_boundary"].all()
    assert "fixed K grid" in report.coordinate_conventions["dupire_call_grid"]


def test_moving_forward_y_call_grid_creates_false_interior_dupire_invalids() -> None:
    market = MarketData(spot=100.0, rate=0.20, dividend_yield=0.0)
    ctx = market.to_context()
    surface = ESSVIImpliedSurface(params=_smooth_essvi_params())
    expiries = np.linspace(0.30, 2.00, 31, dtype=np.float64)
    y_grid = np.linspace(-0.20, 0.20, 101, dtype=np.float64)
    fixed_strikes = fixed_strikes_from_forward_y(
        y_grid,
        forward=ctx.fwd,
        reference_expiry=1.0,
    )

    moving_y_calls = np.vstack(
        [
            bs.black76_call_price_vec(
                forward=ctx.fwd(float(T)),
                strikes=ctx.fwd(float(T)) * np.exp(y_grid),
                sigma=surface.iv(y_grid, float(T)),
                tau=float(T),
                df=ctx.df(float(T)),
            )
            for T in expiries
        ]
    )

    report = local_vol_from_call_grid_diagnostics(
        moving_y_calls,
        fixed_strikes,
        expiries,
        market=market,
        price_convention="discounted",
        strike_coordinate="logK",
        trim_t=1,
        trim_k=1,
    )

    nT, nK = moving_y_calls.shape
    t_idx = np.arange(nT)[:, None]
    k_idx = np.arange(nK)[None, :]
    trimmed = (t_idx < 1) | (t_idx >= nT - 1) | (k_idx < 1) | (k_idx >= nK - 1)
    interior_invalid_count = int(np.sum(report.invalid & ~trimmed))

    assert interior_invalid_count > 0
