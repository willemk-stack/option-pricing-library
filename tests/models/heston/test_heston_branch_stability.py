from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston import HestonParams
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.pricers.heston import heston_price_call_from_ctx
from option_pricing.types import MarketData, OptionSpec, OptionType
from option_pricing.vol.implied_vol_scalar import implied_vol_bs

MARKET = MarketData(spot=100.0, rate=0.02, dividend_yield=0.005)
CTX = MARKET.to_context()

# Dense enough to reveal branch/log discontinuities, but still practical for CI.
BRANCH_GUARD_QUAD_CFG = QuadratureConfig(u_max=260.0, n_panels=56, nodes_per_panel=32)
EXTREME_BRANCH_GUARD_QUAD_CFG = QuadratureConfig(
    u_max=1200.0,
    n_panels=300,
    nodes_per_panel=64,
)


BRANCH_GUARD_PARAMS = HestonParams(
    kappa=1.35,
    vbar=0.045,
    eta=0.95,
    rho=-0.82,
    v=0.035,
)


def _call_slice(
    *,
    params: HestonParams,
    strikes: np.ndarray,
    tau: float,
    quad_cfg: QuadratureConfig = BRANCH_GUARD_QUAD_CFG,
) -> np.ndarray:
    return np.asarray(
        heston_price_call_from_ctx(
            strike=strikes,
            ctx=CTX,
            tau=tau,
            params=params,
            backend="gauss_legendre",
            quad_cfg=quad_cfg,
        ),
        dtype=np.float64,
    )


def _implied_vol_slice(
    *,
    prices: np.ndarray,
    strikes: np.ndarray,
    tau: float,
) -> np.ndarray:
    vols: list[float] = []
    for price, strike in zip(prices, strikes, strict=True):
        vols.append(
            float(
                implied_vol_bs(
                    float(price),
                    OptionSpec(
                        kind=OptionType.CALL,
                        strike=float(strike),
                        expiry=float(tau),
                    ),
                    MARKET,
                )
            )
        )
    return np.asarray(vols, dtype=np.float64)


def _assert_no_large_local_spike(values: np.ndarray, *, max_spike_ratio: float) -> None:
    """Detect isolated one-point jumps relative to neighboring slope scale."""
    first_diff = np.diff(values)
    local_scale = np.maximum(
        np.maximum(np.abs(first_diff[:-1]), np.abs(first_diff[1:])),
        1.0e-12,
    )
    second_diff = np.diff(values, n=2)
    spike_ratio = np.abs(second_diff) / local_scale

    assert float(np.max(spike_ratio)) < max_spike_ratio, (
        "Potential branch/log discontinuity: isolated local spike detected.\n"
        f"max_spike_ratio={float(np.max(spike_ratio)):.6f}\n"
        f"values={values!r}"
    )


def test_heston_branch_guard_price_slice_is_smooth_and_arbitrage_sane() -> None:
    """Guard against Little-Heston-Trap-style discontinuities in price space.

    A branch/log discontinuity often appears as a one-point jump or jagged kink
    in a dense strike slice even when individual scalar prices are finite.
    """
    tau = 1.0
    strikes = np.linspace(70.0, 135.0, 66)
    calls = _call_slice(params=BRANCH_GUARD_PARAMS, strikes=strikes, tau=tau)

    assert np.all(np.isfinite(calls))
    assert np.all(calls >= -1.0e-10)

    # European call prices should be decreasing and convex in strike.
    assert np.all(np.diff(calls) <= 1.0e-8)
    assert np.all(np.diff(calls, n=2) >= -2.0e-5)

    _assert_no_large_local_spike(calls, max_spike_ratio=2.25)


@pytest.mark.slow
def test_heston_branch_guard_implied_vol_smile_has_no_jagged_branch_artifact() -> None:
    """Guard against branch/log discontinuities after IV conversion.

    This test is marked slow because it performs scalar implied-vol inversion
    across a dense slice. Keep it in the slow suite if CI budget is tight.
    """
    tau = 1.0
    strikes = np.linspace(75.0, 125.0, 41)
    calls = _call_slice(params=BRANCH_GUARD_PARAMS, strikes=strikes, tau=tau)
    vols = _implied_vol_slice(prices=calls, strikes=strikes, tau=tau)

    assert np.all(np.isfinite(vols))
    assert np.all(vols > 0.0)

    # NOTE: This threshold is a numerical-regression guard, not a model-law.
    # It is scoped to the fixed quadrature reference cases used here.
    assert float(np.max(np.abs(np.diff(vols)))) < 0.025
    _assert_no_large_local_spike(vols, max_spike_ratio=2.75)


def test_heston_branch_guard_prices_move_continuously_under_parameter_perturbations() -> (
    None
):
    """Small parameter perturbations should not cause discontinuous repricing."""
    tau = 1.25
    strikes = np.linspace(80.0, 125.0, 31)
    base = _call_slice(params=BRANCH_GUARD_PARAMS, strikes=strikes, tau=tau)

    perturbations = (
        HestonParams(kappa=1.35 * 1.001, vbar=0.045, eta=0.95, rho=-0.82, v=0.035),
        HestonParams(kappa=1.35, vbar=0.045 * 1.001, eta=0.95, rho=-0.82, v=0.035),
        HestonParams(kappa=1.35, vbar=0.045, eta=0.95 * 1.001, rho=-0.82, v=0.035),
        HestonParams(kappa=1.35, vbar=0.045, eta=0.95, rho=-0.819, v=0.035),
        HestonParams(kappa=1.35, vbar=0.045, eta=0.95, rho=-0.82, v=0.035 * 1.001),
    )

    price_floor = np.maximum(np.abs(base), 1.0)
    for params in perturbations:
        moved = _call_slice(params=params, strikes=strikes, tau=tau)
        rel_move = (moved - base) / price_floor

        assert np.all(np.isfinite(moved))
        # NOTE: This guards against local jaggedness in the perturbation
        # response, not the magnitude of legitimate smooth Heston sensitivity.
        _assert_no_large_local_spike(rel_move, max_spike_ratio=2.25)


def test_heston_branch_guard_extreme_but_admissible_negative_rho_slice_is_finite() -> (
    None
):
    """Exercise a difficult skew regime without requiring perfect calibration behavior."""
    tau = 0.35
    strikes = np.linspace(75.0, 130.0, 45)
    params = HestonParams(kappa=0.85, vbar=0.055, eta=1.15, rho=-0.94, v=0.025)
    calls = _call_slice(
        params=params,
        strikes=strikes,
        tau=tau,
        quad_cfg=EXTREME_BRANCH_GUARD_QUAD_CFG,
    )

    assert np.all(np.isfinite(calls))
    assert np.all(calls >= -1.0e-9)
    assert np.all(np.diff(calls) <= 1.0e-7)
    _assert_no_large_local_spike(calls, max_spike_ratio=3.50)
