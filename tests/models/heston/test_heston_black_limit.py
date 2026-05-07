from __future__ import annotations

import numpy as np

from option_pricing.models.black_scholes.bs import black76_call_price
from option_pricing.models.heston import HestonParams
from option_pricing.models.heston.charfunc import HESTON_ETA_DETERMINISTIC_THRESHOLD
from option_pricing.numerics.quadrature import QuadratureConfig
from option_pricing.pricers.heston import (
    heston_price_call_from_ctx,
    heston_price_put_from_ctx,
)
from option_pricing.types import MarketData, OptionSpec, OptionType
from option_pricing.vol.implied_vol_scalar import implied_vol_bs


def _deterministic_variance_black76_sigma(params: HestonParams, tau: float) -> float:
    """Equivalent Black-76 vol for the eta=0 deterministic-variance Heston limit.

    When eta = 0, the variance path is deterministic and the Heston call price
    should collapse to a Black-76 call price with total variance equal to the
    time-average of v(t).
    """
    tau = float(tau)
    integrated_variance = (
        params.vbar * tau
        + (params.v - params.vbar) * (1.0 - np.exp(-params.kappa * tau)) / params.kappa
    )
    return float(np.sqrt(integrated_variance / tau))


def _format_limit_diagnostics(
    *,
    strikes: np.ndarray,
    heston_prices: np.ndarray,
    black_prices: np.ndarray,
    sigma_eq: float,
    params: HestonParams,
    tau: float,
) -> str:
    errors = heston_prices - black_prices
    max_abs_error = float(np.max(np.abs(errors)))
    worst_idx = int(np.argmax(np.abs(errors)))
    rows = [
        f"tau={tau:.6f}",
        f"params={params}",
        f"sigma_eq={sigma_eq:.12f}",
        f"max_abs_error={max_abs_error:.12e} at strike={float(strikes[worst_idx]):.6f}",
        "",
        "strike | heston | black76 | diff",
    ]
    for strike, heston_px, black_px, diff in zip(
        strikes, heston_prices, black_prices, errors, strict=True
    ):
        rows.append(
            f"{float(strike):7.3f} | {float(heston_px): .12f} |"
            f" {float(black_px): .12f} | {float(diff): .12e}"
        )
    return "\n".join(rows)


def test_heston_sprint1_single_maturity_smile_smoke() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    ctx = market.to_context()

    tau = 1.0
    strikes = np.linspace(70.0, 130.0, 25)
    params = HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)

    call_prices: list[float] = []
    put_prices: list[float] = []
    implied_vols: list[float] = []

    for strike in strikes:
        strike = float(strike)
        call_price = heston_price_call_from_ctx(
            strike=strike, ctx=ctx, tau=tau, params=params
        )
        put_price = heston_price_put_from_ctx(
            strike=strike, tau=tau, ctx=ctx, params=params
        )
        iv = implied_vol_bs(
            call_price,
            OptionSpec(kind=OptionType.CALL, strike=strike, expiry=tau),
            market,
        )

        call_prices.append(float(call_price))
        put_prices.append(float(put_price))
        implied_vols.append(float(iv))

    call_prices_arr = np.asarray(call_prices, dtype=float)
    put_prices_arr = np.asarray(put_prices, dtype=float)
    strikes_arr = np.asarray(strikes, dtype=float)
    implied_vols_arr = np.asarray(implied_vols, dtype=float)

    forward = ctx.fwd(tau)
    df = ctx.df(tau)
    parity_error = (call_prices_arr - put_prices_arr) - df * (forward - strikes_arr)

    assert np.all(call_prices_arr >= -1e-12)
    assert np.all(np.diff(call_prices_arr) <= 1e-10)
    assert np.all(np.isfinite(implied_vols_arr))
    assert float(np.max(np.abs(parity_error))) < 1e-8


def test_heston_exact_deterministic_variance_limit_matches_black76() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    ctx = market.to_context()

    tau = 1.0
    forward = ctx.fwd(tau)
    df = ctx.df(tau)
    strikes = np.linspace(80.0, 120.0, 9)

    params = HestonParams(
        kappa=3.5,
        vbar=0.04,
        eta=0.0,
        rho=-0.5,
        v=0.09,
    )
    sigma_eq = _deterministic_variance_black76_sigma(params, tau)

    heston_prices = np.asarray(
        [
            heston_price_call_from_ctx(
                strike=float(strike), ctx=ctx, tau=tau, params=params
            )
            for strike in strikes
        ],
        dtype=float,
    )
    black_prices = np.asarray(
        [
            black76_call_price(
                forward=forward,
                strike=float(strike),
                sigma=sigma_eq,
                tau=tau,
                df=df,
            )
            for strike in strikes
        ],
        dtype=float,
    )

    assert np.allclose(heston_prices, black_prices, atol=1e-10, rtol=0.0), (
        "Exact deterministic-variance limit should agree with Black-76.\n"
        + _format_limit_diagnostics(
            strikes=strikes,
            heston_prices=heston_prices,
            black_prices=black_prices,
            sigma_eq=sigma_eq,
            params=params,
            tau=tau,
        )
    )


def test_heston_near_deterministic_variance_limit_is_close_to_black76() -> None:
    """Iteration target: this should pass once the limiting regime is stable.

    The regime uses very small vol-of-vol and very fast mean reversion so the
    variance dynamics are *almost* deterministic. The current implementation is
    still visibly off here, which is exactly what this test is meant to surface.
    """
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    ctx = market.to_context()

    tau = 1.0
    forward = ctx.fwd(tau)
    df = ctx.df(tau)
    strikes = np.linspace(80.0, 120.0, 9)

    params = HestonParams(
        kappa=100.0,
        vbar=0.04,
        eta=1e-6,
        rho=0.0,
        v=0.04,
    )
    sigma_eq = _deterministic_variance_black76_sigma(params, tau)

    heston_prices = np.asarray(
        [
            heston_price_call_from_ctx(
                strike=float(strike), ctx=ctx, tau=tau, params=params
            )
            for strike in strikes
        ],
        dtype=float,
    )
    black_prices = np.asarray(
        [
            black76_call_price(
                forward=forward,
                strike=float(strike),
                sigma=sigma_eq,
                tau=tau,
                df=df,
            )
            for strike in strikes
        ],
        dtype=float,
    )

    max_abs_error = float(np.max(np.abs(heston_prices - black_prices)))

    assert max_abs_error < 1e-3, (
        "Near-deterministic variance limit should be close to Black-76.\n"
        + _format_limit_diagnostics(
            strikes=strikes,
            heston_prices=heston_prices,
            black_prices=black_prices,
            sigma_eq=sigma_eq,
            params=params,
            tau=tau,
        )
    )


def test_heston_eta_threshold_prices_converge_smoothly_to_deterministic_limit() -> None:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.005)
    ctx = market.to_context()
    tau = 1.25
    strikes = np.array([80.0, 100.0, 120.0], dtype=float)
    eta_grid = np.array([1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8], dtype=float)
    quad_cfg = QuadratureConfig(u_max=240.0, n_panels=48, nodes_per_panel=24)

    def params_for_eta(eta: float) -> HestonParams:
        return HestonParams(
            kappa=1.7,
            vbar=0.04,
            eta=float(eta),
            rho=-0.55,
            v=0.06,
        )

    call_prices = np.asarray(
        [
            heston_price_call_from_ctx(
                strike=strikes,
                ctx=ctx,
                tau=tau,
                params=params_for_eta(float(eta)),
                quad_cfg=quad_cfg,
            )
            for eta in eta_grid
        ],
        dtype=float,
    )
    put_prices = np.asarray(
        [
            heston_price_put_from_ctx(
                strike=strikes,
                ctx=ctx,
                tau=tau,
                params=params_for_eta(float(eta)),
                quad_cfg=quad_cfg,
            )
            for eta in eta_grid
        ],
        dtype=float,
    )
    deterministic_calls = np.asarray(
        heston_price_call_from_ctx(
            strike=strikes,
            ctx=ctx,
            tau=tau,
            params=params_for_eta(0.0),
            quad_cfg=quad_cfg,
        ),
        dtype=float,
    )

    call_diffs = np.max(np.abs(np.diff(call_prices, axis=0)), axis=1)

    assert HESTON_ETA_DETERMINISTIC_THRESHOLD == 1.0e-8
    assert np.all(np.isfinite(call_prices))
    assert np.all(np.isfinite(put_prices))
    assert call_diffs[1] < 0.05 * call_diffs[0]
    assert call_diffs[2] < 1.0e-3
    np.testing.assert_allclose(call_prices[-1], deterministic_calls, atol=1.0e-10)
