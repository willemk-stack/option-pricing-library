from __future__ import annotations

import numpy as np
import pytest

import option_pricing.models.heston.fourier as heston_fourier
from option_pricing.models.black_scholes.bs import black76_call_price
from option_pricing.models.heston import HestonParams
from option_pricing.pricers.heston import heston_price_call_from_ctx
from option_pricing.types import MarketData, OptionSpec, OptionType
from option_pricing.vol.implied_vol_scalar import implied_vol_bs


def _ctx():
    return MarketData(spot=100.0, rate=0.02, dividend_yield=0.0).to_context()


def _deterministic_variance_black76_sigma(params: HestonParams, tau: float) -> float:
    tau = float(tau)
    integrated_variance = (
        params.vbar * tau
        + (params.v - params.vbar) * (1.0 - np.exp(-params.kappa * tau)) / params.kappa
    )
    return float(np.sqrt(integrated_variance / tau))


def _call_price_slice(
    *,
    strikes: np.ndarray,
    tau: float,
    params: HestonParams,
    backend: str = "gauss_legendre",
) -> np.ndarray:
    ctx = _ctx()
    return np.asarray(
        [
            heston_price_call_from_ctx(
                strike=float(strike),
                ctx=ctx,
                tau=tau,
                params=params,
                backend=backend,
            )
            for strike in strikes
        ],
        dtype=float,
    )


def _black76_call_slice(
    *,
    strikes: np.ndarray,
    tau: float,
    sigma: float,
) -> np.ndarray:
    ctx = _ctx()
    forward = ctx.fwd(tau)
    df = ctx.df(tau)
    return np.asarray(
        [
            black76_call_price(
                forward=forward,
                strike=float(strike),
                sigma=sigma,
                tau=tau,
                df=df,
            )
            for strike in strikes
        ],
        dtype=float,
    )


def _implied_vol_slice(
    *,
    strikes: np.ndarray,
    tau: float,
    prices: np.ndarray,
) -> np.ndarray:
    market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
    return np.asarray(
        [
            implied_vol_bs(
                float(price),
                OptionSpec(kind=OptionType.CALL, strike=float(strike), expiry=tau),
                market,
            )
            for strike, price in zip(strikes, prices, strict=True)
        ],
        dtype=float,
    )


def _format_limit_diagnostics(
    *,
    strikes: np.ndarray,
    heston_prices: np.ndarray,
    black_prices: np.ndarray,
    params: HestonParams,
    tau: float,
) -> str:
    errors = heston_prices - black_prices
    worst_idx = int(np.argmax(np.abs(errors)))
    rows = [
        f"tau={tau:.6f}",
        f"params={params}",
        (
            "max_abs_error="
            f"{float(np.max(np.abs(errors))):.12e} at strike={float(strikes[worst_idx]):.6f}"
        ),
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


def _format_skew_diagnostics(
    *,
    strikes: np.ndarray,
    low_corr_ivs: np.ndarray,
    neg_corr_ivs: np.ndarray,
    params_low_corr: HestonParams,
    params_neg_corr: HestonParams,
) -> str:
    low_corr_wing_spread = float(low_corr_ivs[0] - low_corr_ivs[-1])
    neg_corr_wing_spread = float(neg_corr_ivs[0] - neg_corr_ivs[-1])
    rows = [
        f"params_low_corr={params_low_corr}",
        f"params_neg_corr={params_neg_corr}",
        f"low_corr_wing_spread={low_corr_wing_spread:.12e}",
        f"neg_corr_wing_spread={neg_corr_wing_spread:.12e}",
        "",
        "strike | iv_low_corr | iv_neg_corr | diff",
    ]
    for strike, low_corr_iv, neg_corr_iv in zip(
        strikes, low_corr_ivs, neg_corr_ivs, strict=True
    ):
        rows.append(
            f"{float(strike):7.3f} | {float(low_corr_iv): .12f} |"
            f" {float(neg_corr_iv): .12f} |"
            f" {float(low_corr_iv - neg_corr_iv): .12e}"
        )
    return "\n".join(rows)


def _format_refinement_diagnostics(
    *,
    strikes: np.ndarray,
    coarse_prices: np.ndarray,
    refined_prices: np.ndarray,
    extra_refined_prices: np.ndarray,
    params: HestonParams,
    tau: float,
) -> str:
    coarse_to_refined = coarse_prices - refined_prices
    refined_to_extra_refined = refined_prices - extra_refined_prices
    rows = [
        f"tau={tau:.6f}",
        f"params={params}",
        ("max_abs(coarse-refined)=" f"{float(np.max(np.abs(coarse_to_refined))):.12e}"),
        (
            "max_abs(refined-extra_refined)="
            f"{float(np.max(np.abs(refined_to_extra_refined))):.12e}"
        ),
        "",
        "strike | coarse | refined | extra_refined | coarse-refined | refined-extra",
    ]
    for strike, coarse, refined, extra_refined, coarse_diff, refined_diff in zip(
        strikes,
        coarse_prices,
        refined_prices,
        extra_refined_prices,
        coarse_to_refined,
        refined_to_extra_refined,
        strict=True,
    ):
        rows.append(
            f"{float(strike):7.3f} | {float(coarse): .12f} |"
            f" {float(refined): .12f} | {float(extra_refined): .12f} |"
            f" {float(coarse_diff): .12e} | {float(refined_diff): .12e}"
        )
    return "\n".join(rows)


def _call_price_slice_with_quad_settings(
    *,
    strikes: np.ndarray,
    tau: float,
    params: HestonParams,
    quad_kwargs: dict[str, float | int],
) -> np.ndarray:
    original_quad = heston_fourier.quad

    def _quad_with_overrides(func, a, b, args=(), complex_func=False):
        return original_quad(
            func,
            a=a,
            b=b,
            args=args,
            complex_func=complex_func,
            **quad_kwargs,
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(heston_fourier, "quad", _quad_with_overrides)
        return _call_price_slice(
            strikes=strikes,
            tau=tau,
            params=params,
            backend="quad",
        )


def test_heston_very_small_vol_of_vol_limit_is_close_to_black76() -> None:
    """Smoke-test a small-eta regime against its Black-76 deterministic limit."""
    tau = 1.0
    strikes = np.linspace(85.0, 115.0, 7)
    params = HestonParams(
        kappa=3.0,
        vbar=0.04,
        eta=1e-4,
        rho=-0.35,
        v=0.07,
    )

    heston_prices = _call_price_slice(strikes=strikes, tau=tau, params=params)
    black_prices = _black76_call_slice(
        strikes=strikes,
        tau=tau,
        sigma=_deterministic_variance_black76_sigma(params, tau),
    )

    max_abs_error = float(np.max(np.abs(heston_prices - black_prices)))

    assert max_abs_error < 5e-4, (
        "Very small vol-of-vol regime should stay close to Black-76.\n"
        + _format_limit_diagnostics(
            strikes=strikes,
            heston_prices=heston_prices,
            black_prices=black_prices,
            params=params,
            tau=tau,
        )
    )


def test_heston_low_correlation_regime_has_milder_skew_than_negative_correlation() -> (
    None
):
    """Smoke-test that low correlation produces a visibly milder downside skew."""
    tau = 1.0
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)
    base = dict(kappa=2.0, vbar=0.04, eta=0.70, v=0.05)
    params_low_corr = HestonParams(rho=-0.05, **base)
    params_neg_corr = HestonParams(rho=-0.75, **base)

    low_corr_prices = _call_price_slice(
        strikes=strikes,
        tau=tau,
        params=params_low_corr,
    )
    neg_corr_prices = _call_price_slice(
        strikes=strikes,
        tau=tau,
        params=params_neg_corr,
    )

    low_corr_ivs = _implied_vol_slice(strikes=strikes, tau=tau, prices=low_corr_prices)
    neg_corr_ivs = _implied_vol_slice(strikes=strikes, tau=tau, prices=neg_corr_prices)

    low_corr_wing_spread = float(low_corr_ivs[0] - low_corr_ivs[-1])
    neg_corr_wing_spread = float(neg_corr_ivs[0] - neg_corr_ivs[-1])

    assert np.all(np.isfinite(low_corr_ivs))
    assert np.all(np.isfinite(neg_corr_ivs))
    assert abs(low_corr_wing_spread) < abs(neg_corr_wing_spread), (
        "Low-correlation Heston smile should have milder wing asymmetry than a"
        " strongly negatively correlated smile.\n"
        + _format_skew_diagnostics(
            strikes=strikes,
            low_corr_ivs=low_corr_ivs,
            neg_corr_ivs=neg_corr_ivs,
            params_low_corr=params_low_corr,
            params_neg_corr=params_neg_corr,
        )
    )


def test_heston_prices_stabilize_when_quad_settings_are_tightened() -> None:
    """Iteration target: tighter integration settings should reduce price drift."""
    tau = 1.0
    strikes = np.linspace(80.0, 120.0, 9)
    params = HestonParams(
        kappa=100.0,
        vbar=0.04,
        eta=1e-6,
        rho=0.0,
        v=0.04,
    )

    coarse_prices = _call_price_slice_with_quad_settings(
        strikes=strikes,
        tau=tau,
        params=params,
        quad_kwargs={"limit": 50, "epsabs": 1e-6, "epsrel": 1e-6},
    )
    refined_prices = _call_price_slice_with_quad_settings(
        strikes=strikes,
        tau=tau,
        params=params,
        quad_kwargs={"limit": 150, "epsabs": 1e-10, "epsrel": 1e-10},
    )
    extra_refined_prices = _call_price_slice_with_quad_settings(
        strikes=strikes,
        tau=tau,
        params=params,
        quad_kwargs={"limit": 300, "epsabs": 1e-12, "epsrel": 1e-12},
    )

    coarse_to_refined = float(np.max(np.abs(coarse_prices - refined_prices)))
    refined_to_extra_refined = float(
        np.max(np.abs(refined_prices - extra_refined_prices))
    )

    assert refined_to_extra_refined < coarse_to_refined, (
        "Tightening quad settings should stabilize Heston prices.\n"
        + _format_refinement_diagnostics(
            strikes=strikes,
            coarse_prices=coarse_prices,
            refined_prices=refined_prices,
            extra_refined_prices=extra_refined_prices,
            params=params,
            tau=tau,
        )
    )
