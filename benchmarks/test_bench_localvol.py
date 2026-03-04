from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.black_scholes import bs as bs_model
from option_pricing.types import MarketData
from option_pricing.vol.local_vol_dupire import local_vol_from_call_grid_diagnostics
from option_pricing.vol.svi import SVIParams, calibrate_svi, svi_total_variance


def _make_call_grid(
    *,
    strikes: np.ndarray,
    taus: np.ndarray,
    spot: float,
    r: float,
    q: float,
    sigma: float,
) -> np.ndarray:
    rows = []
    for tau in taus:
        row = [
            bs_model.call_price(
                spot=spot, strike=float(K), r=r, q=q, sigma=sigma, tau=float(tau)
            )
            for K in strikes
        ]
        rows.append(row)
    return np.asarray(rows, dtype=float)


@pytest.mark.parametrize("nT,nK", [(31, 61), (61, 121), (121, 241)])
def test_bench_local_vol_dupire_scaling(benchmark, nT: int, nK: int) -> None:
    market = MarketData(spot=100.0, rate=0.01, dividend_yield=0.0)
    strikes = np.linspace(60.0, 140.0, nK, dtype=float)
    taus = np.linspace(0.1, 2.0, nT, dtype=float)

    call = _make_call_grid(
        strikes=strikes,
        taus=taus,
        spot=market.spot,
        r=market.rate,
        q=market.dividend_yield,
        sigma=0.2,
    )

    def _run() -> object:
        report = local_vol_from_call_grid_diagnostics(
            call=call, strikes=strikes, taus=taus, market=market
        )
        _ = float(np.mean(report.invalid))
        return report

    benchmark(_run)


@pytest.mark.parametrize("nK", [41, 81])
def test_bench_svi_slice_fit(benchmark, nK: int) -> None:
    y = np.linspace(-0.5, 0.5, nK, dtype=float)
    true_params = SVIParams(a=0.02, b=0.3, rho=-0.4, m=0.0, sigma=0.4)
    w_obs = svi_total_variance(y, true_params)

    def _run() -> object:
        fit = calibrate_svi(
            y=y,
            w_obs=w_obs,
            x0=true_params,
            loss="linear",
            slice_T=1.0,
            irls_max_outer=0,
            repair_butterfly=False,
        )
        return fit

    benchmark(_run)
