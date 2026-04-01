from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from option_pricing.numerics.grids import SpacingPolicy
from option_pricing.numerics.pde import AdvectionScheme
from option_pricing.numerics.pde.domain import Coord
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy
from option_pricing.pricers.pde_pricer import local_vol_price_pde_european
from option_pricing.types import MarketData, OptionSpec, OptionType, PricingInputs


@dataclass(frozen=True)
class _CalendarTimeLocalVol:
    spot_ref: float
    total_maturity: float
    reverse_time: bool = False

    def sigma(self, spot, calendar_time):
        spot_arr = np.asarray(spot, dtype=float)
        time_ratio = np.clip(
            np.asarray(calendar_time, dtype=float) / float(self.total_maturity),
            0.0,
            1.0,
        )
        regime = 1.0 / (1.0 + np.exp(-10.0 * (spot_arr / self.spot_ref - 1.0)))
        sigma = (
            0.12
            + 0.30 * time_ratio * regime
            + 0.10 * (1.0 - time_ratio) * (1.0 - regime)
        )
        return np.clip(sigma, 0.05, 0.65)

    def local_var(self, spot, tau):
        time_query = float(tau)
        if self.reverse_time:
            time_query = max(float(self.total_maturity) - time_query, 0.0)
        sigma = self.sigma(spot, time_query)
        return np.asarray(sigma * sigma, dtype=float)


def _mc_price_under_calendar_time_surface(
    *,
    lv: _CalendarTimeLocalVol,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    n_paths: int = 60_000,
    n_steps: int = 120,
    seed: int = 123,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    dt = float(T) / float(n_steps)
    sqrt_dt = math.sqrt(dt)

    half_paths = n_paths // 2
    z = rng.standard_normal((n_steps, half_paths))
    z = np.concatenate([z, -z], axis=1)

    S = np.full(z.shape[1], float(S0), dtype=float)
    drift = float(r - q)

    for step in range(n_steps):
        calendar_time = min(max(step * dt, 1.0e-8), float(T))
        sigma = lv.sigma(S, calendar_time)
        S *= np.exp((drift - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z[step])

    payoff = np.maximum(S - float(K), 0.0)
    discount = math.exp(-float(r) * float(T))
    price = discount * float(np.mean(payoff))
    se = discount * float(np.std(payoff, ddof=1) / math.sqrt(payoff.size))
    return price, se


def test_localvol_pde_calendar_time_mapping_matches_calendar_time_mc() -> None:
    S0 = 100.0
    K = 110.0
    T = 1.0
    r = 0.01
    q = 0.0
    sigma_for_bounds = 0.25

    market = MarketData(spot=S0, rate=r, dividend_yield=q)
    p = PricingInputs(
        spec=OptionSpec(kind=OptionType.CALL, strike=K, expiry=T),
        market=market,
        sigma=sigma_for_bounds,
        t=0.0,
    )
    domain_cfg = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=6.5,
        center="strike",
        spacing=SpacingPolicy.CLUSTERED,
        cluster_strength=2.0,
    )

    lv_calendar = _CalendarTimeLocalVol(spot_ref=S0, total_maturity=T)
    lv_remaining_tau = _CalendarTimeLocalVol(
        spot_ref=S0,
        total_maturity=T,
        reverse_time=True,
    )

    pde_calendar = float(
        local_vol_price_pde_european(
            p,
            lv=lv_calendar,
            coord=Coord.LOG_S,
            domain_cfg=domain_cfg,
            Nx=251,
            Nt=251,
            method="cn",
            advection=AdvectionScheme.CENTRAL,
            sigma_for_bounds=sigma_for_bounds,
        )
    )
    pde_remaining_tau = float(
        local_vol_price_pde_european(
            p,
            lv=lv_remaining_tau,
            coord=Coord.LOG_S,
            domain_cfg=domain_cfg,
            Nx=251,
            Nt=251,
            method="cn",
            advection=AdvectionScheme.CENTRAL,
            sigma_for_bounds=sigma_for_bounds,
        )
    )

    mc_price, mc_se = _mc_price_under_calendar_time_surface(
        lv=lv_calendar,
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
    )

    err_calendar = abs(pde_calendar - mc_price)
    err_remaining_tau = abs(pde_remaining_tau - mc_price)

    assert err_calendar <= max(0.12, 2.5 * mc_se)
    assert err_remaining_tau >= err_calendar + 0.20
    assert err_remaining_tau >= 4.0 * mc_se
