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
class _RemainingTauLocalVol:
    spot_ref: float
    total_maturity: float
    reverse_time: bool = False

    def sigma(self, spot, tau_remaining):
        spot_arr = np.asarray(spot, dtype=float)
        tau_ratio = np.clip(
            np.asarray(tau_remaining, dtype=float) / float(self.total_maturity),
            0.0,
            1.0,
        )
        regime = 1.0 / (1.0 + np.exp(-10.0 * (spot_arr / self.spot_ref - 1.0)))
        sigma = (
            0.12 + 0.30 * tau_ratio * regime + 0.10 * (1.0 - tau_ratio) * (1.0 - regime)
        )
        return np.clip(sigma, 0.05, 0.65)

    def local_var(self, spot, tau):
        tau_query = float(tau)
        if self.reverse_time:
            tau_query = max(float(self.total_maturity) - tau_query, 0.0)
        sigma = self.sigma(spot, tau_query)
        return np.asarray(sigma * sigma, dtype=float)


def _mc_price_under_remaining_tau_surface(
    *,
    lv: _RemainingTauLocalVol,
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
        tau_remaining = float(T) - step * dt
        sigma = lv.sigma(S, tau_remaining)
        S *= np.exp((drift - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z[step])

    payoff = np.maximum(S - float(K), 0.0)
    discount = math.exp(-float(r) * float(T))
    price = discount * float(np.mean(payoff))
    se = discount * float(np.std(payoff, ddof=1) / math.sqrt(payoff.size))
    return price, se


def test_localvol_pde_identity_time_mapping_matches_remaining_tau_mc() -> None:
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

    lv_identity = _RemainingTauLocalVol(spot_ref=S0, total_maturity=T)
    lv_reversed = _RemainingTauLocalVol(
        spot_ref=S0,
        total_maturity=T,
        reverse_time=True,
    )

    pde_identity = float(
        local_vol_price_pde_european(
            p,
            lv=lv_identity,
            coord=Coord.LOG_S,
            domain_cfg=domain_cfg,
            Nx=251,
            Nt=251,
            method="cn",
            advection=AdvectionScheme.CENTRAL,
            sigma_for_bounds=sigma_for_bounds,
        )
    )
    pde_reversed = float(
        local_vol_price_pde_european(
            p,
            lv=lv_reversed,
            coord=Coord.LOG_S,
            domain_cfg=domain_cfg,
            Nx=251,
            Nt=251,
            method="cn",
            advection=AdvectionScheme.CENTRAL,
            sigma_for_bounds=sigma_for_bounds,
        )
    )

    mc_price, mc_se = _mc_price_under_remaining_tau_surface(
        lv=lv_identity,
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
    )

    err_identity = abs(pde_identity - mc_price)
    err_reversed = abs(pde_reversed - mc_price)

    assert err_identity <= max(0.12, 2.5 * mc_se)
    assert err_reversed >= err_identity + 0.20
    assert err_reversed >= 4.0 * mc_se
