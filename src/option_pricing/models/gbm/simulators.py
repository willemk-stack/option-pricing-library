from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...market.curves import PricingContext
from ...monte_carlo.config import MCConfig
from ...monte_carlo.rng import make_rng, standard_normals
from ...typing import FloatArray, FloatDType
from .simulation import GBMParams, simulate_gbm_paths, simulate_gbm_terminal


def _validated_positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class GBMTerminalSimulator:
    sigma: float

    def simulate_terminal(
        self,
        *,
        ctx: PricingContext,
        tau: float,
        cfg: MCConfig,
    ) -> FloatArray:
        tau = _validated_positive_finite("tau", tau)
        sigma = _validated_positive_finite("sigma", self.sigma)

        rng = make_rng(cfg)
        normals = standard_normals(
            rng,
            n_paths=int(cfg.n_paths),
            antithetic=bool(cfg.antithetic),
            dtype=np.dtype(FloatDType),
        )
        params = GBMParams(
            spot=float(ctx.spot),
            drift=float(ctx.r_avg(tau)) - float(ctx.q_avg(tau)),
            sigma=sigma,
        )
        return simulate_gbm_terminal(params=params, tau=tau, normals=normals)


@dataclass(frozen=True, slots=True)
class GBMPathSimulator:
    sigma: float
    n_steps: int

    def simulate_paths(
        self,
        *,
        ctx: PricingContext,
        tau: float,
        cfg: MCConfig,
    ) -> FloatArray:
        tau = _validated_positive_finite("tau", tau)
        sigma = _validated_positive_finite("sigma", self.sigma)
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")

        time_grid = np.linspace(0.0, tau, int(self.n_steps) + 1, dtype=FloatDType)
        rng = make_rng(cfg)
        normals = standard_normals(
            rng,
            n_paths=int(cfg.n_paths),
            sample_shape=(int(self.n_steps),),
            antithetic=bool(cfg.antithetic),
            dtype=np.dtype(FloatDType),
        )
        params = GBMParams(
            spot=float(ctx.spot),
            drift=float(ctx.r_avg(tau)) - float(ctx.q_avg(tau)),
            sigma=sigma,
        )
        return simulate_gbm_paths(
            params=params,
            time_grid=time_grid,
            normals=normals,
        )
