"""High-level Heston simulation entry points."""

from dataclasses import dataclass

import numpy as np

from ....monte_carlo import (
    MCConfig,
    correlated_normals,
    make_rng,
    standard_normals,
)
from ....types import PricingContext
from ....typing import FloatArray, FloatDType
from ..params import HestonParams
from .euler import simulate_heston_euler_paths
from .qe import simulate_heston_qe_paths
from .types import HestonScheme


def _qe_uniforms(
    *,
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    antithetic: bool,
) -> FloatArray:
    draw_paths = n_paths // 2 if antithetic else n_paths
    uniforms = np.asarray(
        rng.random((draw_paths, n_steps)),
        dtype=FloatDType,
    )

    if not antithetic:
        return uniforms

    upper = np.nextafter(np.float64(1.0), np.float64(0.0))
    antithetic_uniforms = np.minimum(1.0 - uniforms, upper)
    return np.concatenate([uniforms, antithetic_uniforms], axis=0)


def _qe_shocks(
    *,
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    antithetic: bool,
) -> FloatArray:
    z_v = standard_normals(
        rng,
        n_paths=n_paths,
        sample_shape=(n_steps,),
        antithetic=antithetic,
        dtype=np.dtype(FloatDType),
    )
    u_v = _qe_uniforms(
        rng=rng,
        n_paths=n_paths,
        n_steps=n_steps,
        antithetic=antithetic,
    )
    z_x = standard_normals(
        rng,
        n_paths=n_paths,
        sample_shape=(n_steps,),
        antithetic=antithetic,
        dtype=np.dtype(FloatDType),
    )
    return np.stack([z_v, u_v, z_x], axis=-1)


@dataclass(frozen=True, slots=True)
class HestonPathSimulator:
    params: HestonParams
    n_steps: int
    scheme: HestonScheme = "quadratic_exponential"

    def simulate_paths(
        self,
        *,
        ctx: PricingContext,
        tau: float,
        cfg: MCConfig,
    ) -> FloatArray:
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")

        time_grid = np.linspace(
            0.0,
            float(tau),
            int(self.n_steps) + 1,
            dtype=FloatDType,
        )

        fwd_grid = np.array(
            [ctx.fwd(float(t)) for t in time_grid],
            dtype=FloatDType,
        )

        if np.any(fwd_grid <= 0.0):
            raise ValueError("forward curve must be positive on the time grid")

        log_drift_increments = np.diff(np.log(fwd_grid))

        rng = make_rng(cfg)
        n_steps = int(self.n_steps)
        n_paths = int(cfg.n_paths)
        antithetic = bool(cfg.antithetic)

        if self.scheme == "euler_full_truncation":
            shocks = correlated_normals(
                rng,
                n_paths=n_paths,
                sample_shape=(n_steps,),
                corr=np.array(
                    [[1.0, self.params.rho], [self.params.rho, 1.0]],
                    dtype=FloatDType,
                ),
                antithetic=antithetic,
                dtype=np.dtype(FloatDType),
            )

            result = simulate_heston_euler_paths(
                params=self.params,
                x0=float(ctx.spot),
                tau=float(tau),
                n_steps=n_steps,
                shocks=shocks,
                log_drift_increments=log_drift_increments,
            )
        elif self.scheme == "quadratic_exponential":
            shocks = _qe_shocks(
                rng=rng,
                n_paths=n_paths,
                n_steps=n_steps,
                antithetic=antithetic,
            )

            result = simulate_heston_qe_paths(
                params=self.params,
                x0=float(ctx.spot),
                tau=float(tau),
                n_steps=n_steps,
                shocks=shocks,
                log_drift_increments=log_drift_increments,
            )
        else:
            raise ValueError(f"unsupported Heston scheme: {self.scheme}")

        return result.spot_paths


@dataclass(frozen=True, slots=True)
class HestonTerminalSimulator:
    params: HestonParams
    n_steps: int
    scheme: HestonScheme = "quadratic_exponential"

    def simulate_terminal(
        self,
        *,
        ctx: PricingContext,
        tau: float,
        cfg: MCConfig,
    ) -> FloatArray:
        paths = HestonPathSimulator(
            params=self.params,
            n_steps=self.n_steps,
            scheme=self.scheme,
        ).simulate_paths(ctx=ctx, tau=tau, cfg=cfg)
        return paths[:, -1]


def simulate_heston_paths(
    *,
    ctx: PricingContext,
    tau: float,
    params: HestonParams,
    n_steps: int,
    cfg: MCConfig,
    scheme: HestonScheme = "quadratic_exponential",
) -> FloatArray:
    """Simulate Heston spot paths on a pricing-context time horizon."""
    return HestonPathSimulator(
        params=params,
        n_steps=int(n_steps),
        scheme=scheme,
    ).simulate_paths(ctx=ctx, tau=tau, cfg=cfg)


def simulate_heston_terminal(
    *,
    ctx: PricingContext,
    tau: float,
    params: HestonParams,
    n_steps: int,
    cfg: MCConfig,
    scheme: HestonScheme = "quadratic_exponential",
) -> FloatArray:
    """Simulate Heston terminal spots on a pricing-context time horizon."""
    return HestonTerminalSimulator(
        params=params,
        n_steps=int(n_steps),
        scheme=scheme,
    ).simulate_terminal(ctx=ctx, tau=tau, cfg=cfg)
