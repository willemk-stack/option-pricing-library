from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SVIRegConfig:
    lambda_m: float = 0.5
    m_prior: float = 0.0
    m_scale: float = 0.2

    lambda_inv_sigma: float = 2.0
    sigma_floor: float = 0.03

    lambda_slope_L: float = 0.75
    lambda_slope_R: float = 0.75
    slope_cap: float = 1.999  # Lee bound cap for total variance slope

    slope_floor: float = 0.10
    slope_denom: float = 0.15  # overwritten per smile

    lambda_g: float = 1e5
    g_scale: float = 0.02
    g_floor: float = 0.0
    g_n_grid: int = 81  # optional if you use the data-based n above


class SVIRegOverrideDict(TypedDict, total=False):
    lambda_m: float
    m_prior: float
    m_scale: float
    lambda_inv_sigma: float
    sigma_floor: float
    lambda_slope_L: float
    lambda_slope_R: float
    slope_cap: float
    slope_floor: float
    slope_denom: float
    lambda_g: float
    g_scale: float
    g_floor: float
    g_n_grid: int


RegOverride = SVIRegOverrideDict | Callable[[SVIRegConfig], SVIRegConfig]


def default_reg_from_data(
    y: NDArray[np.float64],
    w_obs: NDArray[np.float64],
    sqrt_w: NDArray[np.float64] | None = None,
) -> SVIRegConfig:
    y_sorted = np.unique(np.sort(y))
    dy = np.diff(y_sorted)
    dy_med = float(np.median(dy)) if dy.size else 0.1

    n = int(y.size)

    n_factor = float(np.clip(10.0 / max(n, 1), 0.25, 4.0))

    rail_factor = float(np.clip(30.0 / max(n, 1), 1.0, 10.0))
    m_prior = float(y[int(np.argmin(w_obs))])

    q05, q95 = np.quantile(y, [0.05, 0.95])
    y_span = float(max(q95 - q05, 1e-6))
    m_scale = float(max(0.25 * y_span, 0.10))

    sigma_floor = float(max(0.03, 1.5 * dy_med))

    slope_denom = 0.15

    well_determined = (n >= 5000) and (y_sorted[0] < -0.3) and (y_sorted[-1] > 0.3)

    return SVIRegConfig(
        lambda_m=0.0 if well_determined else 0.5 * n_factor,
        m_prior=m_prior,
        m_scale=m_scale,
        lambda_inv_sigma=0.0 if well_determined else 10.0 * rail_factor,
        sigma_floor=sigma_floor,
        lambda_slope_L=0.0 if well_determined else 0.25 * n_factor,
        lambda_slope_R=0.0 if well_determined else 0.25 * n_factor,
        slope_cap=1.999,
        slope_denom=slope_denom,
    )


def apply_reg_override(
    base: SVIRegConfig, override: RegOverride | None
) -> SVIRegConfig:
    if override is None:
        return base
    if callable(override):
        out = override(base)
        if not isinstance(out, SVIRegConfig):
            raise TypeError("reg_override callable must return SVIRegConfig")
        return out
    return replace(base, **override)


def _robust_rhoprime(z: NDArray[np.float64], loss: str) -> NDArray[np.float64]:
    """
    rho'(z) for SciPy least_squares robust losses, where z = (r / f_scale)^2.

    In IRLS we use weights w_i = rho'(z_i) and solve a weighted LS step.
    """
    z = np.asarray(z, dtype=np.float64)
    if loss == "linear":
        return np.ones_like(z)

    if loss == "soft_l1":
        # rho(z) = 2*(sqrt(1+z)-1)  => rho'(z) = 1/sqrt(1+z)
        return 1.0 / np.sqrt(1.0 + z)

    if loss == "huber":
        # rho(z)=z if z<=1 else 2*sqrt(z)-1  => rho'(z)=1 if z<=1 else 1/sqrt(z)
        w = np.ones_like(z)
        big = z > 1.0
        w[big] = 1.0 / np.sqrt(z[big])
        return w

    if loss == "cauchy":
        # rho(z)=ln(1+z) => rho'(z)=1/(1+z)
        return 1.0 / (1.0 + z)

    if loss == "arctan":
        # rho(z)=arctan(z) => rho'(z)=1/(1+z^2)
        return 1.0 / (1.0 + z * z)

    raise ValueError(f"Unknown loss: {loss}")


def soft_hinge_log_ratio(ratio: float) -> float:
    # ``softplus`` lives in the transforms module, not math; import locally to
    # avoid a cyclic dependency and reduce startup cost when this helper is
    # rarely used.
    from .math import EPS, LOG2
    from .transforms import softplus

    x = float(np.log(max(ratio, EPS)))  # x > 0 iff sigma < floor
    h = float(softplus(x) - LOG2)  # smooth around 0
    return max(h, 0.0)  # true hinge (piecewise)


def soft_hinge_ratio_excess(ratio: float) -> float:
    from .transforms import softplus

    return float(softplus(ratio - 1.0))
