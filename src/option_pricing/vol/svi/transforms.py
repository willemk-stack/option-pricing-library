from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .math import EPS

if TYPE_CHECKING:
    from .models import SVIParams


def sigmoid(x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    ex = np.exp(x_arr[~pos])
    out[~pos] = ex / (1.0 + ex)
    return float(out) if out.ndim == 0 else out


def softplus(x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)
    return float(out) if out.ndim == 0 else out


def softplus_inv(y: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    y_arr = np.asarray(y, dtype=np.float64)
    y_arr = np.maximum(y_arr, EPS)
    out = y_arr + np.log(-np.expm1(-y_arr))
    return float(out) if out.ndim == 0 else out


def logit(x: float) -> float:
    x = float(np.clip(x, 1e-12, 1.0 - 1e-12))
    return float(np.log(x) - np.log1p(-x))


class SVITransformLeeCap:
    """
    u = [ua, uR, uL, um, usig]

    Slopes (Lee-capped) with a strictly-positive floor:
      s(u) = s_min + (cap - s_min) * sigmoid(u)   in (s_min, cap)

      sR    = s(uR)
      sLmag = s(uL)

    Then:
      b   = 0.5*(sR + sLmag) + eps
      rho = (sR - sLmag)/(sR + sLmag + eps)

    a is encoded via alpha >= 0:
      a = alpha - b*sigma*sqrt(1-rho^2)
      alpha = softplus(ua) + eps
      sigma = softplus(usig) + eps
      m = um
    """

    slope_cap: float
    slope_min: float
    eps: float

    def __init__(self, slope_cap: float = 1.999) -> None:
        """Constructor storing cap, min, and epsilon.

        Parameters
        ----------
        slope_cap : float
            Maximum allowed Lee slope.  Must be greater than ``slope_min``.
        """
        self.slope_cap = float(slope_cap)
        # constants are stored on the instance for potential future
        # configuration flexibility; keep them aligned with their class
        # defaults so they remain documented above.
        self.slope_min = 1e-4
        self.eps = EPS

    def decode(self, u: NDArray[np.float64]) -> SVIParams:
        ua, uR, uL, um, usig = map(float, u)

        alpha = float(softplus(ua)) + self.eps
        sigma = float(softplus(usig)) + self.eps
        m = float(um)

        cap = float(self.slope_cap)
        s_min = float(self.slope_min)
        span = cap - s_min
        if span <= 0.0:
            raise ValueError("slope_cap must be > slope_min")

        sR = s_min + span * float(sigmoid(uR))
        sLmag = s_min + span * float(sigmoid(uL))

        b = 0.5 * (sR + sLmag) + self.eps
        denom = (sR + sLmag) + self.eps
        rho = (sR - sLmag) / denom  # in (-1, 1)

        k2 = max(1.0 - rho * rho, 0.0)
        a = alpha - b * sigma * float(np.sqrt(k2))

        from .models import SVIParams

        return SVIParams(a=float(a), b=float(b), rho=float(rho), m=m, sigma=sigma)

    def encode(self, p: SVIParams) -> NDArray[np.float64]:
        if p.sigma <= 0.0 or p.b <= 0.0:
            raise ValueError("encode expects b>0 and sigma>0")

        cap = float(self.slope_cap)
        s_min = float(self.slope_min)
        span = cap - s_min
        if span <= 0.0:
            raise ValueError("slope_cap must be > slope_min")

        # convert (b,rho) -> slopes
        sR = float(p.b) * (1.0 + float(p.rho))
        sLmag = float(p.b) * (1.0 - float(p.rho))

        tiny = 1e-10
        sR = float(np.clip(sR, s_min + tiny, cap - tiny))
        sLmag = float(np.clip(sLmag, s_min + tiny, cap - tiny))

        sigR = float(np.clip((sR - s_min) / span, 1e-12, 1.0 - 1e-12))
        sigL = float(np.clip((sLmag - s_min) / span, 1e-12, 1.0 - 1e-12))

        uR = logit(sigR)
        uL = logit(sigL)

        alpha = float(p.a + p.b * p.sigma * np.sqrt(max(1.0 - p.rho * p.rho, 0.0)))
        alpha = max(alpha, self.eps)
        ua = float(softplus_inv(alpha))

        um = float(p.m)
        usig = float(softplus_inv(max(float(p.sigma), self.eps)))

        return np.array([ua, uR, uL, um, usig], dtype=np.float64)

    def dp_du(self, u: NDArray[np.float64], p: SVIParams) -> NDArray[np.float64]:
        """
        dp/du as (5,5), p=[a,b,rho,m,sigma], u=[ua,uR,uL,um,usig]
        """
        ua, uR, uL, um, usig = map(float, u)

        cap = float(self.slope_cap)
        s_min = float(self.slope_min)
        span = cap - s_min
        if span <= 0.0:
            raise ValueError("slope_cap must be > slope_min")

        sigR = float(sigmoid(uR))
        sigL = float(sigmoid(uL))

        sR = s_min + span * sigR
        sLmag = s_min + span * sigL

        dsR_duR = span * sigR * (1.0 - sigR)
        dsL_duL = span * sigL * (1.0 - sigL)

        D = float(sR + sLmag + self.eps)

        db_duR = 0.5 * dsR_duR
        db_duL = 0.5 * dsL_duL

        # rho = (sR - sLmag)/D
        drho_dsR = (2.0 * sLmag + self.eps) / (D * D)
        drho_dsL = (-2.0 * sR - self.eps) / (D * D)
        drho_duR = drho_dsR * dsR_duR
        drho_duL = drho_dsL * dsL_duL

        dalpha_dua = float(sigmoid(ua))  # d softplus / d ua
        dsig_dusig = float(sigmoid(usig))  # d softplus / d usig

        rho = float(p.rho)
        b = float(p.b)
        sigma = float(p.sigma)

        k2 = max(1.0 - rho * rho, 0.0)
        k = float(np.sqrt(k2))
        k_safe = max(k, 1e-12)

        # a = alpha - b*sigma*k, dk/drho = -rho/k
        common_rho = b * sigma * (rho / k_safe)

        da_dua = dalpha_dua
        da_duR = -(db_duR * sigma * k) + common_rho * drho_duR
        da_duL = -(db_duL * sigma * k) + common_rho * drho_duL
        da_dusig = -(b * dsig_dusig * k)

        return np.array(
            [
                [da_dua, da_duR, da_duL, 0.0, da_dusig],  # a
                [0.0, db_duR, db_duL, 0.0, 0.0],  # b
                [0.0, drho_duR, drho_duL, 0.0, 0.0],  # rho
                [0.0, 0.0, 0.0, 1.0, 0.0],  # m
                [0.0, 0.0, 0.0, 0.0, dsig_dusig],  # sigma
            ],
            dtype=np.float64,
        )
