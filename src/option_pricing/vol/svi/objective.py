from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .math import (
    EPS,
    LOG2,
    gatheral_g_jac_params,
    gatheral_g_vec,
    svi_jac_wrt_params,
    svi_total_variance,
)
from .regularization import SVIRegConfig, soft_hinge_log_ratio
from .transforms import SVITransformLeeCap, sigmoid, softplus
from .wings import usable_obs_slopes


@dataclass
class SVIObjective:
    y: NDArray[np.float64]
    w_obs: NDArray[np.float64]
    sqrt_w: NDArray[np.float64]
    transform: SVITransformLeeCap
    reg: SVIRegConfig
    sL_obs: float | None = None
    sR_obs: float | None = None
    y_g: NDArray[np.float64] | None = None
    w_floor: float = 0.0

    def __post_init__(self):
        sw = np.asarray(self.sqrt_w, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(sw)) or np.any(sw < 0.0):
            raise ValueError("sqrt_weights must be finite and nonnegative")

    def residual(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        p = self.transform.decode(u)
        w_model = svi_total_variance(self.y, p)
        r = self.sqrt_w * (w_model - self.w_obs)

        reg_terms: list[NDArray[np.float64]] = []

        if self.reg.lambda_m > 0.0:
            val = (
                np.sqrt(self.reg.lambda_m) * (p.m - self.reg.m_prior) / self.reg.m_scale
            )
            reg_terms.append(np.array([val], dtype=np.float64))

        if self.reg.lambda_inv_sigma > 0.0:
            ratio = self.reg.sigma_floor / max(p.sigma, EPS)
            val = np.sqrt(self.reg.lambda_inv_sigma) * soft_hinge_log_ratio(ratio)
            reg_terms.append(np.array([val], dtype=np.float64))

        sL_use, sR_use = usable_obs_slopes(
            self.sL_obs, self.sR_obs, slope_cap=self.reg.slope_cap
        )
        cap = float(self.reg.slope_cap)
        s_min = float(self.transform.slope_min)
        span = cap - s_min

        if self.reg.lambda_slope_R > 0.0 and sR_use is not None:
            uR = float(u[1])
            sR_model = s_min + span * float(sigmoid(uR))
            val = (
                np.sqrt(self.reg.lambda_slope_R)
                * (sR_model - sR_use)
                / self.reg.slope_denom
            )
            reg_terms.append(np.array([val], dtype=np.float64))

        if self.reg.lambda_slope_L > 0.0 and sL_use is not None:
            uL = float(u[2])
            sL_model = -(s_min + span * float(sigmoid(uL)))
            val = (
                np.sqrt(self.reg.lambda_slope_L)
                * (sL_model - sL_use)
                / self.reg.slope_denom
            )
            reg_terms.append(np.array([val], dtype=np.float64))

        if self.reg.lambda_g > 0.0 and self.y_g is not None and self.y_g.size:
            g = gatheral_g_vec(self.y_g, p, w_floor=self.w_floor)
            deficit = (self.reg.g_floor - g) / max(self.reg.g_scale, 1e-12)
            h = softplus(deficit) - LOG2
            h = np.maximum(h, 0.0)
            r_g = (np.sqrt(self.reg.lambda_g) * h).astype(np.float64, copy=False)
            reg_terms.append(r_g)

        if reg_terms:
            return np.concatenate([r, np.concatenate(reg_terms)])
        return r

    def jac(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.asarray(u, dtype=np.float64).reshape(5)
        p = self.transform.decode(u)

        # ---- data block ----
        J_wp = svi_jac_wrt_params(self.y, p)  # (n,5) wrt [a,b,rho,m,sigma]
        dpdu = self.transform.dp_du(u, p)  # (5,5)
        J_wu = J_wp @ dpdu  # (n,5) wrt u
        J = self.sqrt_w[:, None] * J_wu  # (n,5)

        # Collect extra reg Jacobian blocks in the SAME order as residual()
        blocks: list[NDArray[np.float64]] = []

        # ---- m prior row (1,5) ----
        if self.reg.lambda_m > 0.0:
            row = np.zeros((1, 5), dtype=np.float64)
            row[0, 3] = np.sqrt(self.reg.lambda_m) / self.reg.m_scale
            blocks.append(row)

        # ---- inv-sigma hinge row (1,5) ----
        if self.reg.lambda_inv_sigma > 0.0:
            sigma = max(float(p.sigma), EPS)
            x = float(np.log(max(self.reg.sigma_floor / sigma, EPS)))
            h = float(softplus(x) - LOG2)

            row = np.zeros((1, 5), dtype=np.float64)
            if h > 0.0:
                dsoft_dx = float(sigmoid(x))
                dx_dsigma = -1.0 / sigma
                dsigma_dusig = float(sigmoid(float(u[4])))
                row[0, 4] = (
                    np.sqrt(self.reg.lambda_inv_sigma)
                    * dsoft_dx
                    * dx_dsigma
                    * dsigma_dusig
                )
            blocks.append(row)

        # ---- slope target rows (1,5) each ----
        denom = float(self.reg.slope_denom)
        sL_use, sR_use = usable_obs_slopes(
            self.sL_obs, self.sR_obs, slope_cap=self.reg.slope_cap
        )
        cap = float(self.reg.slope_cap)
        s_min = float(self.transform.slope_min)
        span = cap - s_min

        if self.reg.lambda_slope_R > 0.0 and sR_use is not None:
            uR = float(u[1])
            sigR = float(sigmoid(uR))
            dsR_duR = span * sigR * (1.0 - sigR)

            row = np.zeros((1, 5), dtype=np.float64)
            row[0, 1] = np.sqrt(self.reg.lambda_slope_R) * dsR_duR / denom
            blocks.append(row)

        if self.reg.lambda_slope_L > 0.0 and sL_use is not None:
            uL = float(u[2])
            sigL = float(sigmoid(uL))
            dsL_duL = -span * sigL * (1.0 - sigL)

            row = np.zeros((1, 5), dtype=np.float64)
            row[0, 2] = np.sqrt(self.reg.lambda_slope_L) * dsL_duL / denom
            blocks.append(row)

        # ---- g-penalty block (ng,5) ----
        if self.reg.lambda_g > 0.0 and self.y_g is not None and self.y_g.size:
            y_g = np.asarray(self.y_g, dtype=np.float64).reshape(-1)

            g = gatheral_g_vec(y_g, p, w_floor=self.w_floor)
            g_scale = max(float(self.reg.g_scale), 1e-12)
            deficit = (float(self.reg.g_floor) - g) / g_scale

            # hinge derivative: 0 when deficit<=0, else sigmoid(deficit)
            dh_ddef = np.zeros_like(deficit)
            active = deficit > 0.0
            dh_ddef[active] = sigmoid(deficit[active])

            Jg_p = gatheral_g_jac_params(y_g, p, w_floor=self.w_floor)  # (ng,5) wrt p
            Jg_u = Jg_p @ dpdu  # (ng,5) wrt u

            scale = -np.sqrt(self.reg.lambda_g) / g_scale
            Jg_rows = (scale * dh_ddef)[:, None] * Jg_u  # (ng,5)

            blocks.append(Jg_rows)

        # ---- final stack ----
        if blocks:
            J = np.vstack([J, np.vstack(blocks)])
        return J


def svi_residual_vector(
    u: NDArray[np.float64],
    *,
    y: NDArray[np.float64],
    w_obs: NDArray[np.float64],
    sqrt_w: NDArray[np.float64],
    transform: SVITransformLeeCap,
    reg: SVIRegConfig,
    sL_obs: float | None,
    sR_obs: float | None,
) -> NDArray[np.float64]:
    """
    Residual vector used by the optimizer:
      [ sqrt_w*(w_model - w_obs) , reg_terms... ]
    This is NOT the same as `err = w_model - w_obs` because:
      - it is weighted by sqrt_w
      - it appends regularization residuals
    """
    u = np.asarray(u, dtype=np.float64).reshape(5)
    p = transform.decode(u)

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w_obs = np.asarray(w_obs, dtype=np.float64).reshape(-1)
    sqrt_w = np.asarray(sqrt_w, dtype=np.float64).reshape(-1)

    w_model = svi_total_variance(y, p)
    r = sqrt_w * (w_model - w_obs)

    reg_terms: list[float] = []

    if reg.lambda_m > 0.0:
        reg_terms.append(np.sqrt(reg.lambda_m) * (p.m - reg.m_prior) / reg.m_scale)

    if reg.lambda_inv_sigma > 0.0:
        ratio = reg.sigma_floor / max(p.sigma, EPS)
        reg_terms.append(np.sqrt(reg.lambda_inv_sigma) * soft_hinge_log_ratio(ratio))

    sL_use, sR_use = usable_obs_slopes(sL_obs, sR_obs, slope_cap=reg.slope_cap)
    cap = float(reg.slope_cap)
    s_min = float(transform.slope_min)
    span = cap - s_min

    if reg.lambda_slope_R > 0.0 and sR_use is not None:
        uR = float(u[1])
        sR_model = s_min + span * float(sigmoid(uR))
        reg_terms.append(
            np.sqrt(reg.lambda_slope_R) * (sR_model - sR_use) / reg.slope_denom
        )

    if reg.lambda_slope_L > 0.0 and sL_use is not None:
        uL = float(u[2])
        sL_model = -(s_min + span * float(sigmoid(uL)))
        reg_terms.append(
            np.sqrt(reg.lambda_slope_L) * (sL_model - sL_use) / reg.slope_denom
        )

    if reg_terms:
        return np.concatenate([r, np.asarray(reg_terms, dtype=np.float64)])
    return r
