from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _weighted_linear_fit(
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    sqrt_weights: NDArray[np.float64],
) -> tuple[float, float]:
    """
    Fit w ~ c + s*y using sqrt-weights (i.e. minimize ||sqrt_w*(Xb - w)||_2).
    Returns (c, s).
    """
    y = np.asarray(y, np.float64).reshape(-1)
    w = np.asarray(w, np.float64).reshape(-1)
    sw = np.asarray(sqrt_weights, np.float64).reshape(-1)

    if y.shape != w.shape or y.shape != sw.shape:
        raise ValueError("y, w, sqrt_weights must have the same shape")

    X = np.column_stack([np.ones_like(y), y])  # (n, 2)
    WX = sw[:, None] * X
    Wy = sw * w
    beta, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
    return float(beta[0]), float(beta[1])


def estimate_wing_slopes_one_sided(
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    sqrt_weights: NDArray[np.float64] | None = None,
    *,
    wing_threshold: float = 0.30,
    q_tail: float | None = None,
    min_pts_cap: int = 12,
) -> tuple[float | None, float | None]:
    """
    Estimate observed wing slopes (sL_obs, sR_obs) by fitting a line to
    the left and right tails of (y, w).

    Tail regression points are restricted to those beyond +/- wing_threshold.
    """
    y = np.asarray(y, np.float64).reshape(-1)
    w = np.asarray(w, np.float64).reshape(-1)

    if y.shape != w.shape:
        raise ValueError("y and w must have the same shape")
    if y.size < 8:
        return None, None
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(w))):
        return None, None

    sw = (
        np.ones_like(w)
        if sqrt_weights is None
        else np.asarray(sqrt_weights, np.float64).reshape(-1)
    )
    if sw.shape != w.shape:
        raise ValueError("sqrt_weights must have the same shape as w")
    if not (np.all(np.isfinite(sw)) and np.all(sw >= 0.0)):
        return None, None

    n = int(y.size)

    # choose tail fraction adaptively if not provided
    if q_tail is None:
        if n < 20:
            q_tail = 0.25
        elif n < 35:
            q_tail = 0.20
        else:
            q_tail = 0.10
    q_tail = float(np.clip(q_tail, 0.05, 0.45))

    min_pts = int(np.clip(0.12 * n, 4, min_pts_cap))
    k = max(min_pts, int(np.ceil(q_tail * n)))

    sL_obs: float | None = None
    sR_obs: float | None = None

    # Left wing
    left_cand = np.where(y < -wing_threshold)[0]
    if left_cand.size >= min_pts:
        order = np.argsort(y[left_cand])  # most negative first
        left = left_cand[order[: min(k, left_cand.size)]]
        if float(np.sum(sw[left])) > 0.0:
            _, sL = _weighted_linear_fit(y[left], w[left], sw[left])
            if np.isfinite(sL):
                sL = float(sL)
                if sL <= 0.0:
                    sL_obs = sL

    # Right wing
    right_cand = np.where(y > wing_threshold)[0]
    if right_cand.size >= min_pts:
        order = np.argsort(y[right_cand])  # ascending
        right = right_cand[order[-min(k, right_cand.size) :]]
        if float(np.sum(sw[right])) > 0.0:
            _, sR = _weighted_linear_fit(y[right], w[right], sw[right])
            if np.isfinite(sR):
                sR = float(sR)
                if sR >= 0.0:
                    sR_obs = sR

    return sL_obs, sR_obs


def usable_obs_slopes(
    sL_obs: float | None,
    sR_obs: float | None,
    *,
    slope_cap: float,
) -> tuple[float | None, float | None]:
    """Apply sign sanity and Lee-cap clipping to observed slope targets."""
    cap = float(slope_cap)
    cap_eff = cap * 0.995

    sL = sL_obs
    sR = sR_obs

    if sR is not None:
        sR = float(sR)
        if not np.isfinite(sR) or sR < 0.0:
            sR = None
        else:
            sR = float(np.clip(sR, 0.0, cap_eff))

    if sL is not None:
        sL = float(sL)
        if not np.isfinite(sL) or sL > 0.0:
            sL = None
        else:
            mag = float(np.clip(-sL, 0.0, cap_eff))
            sL = -mag

    return sL, sR
