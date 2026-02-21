from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ...models.black_scholes.pde import bs_coord_maps
from ...typing import ArrayLike, FloatArray

type CoordLike = str | Any  # accepts "S"/"LOG_S" or an Enum-like object

type CoordKey = str  # "S" | "LOG_S"

# (x, tau) -> FloatArray, broadcasted to x
CoeffFn = Callable[[ArrayLike, float], FloatArray]

# (S, tau) -> local variance sigma_loc^2
LocalVarFn = Callable[[ArrayLike, float], ArrayLike]


def _coord_key(coord: CoordLike) -> CoordKey:
    """Normalize coordinate spec (same behavior as BS helper)."""
    if hasattr(coord, "name"):
        key = str(coord.name)
    else:
        key = str(coord)

    k = key.strip().upper()
    if k in {"S"}:
        return "S"
    if k in {"LOG_S", "LOG", "LN_S", "LN"}:
        return "LOG_S"

    raise ValueError(f"Unsupported coord: {coord!r} (expected 'S' or 'LOG_S')")


@dataclass(frozen=True, slots=True)
class LocalVolPDECoeffs1D:
    """Coefficients for a 1D local-vol forward-in-time (tau) PDE.

    The PDE is expressed in solver coordinates ``x``:

        u_tau = a(x,tau) * u_xx + b(x,tau) * u_x + c(x,tau) * u
    """

    a: CoeffFn
    b: CoeffFn
    c: CoeffFn


def local_vol_pde_coeffs(
    *,
    coord: CoordLike,
    local_var: LocalVarFn,
    r: float,
    q: float,
    # guardrails
    tau_floor: float = 1e-8,
    sigma2_floor: float = 1e-14,
    sigma2_cap: float | None = None,
) -> LocalVolPDECoeffs1D:
    """Build local-vol PDE coefficients in the chosen coordinate.

    Parameters
    ----------
    coord:
        "S" or "LOG_S" (or an Enum-like object with .name).
    local_var:
        Callable returning *local variance* sigma_loc^2(S, tau).
        (For a Dupire/Gatheral surface, passing K=S is standard.)
    r, q:
        Flat short rate and dividend yield.

    Guardrails
    ----------
    tau_floor:
        Clamp tau to at least this value inside coefficient evaluation.
        Helpful if your local-vol surface disallows T<=0.
    sigma2_floor:
        Floors sigma^2 to keep the PDE uniformly parabolic.
    sigma2_cap:
        Optional cap on sigma^2 to prevent extreme coefficients from destabilizing.
    """

    rr = float(r)
    qq = float(q)
    mu = rr - qq

    # Use the same coord normalization / mapping as BS.
    to_x, to_S = bs_coord_maps(coord)

    # ---- caching ----
    # Your operator builder evaluates a(x,t) and b(x,t) by looping over x *scalars*.
    # A 1-element "last value" cache won't help there. Instead we cache sigma^2 for
    # all scalar x values within the same tau step.
    _cache_tau_eff: float | None = None
    _cache_sig2_by_x: dict[float, float] = {}

    # Also keep a fast-path for vector calls (if any future code evaluates coeffs
    # in a vectorized way).
    _last_tau_eff: float | None = None
    _last_x_arr: np.ndarray | None = None
    _last_sig2: FloatArray | None = None

    def _sanitize_sig2(sig2: np.ndarray) -> np.ndarray:
        sig2 = np.where(np.isfinite(sig2), sig2, float(sigma2_floor))
        sig2 = np.maximum(sig2, float(sigma2_floor))
        if sigma2_cap is not None:
            sig2 = np.minimum(sig2, float(sigma2_cap))
        return sig2

    def _sigma2(x: ArrayLike, tau: float) -> FloatArray:
        nonlocal _cache_tau_eff, _cache_sig2_by_x
        nonlocal _last_tau_eff, _last_x_arr, _last_sig2

        x_arr = np.asarray(x, dtype=float)
        tau_eff = max(float(tau), float(tau_floor))

        # If tau changed, reset the scalar cache (per-timestep cache).
        if _cache_tau_eff != tau_eff:
            _cache_tau_eff = tau_eff
            _cache_sig2_by_x.clear()

        # Scalar x path (this is what your PDE operator builder uses)
        if x_arr.ndim == 0:
            xi = float(x_arr.reshape(()))
            hit = _cache_sig2_by_x.get(xi)
            if hit is not None:
                return cast(FloatArray, np.asarray(hit, dtype=float))

            S = to_S(x_arr)  # keep as array-like for consistency
            sig2 = np.asarray(local_var(S, tau_eff), dtype=float)
            sig2 = _sanitize_sig2(sig2)
            val = float(np.asarray(sig2).reshape(()))

            _cache_sig2_by_x[xi] = val
            return cast(FloatArray, np.asarray(val, dtype=float))

        # Vector x fast-path (identity cache)
        if _last_sig2 is not None and _last_tau_eff == tau_eff and _last_x_arr is x_arr:
            return _last_sig2

        # Vector compute
        S = to_S(x_arr)
        sig2 = np.asarray(local_var(S, tau_eff), dtype=float)
        sig2 = np.broadcast_to(sig2, np.asarray(S, dtype=float).shape)
        sig2 = _sanitize_sig2(sig2)

        _last_tau_eff, _last_x_arr, _last_sig2 = tau_eff, x_arr, cast(FloatArray, sig2)
        return cast(FloatArray, sig2)

    ck = _coord_key(coord)

    if ck == "LOG_S":

        def a(x: ArrayLike, tau: float) -> FloatArray:
            # (optional) avoid double asarray in callers that already have ndarray
            return 0.5 * _sigma2(x, tau)

        def b(x: ArrayLike, tau: float) -> FloatArray:
            # micro-opt: compute x_arr once, pass it into _sigma2
            x_arr = np.asarray(x, dtype=float)
            sig2 = _sigma2(x_arr, tau)
            return (mu - 0.5 * sig2) + 0.0 * x_arr

        def c(x: ArrayLike, tau: float) -> FloatArray:
            x_arr = np.asarray(x, dtype=float)
            return (-rr) + 0.0 * x_arr

        return LocalVolPDECoeffs1D(a=a, b=b, c=c)

    # ck == "S"
    else:

        def a(x: ArrayLike, tau: float) -> FloatArray:
            x_arr = np.asarray(x, dtype=float)  # x is S
            sig2 = _sigma2(x_arr, tau)
            out = np.asarray(0.5 * sig2 * (x_arr * x_arr))
            return out

        def b(x: ArrayLike, tau: float) -> FloatArray:
            x_arr = np.asarray(x, dtype=float)
            return mu * x_arr

        def c(x: ArrayLike, tau: float) -> FloatArray:
            x_arr = np.asarray(x, dtype=float)
            return (-rr) + 0.0 * x_arr

        return LocalVolPDECoeffs1D(a=a, b=b, c=c)
