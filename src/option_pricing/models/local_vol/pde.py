from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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

    def _sigma2(x: ArrayLike, tau: float) -> FloatArray:
        x_arr = np.asarray(x, dtype=float)
        tau_eff = max(float(tau), float(tau_floor))

        # solver x -> underlying S
        S = to_S(x_arr)
        sig2 = np.asarray(local_var(S, tau_eff), dtype=float)
        sig2 = np.broadcast_to(sig2, np.asarray(S, dtype=float).shape)

        # sanitize
        sig2 = np.where(np.isfinite(sig2), sig2, float(sigma2_floor))
        sig2 = np.maximum(sig2, float(sigma2_floor))
        if sigma2_cap is not None:
            sig2 = np.minimum(sig2, float(sigma2_cap))
        return np.asarray(sig2, dtype=float)

    ck = _coord_key(coord)

    if ck == "LOG_S":

        def a(x: ArrayLike, tau: float) -> FloatArray:
            return 0.5 * _sigma2(x, tau)

        def b(x: ArrayLike, tau: float) -> FloatArray:
            sig2 = _sigma2(x, tau)
            x_arr = np.asarray(x, dtype=float)
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
