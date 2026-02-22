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
    # The PDE operator builder tries to evaluate coefficient functions on a whole
    # x-grid in one vectorized call. When that succeeds, we want to avoid
    # evaluating local_var twice per time step (once via a(x,t) and once via
    # b(x,t)).
    #
    # Some diagnostics / ad-hoc callers still evaluate coefficients in scalar
    # loops or pass list-like x values. For those cases we keep:
    #   - a small per-timestep cache for scalar x
    #   - a "last vector" cache that also matches equal-valued x arrays (not only
    #     identical ndarray objects)
    _cache_tau_eff: float | None = None
    _cache_sig2_by_x: dict[float, float] = {}

    # Also keep a fast-path for vector calls (used by the PDE builder and most
    # performance-sensitive paths).
    _last_tau_eff: float | None = None
    _last_x_arr: np.ndarray | None = None
    _last_sig2: FloatArray | None = None

    def _coerce_to_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Best-effort coerce/broadcast to a target shape.

        This keeps vectorized coefficient evaluation robust even when a
        local_var implementation returns e.g. (N, 1) for an (N,) input.
        """

        if arr.shape == shape:
            return arr

        # Scalar -> broadcast
        if arr.size == 1:
            return np.broadcast_to(np.asarray(arr).reshape(()), shape)

        # Same number of elements -> reshape
        n = int(np.prod(shape))
        if arr.size == n:
            return np.asarray(arr).reshape(shape)

        # Fall back to numpy broadcasting rules
        return np.broadcast_to(arr, shape)

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

        # Scalar x path (used by scalar-only diagnostics or as a fallback when
        # a caller provides a non-vectorizable coefficient evaluation path).
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

        # Vector x fast-path.
        # - First try ndarray identity (zero overhead).
        # - Then fall back to value equality so callers can pass list-like x
        #   without defeating the cache (common in diagnostics).
        if (
            _last_sig2 is not None
            and _last_tau_eff == tau_eff
            and _last_x_arr is not None
        ):
            if _last_x_arr is x_arr:
                return _last_sig2
            # Only pay for an equality check when shapes match.
            if _last_x_arr.shape == x_arr.shape and np.array_equal(_last_x_arr, x_arr):
                return _last_sig2

        # Vector compute
        S = to_S(x_arr)
        sig2 = np.asarray(local_var(S, tau_eff), dtype=float)
        target_shape = np.asarray(S, dtype=float).shape
        sig2 = _coerce_to_shape(sig2, target_shape)
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
