from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from ..numerics.interpolation import FritschCarlson
from ..typing import ArrayLike, FloatArray, ScalarFn
from .dupire import _gatheral_local_var_from_w
from .vol_types import SmileSlice


def _is_monotone(y: np.ndarray) -> bool:
    dy = np.diff(y)
    return bool(np.all(dy >= 0.0) or np.all(dy <= 0.0))


def _u_split_index(w: np.ndarray, *, eps: float | None = None) -> int | None:
    """Heuristic U-shape split index.

    Returns k such that left piece is w[:k+1], right piece is w[k:].
    Uses a tolerance to reduce sensitivity to small numerical noise / plateaus.
    """
    if w.size < 3:
        return None

    dw = np.diff(w)

    if eps is None:
        scale = float(max(1.0, np.max(np.abs(w))))
        eps = 1e-12 * scale

    neg = dw < -eps
    pos = dw > eps
    if not np.any(neg) or not np.any(pos):
        return None

    # first index where slope becomes meaningfully positive
    k = int(np.argmax(pos)) + 1
    if 0 < k < w.size - 1:
        return k
    return None


def _linear_interp_factory(
    x: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """1D linear interpolation with flat extrapolation."""
    y0 = float(y[0])
    y1 = float(y[-1])

    def plin(xq: np.ndarray) -> np.ndarray:
        xq_in = np.asarray(xq, dtype=np.float64)
        out = np.interp(xq_in, x, y, left=y0, right=y1)
        return np.asarray(out, dtype=np.float64)

    return plin


def _stitch_two(
    xk: float,
    left_interp: Callable[[np.ndarray], np.ndarray],
    right_interp: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Stitch two array-capable interpolators at xk, handling scalar xq safely."""

    def pw(xq: np.ndarray) -> np.ndarray:
        xq_in = np.asarray(xq, dtype=np.float64)
        xq_1d = np.atleast_1d(xq_in)

        out = np.empty_like(xq_1d, dtype=np.float64)
        left = xq_1d <= xk
        if np.any(left):
            out[left] = left_interp(xq_1d[left])
        if np.any(~left):
            out[~left] = right_interp(xq_1d[~left])

        if xq_in.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(xq_in.shape)

    return pw


def _make_w_interpolator(
    x: np.ndarray, w: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Pick an interpolator for total variance w(x).

    Priority:
      1) If w is globally monotone: FC on full grid.
      2) Else try split into two monotone pieces (U-shape):
         - robust sign-change split
         - fallback: split at argmin
      3) Else linear with flat extrapolation.
    """
    # Fail-safe: if non-finite, fallback to linear
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(w))):
        return _linear_interp_factory(x, w)

    # Case 1: monotone overall -> single FC
    if _is_monotone(w):
        return FritschCarlson(x, w)

    # Case 2a: robust U-shape split
    k = _u_split_index(w)
    if k is not None:
        xL, wL = x[: k + 1], w[: k + 1]
        xR, wR = x[k:], w[k:]
        if _is_monotone(wL) and _is_monotone(wR):
            pL = FritschCarlson(xL, wL)
            pR = FritschCarlson(xR, wR)
            return _stitch_two(float(x[k]), pL, pR)

    # Case 2b: fallback split at global minimum
    k2 = int(np.argmin(w))
    if 0 < k2 < len(w) - 1:
        xL, wL = x[: k2 + 1], w[: k2 + 1]
        xR, wR = x[k2:], w[k2:]
        if _is_monotone(wL) and _is_monotone(wR):
            pL = FritschCarlson(xL, wL)
            pR = FritschCarlson(xR, wR)
            return _stitch_two(float(x[k2]), pL, pR)

    # Fallback: linear interp with flat extrapolation
    return _linear_interp_factory(x, w)


@dataclass(frozen=True, slots=True)
class Smile:
    """Total-variance smile at a single expiry on a log-moneyness grid.

    The smile is represented in terms of total variance:

        w(y) = T * iv(y)^2

    on a strictly increasing log-moneyness grid:

        y = ln(K / F(T)).

    Interpolation:
      - Uses Fritsch-Carlson monotone cubic interpolation when `w(x)` is monotone
        or can be split into two monotone pieces (common U-shape).
      - Falls back to linear interpolation with flat extrapolation otherwise.
    """

    T: float
    y: FloatArray
    w: FloatArray
    _w_interp: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        y = np.asarray(self.y, dtype=np.float64)
        w = np.asarray(self.w, dtype=np.float64)

        if y.shape != w.shape:
            raise ValueError("Smile.y and Smile.w must have the same shape")
        if y.size < 2:
            raise ValueError("Smile requires at least 2 points")
        if np.any(np.diff(y) <= 0.0):
            raise ValueError("Smile.y must be strictly increasing")

        object.__setattr__(self, "_w_interp", _make_w_interpolator(y, w))

    def w_at(self, xq: ArrayLike) -> FloatArray:
        xq_arr = np.asarray(xq, dtype=np.float64)
        out = self._w_interp(xq_arr)
        return np.asarray(out, dtype=np.float64)

    def iv_at(self, xq: ArrayLike) -> FloatArray:
        wq = self.w_at(xq)
        out = np.sqrt(np.maximum(wq / np.float64(self.T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)

    @property
    def y_min(self) -> float:
        return float(self.y[0])

    @property
    def y_max(self) -> float:
        return float(self.y[-1])


@dataclass(frozen=True, slots=True)
class InterpolatedSmileSlice:
    """A synthetic smile slice formed by linear interpolation in total variance.

        w(y, T) = (1-a) * w0(y) + a * w1(y)

    where (T0, w0) and (T1, w1) are the bracketing slices.

    Notes
    -----
    This is primarily useful for implied-vol interpolation. If the underlying
    slices provide analytic derivatives in y (e.g. SVI), this class exposes
    dw_dy and d2w_dy2 as the same linear blend.
    """

    T: float
    s0: SmileSlice
    s1: SmileSlice
    a: float  # in [0, 1]

    def w_at(self, xq: ArrayLike) -> FloatArray:
        xq_arr = np.asarray(xq, dtype=np.float64)
        w0 = self.s0.w_at(xq_arr)
        w1 = self.s1.w_at(xq_arr)
        w = np.float64(1.0 - self.a) * w0 + np.float64(self.a) * w1
        return np.asarray(w, dtype=np.float64)

    def iv_at(self, xq: ArrayLike) -> FloatArray:
        wq = self.w_at(xq)
        out = np.sqrt(np.maximum(wq / np.float64(self.T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)

    # ---- optional derivatives in y (log-moneyness) ----
    def dw_dy(self, xq: ArrayLike) -> FloatArray:
        if not (hasattr(self.s0, "dw_dy") and hasattr(self.s1, "dw_dy")):
            raise AttributeError("Underlying slices do not provide dw_dy.")
        xq_arr = np.asarray(xq, dtype=np.float64)
        wy0 = self.s0.dw_dy(xq_arr)  # type: ignore[attr-defined]
        wy1 = self.s1.dw_dy(xq_arr)  # type: ignore[attr-defined]
        wy = np.float64(1.0 - self.a) * wy0 + np.float64(self.a) * wy1
        return np.asarray(wy, dtype=np.float64)

    def d2w_dy2(self, xq: ArrayLike) -> FloatArray:
        if not (hasattr(self.s0, "d2w_dy2") and hasattr(self.s1, "d2w_dy2")):
            raise AttributeError("Underlying slices do not provide d2w_dy2.")
        xq_arr = np.asarray(xq, dtype=np.float64)
        wyy0 = self.s0.d2w_dy2(xq_arr)  # type: ignore[attr-defined]
        wyy1 = self.s1.d2w_dy2(xq_arr)  # type: ignore[attr-defined]
        wyy = np.float64(1.0 - self.a) * wyy0 + np.float64(self.a) * wyy1
        return np.asarray(wyy, dtype=np.float64)

    @property
    def y_min(self) -> float:
        # conservative sampling domain: overlap if possible, else union
        lo = max(float(self.s0.y_min), float(self.s1.y_min))
        hi = min(float(self.s0.y_max), float(self.s1.y_max))
        if lo < hi:
            return lo
        return min(float(self.s0.y_min), float(self.s1.y_min))

    @property
    def y_max(self) -> float:
        lo = max(float(self.s0.y_min), float(self.s1.y_min))
        hi = min(float(self.s0.y_max), float(self.s1.y_max))
        if lo < hi:
            return hi
        return max(float(self.s0.y_max), float(self.s1.y_max))


@dataclass(frozen=True, slots=True)
class VolSurface:
    """Total-variance volatility surface over expiry.

    Within each expiry slice (Smile), total variance is interpolated using:
      - Fritsch-Carlson on monotone (or split-monotone) data, else linear fallback.

    Across expiry, total variance is linearly interpolated in time between smiles.
    """

    expiries: FloatArray
    smiles: tuple[SmileSlice, ...]
    forward: ScalarFn  # forward(T) -> float

    @classmethod
    def from_grid(
        cls,
        rows: Iterable[tuple[float, float, float]],  # (T, K, iv)
        forward: ScalarFn,
        *,
        expiry_round_decimals: int = 10,
    ) -> VolSurface:
        buckets: dict[float, list[tuple[float, float]]] = {}

        for T_raw, K_raw, iv_raw in rows:
            T_key = round(float(T_raw), expiry_round_decimals)
            buckets.setdefault(T_key, []).append((float(K_raw), float(iv_raw)))

        expiries = np.asarray(sorted(buckets.keys()), dtype=np.float64)
        smiles: list[SmileSlice] = []

        for T_np in expiries:
            pts = buckets[float(T_np)]
            pts.sort(key=lambda p: p[0])  # sort by strike

            K = np.asarray([p[0] for p in pts], dtype=np.float64)
            iv = np.asarray([p[1] for p in pts], dtype=np.float64)

            F = float(forward(float(T_np)))
            if F <= 0.0:
                raise ValueError(f"forward(T) must be > 0, got {F} at T={float(T_np)}")
            if np.any(K <= 0.0):
                raise ValueError(f"All strikes must be > 0 at T={float(T_np)}")

            y = np.log(K / np.float64(F)).astype(np.float64, copy=False)
            w = (np.float64(T_np) * (iv**2)).astype(np.float64, copy=False)

            if np.any(np.diff(y) <= 0.0):
                raise ValueError(f"x grid not strictly increasing at T={float(T_np)}")

            smiles.append(Smile(T=float(T_np), y=y, w=w))

        return cls(expiries=expiries, smiles=tuple(smiles), forward=forward)

    @classmethod
    def from_svi(
        cls,
        rows: Iterable[tuple[float, float, float]],  # (T, K, iv)
        forward: ScalarFn,
        *,
        expiry_round_decimals: int = 10,
        calibrate_kwargs: dict | None = None,
    ) -> VolSurface:
        """Build a surface by calibrating an SVI smile per expiry.

        Parameters
        ----------
        rows:
            Iterable of (T, K, iv) market points.
        forward:
            forward(T) -> float.
        expiry_round_decimals:
            Bucketing tolerance for float maturities.
        calibrate_kwargs:
            Optional kwargs forwarded to :func:`option_pricing.vol.svi.calibrate_svi`.

        Returns
        -------
        VolSurface
            A surface whose slices are analytic SVI smiles.

        Notes
        -----
        This keeps ``VolSurface`` itself dependency-light: SciPy is only imported
        if you call this method.
        """

        # Local import to keep base surface usage free of SciPy requirements.
        from .svi import SVISmile, calibrate_svi

        buckets: dict[float, list[tuple[float, float]]] = {}

        for T_raw, K_raw, iv_raw in rows:
            T_key = round(float(T_raw), expiry_round_decimals)
            buckets.setdefault(T_key, []).append((float(K_raw), float(iv_raw)))

        expiries = np.asarray(sorted(buckets.keys()), dtype=np.float64)
        smiles: list[SmileSlice] = []

        ck = {} if calibrate_kwargs is None else dict(calibrate_kwargs)

        for T_np in expiries:
            pts = buckets[float(T_np)]
            pts.sort(key=lambda p: p[0])

            K = np.asarray([p[0] for p in pts], dtype=np.float64)
            iv = np.asarray([p[1] for p in pts], dtype=np.float64)

            F = float(forward(float(T_np)))
            if F <= 0.0:
                raise ValueError(f"forward(T) must be > 0, got {F} at T={float(T_np)}")
            if np.any(K <= 0.0):
                raise ValueError(f"All strikes must be > 0 at T={float(T_np)}")

            y = np.log(K / np.float64(F)).astype(np.float64, copy=False)
            w = (np.float64(T_np) * (iv**2)).astype(np.float64, copy=False)

            if np.any(np.diff(y) <= 0.0):
                raise ValueError(f"x grid not strictly increasing at T={float(T_np)}")

            fit = calibrate_svi(y=y, w_obs=w, **ck)

            # Use the diagnostic domain as the recommended sampling range.
            y_lo, y_hi = fit.diag.y_domain
            smiles.append(
                SVISmile(
                    T=float(T_np),
                    params=fit.params,
                    y_min=float(y_lo),
                    y_max=float(y_hi),
                    diagnostics=fit.diag,
                )
            )

        return cls(expiries=expiries, smiles=tuple(smiles), forward=forward)

    def slice(self, T: float) -> SmileSlice:
        """Return a callable single-expiry smile slice at maturity T.

        - Node expiry: returns the stored slice (grid or SVI).
        - Off-node: returns :class:`InterpolatedSmileSlice`, linear in total variance.

        Notes
        -----
        The surface is defined in log-moneyness y = ln(K/F(T)). Interpolating in y
        between expiries corresponds to *constant log-moneyness* interpolation.
        """
        T = float(T)
        if T <= 0.0:
            raise ValueError("T must be > 0")

        # Clamp outside known range
        if T <= float(self.expiries[0]):
            return self.smiles[0]
        if T >= float(self.expiries[-1]):
            return self.smiles[-1]

        j = int(np.searchsorted(self.expiries, np.float64(T), side="left"))

        # Exact node hit (tolerant to tiny float noise)
        if j < len(self.expiries) and bool(
            np.isclose(self.expiries[j], np.float64(T), rtol=0.0, atol=1e-12)
        ):
            return self.smiles[j]

        i = j - 1
        T0 = float(self.expiries[i])
        T1 = float(self.expiries[j])
        a = (T - T0) / (T1 - T0)
        return InterpolatedSmileSlice(
            T=T, s0=self.smiles[i], s1=self.smiles[j], a=float(a)
        )

    def iv(self, K: ArrayLike, T: float) -> FloatArray:
        T = float(T)
        if T <= 0.0:
            raise ValueError("T must be > 0")

        K_arr = np.asarray(K, dtype=np.float64)
        if np.any(K_arr <= 0.0):
            raise ValueError("Strikes must be > 0 for log-moneyness.")

        F = float(self.forward(T))
        if F <= 0.0:
            raise ValueError(f"forward(T) must be > 0, got {F} at T={T}")

        y = np.log(K_arr / np.float64(F)).astype(np.float64, copy=False)
        return self.slice(T).iv_at(y)

    def w(self, y: ArrayLike, T: float) -> FloatArray:
        """Return total variance w(y, T) at log-moneyness y."""
        T = float(T)
        if T <= 0.0:
            raise ValueError("T must be > 0")
        y_arr = np.asarray(y, dtype=np.float64)
        return np.asarray(self.slice(T).w_at(y_arr), dtype=np.float64)


###
# TEMP SVI LocalVolSurface before SSVI is implemented
###


@dataclass(frozen=True, slots=True)
class LocalVolSurface:
    """Local-vol surface computed from an implied surface in total variance.

    This is a *minimal* implementation intended for demos while I work on a
    time-consistent parameterization (e.g. SSVI/eSSVI).

    Requirements
    ------------
    The underlying implied surface must provide, per expiry slice:
      - w_at(y)
      - dw_dy(y)
      - d2w_dy2(y)

    i.e. analytic (or otherwise smooth) derivatives in log-moneyness.

    Time derivative w_T
    -------------------
    w_T is approximated by the same piecewise-linear interpolation you use for
    implied vols:
        w_T(y,T) \approx (w(y,T1) - w(y,T0)) / (T1 - T0)

    which is piecewise-constant in T between expiries and can create visible
    "bands" in local vol.
    """

    implied: VolSurface
    forward: ScalarFn
    discount: ScalarFn

    def __post_init__(self) -> None:
        # Heuristic: if all slices have SVI derivative methods, this is the “SVI-derived” LV path.
        smiles = getattr(self.implied, "smiles", ())
        is_svi_like = bool(smiles) and all(
            hasattr(s, "dw_dy") and hasattr(s, "d2w_dy2") for s in smiles
        )

        if is_svi_like:
            warnings.warn(
                "LocalVolSurface is currently derived from per-expiry SVI slices with piecewise-linear "
                "time interpolation in total variance. This is demo-grade and can be numerically unstable "
                "(e.g., banding from discontinuous w_T). It will be replaced by a time-consistent SSVI/eSSVI "
                "implementation in a future version.",
                category=FutureWarning,  # or UserWarning
                stacklevel=2,
            )

    @classmethod
    def from_implied(
        cls,
        implied: VolSurface,
        *,
        forward: ScalarFn | None = None,
        discount: ScalarFn | None = None,
    ) -> LocalVolSurface:
        """Convenience constructor.

        If forward/discount are omitted, uses implied.forward and a flat df=1.
        """
        fwd = implied.forward if forward is None else forward
        df = (lambda T: 1.0) if discount is None else discount
        return cls(implied=implied, forward=fwd, discount=df)

    def _require_derivs(self) -> None:
        for s in self.implied.smiles:
            if not (hasattr(s, "dw_dy") and hasattr(s, "d2w_dy2")):
                raise TypeError(
                    "LocalVolSurface requires implied slices with dw_dy and d2w_dy2 "
                    "(e.g. SVISmile)."
                )

    def _bracket(self, T: float) -> tuple[int, int, float]:
        """Return (i, j, a) such that T in [Ti,Tj], a in [0,1]."""
        T = float(T)
        exp = np.asarray(self.implied.expiries, dtype=np.float64)
        if exp.size < 2:
            raise ValueError("Need at least 2 expiries to compute w_T.")

        if T <= float(exp[0]):
            return 0, 1, 0.0
        if T >= float(exp[-1]):
            n = int(exp.size)
            return n - 2, n - 1, 1.0

        j = int(np.searchsorted(exp, np.float64(T)))
        i = j - 1
        T0 = float(exp[i])
        T1 = float(exp[j])
        a = (T - T0) / (T1 - T0)
        return i, j, float(a)

    def _w_and_derivs(
        self, y: FloatArray, T: float
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Compute (w, w_y, w_yy, w_T) at fixed y for maturity T."""
        self._require_derivs()

        y_arr = np.asarray(y, dtype=np.float64)
        i, j, a = self._bracket(T)

        exp = np.asarray(self.implied.expiries, dtype=np.float64)
        T0 = float(exp[i])
        T1 = float(exp[j])
        s0 = self.implied.smiles[i]
        s1 = self.implied.smiles[j]

        w0 = np.asarray(s0.w_at(y_arr), dtype=np.float64)
        w1 = np.asarray(s1.w_at(y_arr), dtype=np.float64)
        wy0 = np.asarray(s0.dw_dy(y_arr), dtype=np.float64)  # type: ignore[attr-defined]
        wy1 = np.asarray(s1.dw_dy(y_arr), dtype=np.float64)  # type: ignore[attr-defined]
        wyy0 = np.asarray(s0.d2w_dy2(y_arr), dtype=np.float64)  # type: ignore[attr-defined]
        wyy1 = np.asarray(s1.d2w_dy2(y_arr), dtype=np.float64)  # type: ignore[attr-defined]

        w = (1.0 - a) * w0 + a * w1
        w_y = (1.0 - a) * wy0 + a * wy1
        w_yy = (1.0 - a) * wyy0 + a * wyy1
        w_T = (w1 - w0) / (T1 - T0)

        return (
            np.asarray(w, dtype=np.float64),
            np.asarray(w_y, dtype=np.float64),
            np.asarray(w_yy, dtype=np.float64),
            np.asarray(w_T, dtype=np.float64),
        )

    def local_var(
        self,
        K: ArrayLike,
        T: float,
        *,
        eps_w: float = 1e-12,
        eps_denom: float = 1e-12,
    ) -> FloatArray:
        """Return local variance sigma_loc^2 at strike K and maturity T."""
        T = float(T)
        if T <= 0.0:
            raise ValueError("T must be > 0")

        K_arr = np.asarray(K, dtype=np.float64)
        if np.any(K_arr <= 0.0):
            raise ValueError("K must be > 0")

        F = float(self.forward(T))
        if F <= 0.0:
            raise ValueError(f"forward(T) must be > 0, got {F} at T={T}")

        y = np.log(K_arr / np.float64(F)).astype(np.float64, copy=False)
        w, w_y, w_yy, w_T = self._w_and_derivs(y, T)

        lv, _ = _gatheral_local_var_from_w(
            y=y, w=w, w_y=w_y, w_yy=w_yy, w_T=w_T, eps_w=eps_w, eps_denom=eps_denom
        )
        return np.asarray(lv, dtype=np.float64)

    def local_vol(
        self,
        K: ArrayLike,
        T: float,
        *,
        eps_w: float = 1e-12,
        eps_denom: float = 1e-12,
    ) -> FloatArray:
        """Return local volatility sigma_loc at strike K and maturity T."""
        lv = self.local_var(K, T, eps_w=eps_w, eps_denom=eps_denom)
        with np.errstate(invalid="ignore"):
            out = np.sqrt(lv)
        return np.asarray(out, dtype=np.float64)
