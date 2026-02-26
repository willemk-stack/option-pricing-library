"""Local volatility surface derived from implied volatility.

This module provides the LocalVolSurface class for computing local volatility
from an implied surface using Gatheral's formula.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..typing import ArrayLike, FloatArray, ScalarFn
from .local_vol_gatheral import (
    GatheralLVReport,
    _gatheral_local_var_from_w,
    gatheral_local_var_diagnostics,
)
from .surface_core import VolSurface
from .vol_types import DifferentiableSmileSlice


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
        # Heuristic: if all slices have SVI derivative methods, this is the "SVI-derived" LV path.
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
        """Convenience constructor from an implied volatility surface.

        Parameters
        ----------
        implied
            Implied volatility surface (:class:`VolSurface`).
        forward
            Forward price function. If None, uses implied.forward.
        discount
            Discount factor function. If None, uses flat df=1.0.

        Returns
        -------
        LocalVolSurface
            Local volatility surface derived from implied.
        """
        fwd = implied.forward if forward is None else forward
        df = (lambda T: 1.0) if discount is None else discount
        return cls(implied=implied, forward=fwd, discount=df)

    def _require_derivs(self) -> None:
        for s in self.implied.smiles:
            if not isinstance(s, DifferentiableSmileSlice):
                raise TypeError(
                    "LocalVolSurface requires DifferentiableSmileSlice expiries "
                    "(e.g. SVISmile / SSVI)."
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

        if not isinstance(s0, DifferentiableSmileSlice) or not isinstance(
            s1, DifferentiableSmileSlice
        ):
            raise TypeError("LocalVolSurface requires differentiable slices")

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
        """Local variance sigma_loc^2(K, T).

        Computed from implied surface using Gatheral's one-dimensional
        local-volatility formula in log-moneyness coordinates.

        Parameters
        ----------
        K
            Strike price(s).
        T
            Maturity time.
        eps_w
            Small value to avoid numerical issues near w=0 (default: 1e-12).
        eps_denom
            Small value for denominator regularization (default: 1e-12).

        Returns
        -------
        FloatArray
            Local variance at (K, T).
        """
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

    def local_var_diagnostics(
        self,
        K: ArrayLike,
        T: float,
        *,
        eps_w: float = 1e-12,
        eps_denom: float = 1e-12,
    ) -> GatheralLVReport:
        """Diagnostics for local variance computation.

        Returns intermediate quantities (w derivatives, local vol, etc.)
        useful for analysis and debugging.

        Parameters
        ----------
        K
            Strike price(s).
        T
            Maturity time.
        eps_w
            Small value to avoid numerical issues near w=0 (default: 1e-12).
        eps_denom
            Small value for denominator regularization (default: 1e-12).

        Returns
        -------
        GatheralLVReport
            Report object containing w, derivatives, local vol, and other diagnostics.
        """
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

        return gatheral_local_var_diagnostics(
            y=y, w=w, w_y=w_y, w_yy=w_yy, w_T=w_T, eps_w=eps_w, eps_denom=eps_denom
        )

    def local_vol(
        self,
        K: ArrayLike,
        T: float,
        *,
        eps_w: float = 1e-12,
        eps_denom: float = 1e-12,
    ) -> FloatArray:
        """Local volatility sigma_loc(K, T).

        Square root of local variance.

        Parameters
        ----------
        K
            Strike price(s).
        T
            Maturity time.
        eps_w
            Small value to avoid numerical issues near w=0 (default: 1e-12).
        eps_denom
            Small value for denominator regularization (default: 1e-12).

        Returns
        -------
        FloatArray
            Local volatility at (K, T).
        """
        lv = self.local_var(K, T, eps_w=eps_w, eps_denom=eps_denom)
        with np.errstate(invalid="ignore"):
            out = np.sqrt(lv)
        return np.asarray(out, dtype=np.float64)
