"""Core volatility surface implementation.

This module contains the main VolSurface class for managing implied volatility
surfaces with per-expiry smiles and time interpolation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, cast, overload

import numpy as np

from ..numerics.interpolation import FritschCarlson, linear_interp_factory
from ..numerics.regression import isotonic_regression
from ..typing import ArrayLike, FloatArray, ScalarFn
from .smile_grid import Smile, _is_monotone
from .smile_interpolated import (
    LinearWInterpolatedSmileSlice,
    NoArbInterpolatedSmileSlice,
)
from .vol_types import DifferentiableSmileSlice, SmileSlice

TimeInterp = Literal["no_arb", "linear_w"]


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
    _theta_interp: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)

    def __post_init__(
        self,
    ) -> None:  # TODO: Check if is TimediffbleSurface => yes then skip
        expiries = np.asarray(self.expiries, dtype=np.float64)
        theta_raw = np.asarray(
            [float(np.asarray(s.w_at(0.0))) for s in self.smiles], dtype=np.float64
        )
        theta = isotonic_regression(theta_raw)
        # If theta isn't monotone due to noise, either:
        # - raise, or
        # - fall back to linear, or
        # - monotone-regress it (not shown here).
        if expiries.size < 2:
            theta0 = float(theta[0])

            def theta_const(xq: np.ndarray) -> np.ndarray:
                xq_in = np.asarray(xq, dtype=np.float64)
                return np.full_like(xq_in, theta0, dtype=np.float64)

            object.__setattr__(self, "_theta_interp", theta_const)
        elif _is_monotone(theta) and expiries.size >= 3:
            theta_interp, _ = FritschCarlson(expiries, theta)
            object.__setattr__(self, "_theta_interp", theta_interp)
        else:
            object.__setattr__(
                self, "_theta_interp", linear_interp_factory(expiries, theta)
            )

    @classmethod
    def from_grid(
        cls,
        rows: Iterable[tuple[float, float, float]],  # (T, K, iv)
        forward: ScalarFn,
        *,
        expiry_round_decimals: int = 10,
    ) -> VolSurface:
        """Build a surface from market quotes (T, K, iv) on a grid.

        Parameters
        ----------
        rows
            Iterable of (T, K, iv) market data points.
        forward
            Callable forward(T) -> float.
        expiry_round_decimals
            Number of decimals to round each T for bucketing (default: 10).

        Returns
        -------
        VolSurface
            Surface with Smile slices per unique expiry.
        """
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
            y_lo, y_hi = fit.diag.checks.y_domain

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

    @overload
    def slice(self, T: float, method: Literal["no_arb"] = "no_arb") -> SmileSlice: ...
    @overload
    def slice(
        self, T: float, method: Literal["linear_w"]
    ) -> DifferentiableSmileSlice: ...

    def slice(self, T: float, method: TimeInterp = "no_arb") -> SmileSlice:
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

        if T <= float(self.expiries[0]):
            return self.smiles[0]
        if T >= float(self.expiries[-1]):
            return self.smiles[-1]

        j = int(np.searchsorted(self.expiries, np.float64(T), side="left"))
        if j < len(self.expiries) and bool(
            np.isclose(self.expiries[j], np.float64(T), rtol=0.0, atol=1e-12)
        ):
            return self.smiles[j]

        i = j - 1
        T0 = float(self.expiries[i])
        T1 = float(self.expiries[j])
        a = (T - T0) / (T1 - T0)

        s0 = self.smiles[i]
        s1 = self.smiles[j]

        if method == "no_arb":
            return NoArbInterpolatedSmileSlice(
                T=T, s0=s0, s1=s1, a=float(a), theta_interp=self._theta_interp
            )

        if method == "linear_w":
            if not isinstance(s0, DifferentiableSmileSlice) or not isinstance(
                s1, DifferentiableSmileSlice
            ):
                raise TypeError(
                    "method='linear_w' requires DifferentiableSmileSlice endpoints "
                    "(e.g. SVISmile / SSVI). Use method='no_arb' or rebuild surface with differentiable slices."
                )
            return LinearWInterpolatedSmileSlice(
                T=T,
                s0=cast(DifferentiableSmileSlice, s0),
                s1=cast(DifferentiableSmileSlice, s1),
                a=float(a),
            )

        raise ValueError(f"method={method!r} not supported")

    def iv(self, K: ArrayLike, T: float) -> FloatArray:
        """Implied volatility at given strikes and maturity.

        Parameters
        ----------
        K
            Strike price(s).
        T
            Maturity time.

        Returns
        -------
        FloatArray
            Implied volatility at (K, T).
        """
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
        """Total variance at given log-moneyness and maturity.

        Parameters
        ----------
        y
            Log-moneyness y = ln(K/F).
        T
            Maturity time.

        Returns
        -------
        FloatArray
            Total variance w(y, T) = T * iv(y, T)^2.
        """
        T = float(T)
        if T <= 0.0:
            raise ValueError("T must be > 0")
        y_arr = np.asarray(y, dtype=np.float64)
        return np.asarray(self.slice(T).w_at(y_arr), dtype=np.float64)
