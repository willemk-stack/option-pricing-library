from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ArrayLike = float | FloatArray


@dataclass(frozen=True, slots=True)
class Smile:
    """Total-variance smile at a single expiry on a log-moneyness grid.

    The smile is represented in terms of total variance:

    ``w(x) = T * iv(x)^2``

    on a grid of log-moneyness values:

    ``x = ln(K / F(T))``.

    Parameters
    ----------
    T : float
        Expiry in years.
    x : FloatArray
        Strictly increasing log-moneyness grid ``ln(K/F(T))``.
    w : FloatArray
        Total variance values on the grid, ``w = T * iv^2``.

    Notes
    -----
    Interpolation is performed in total variance space using 1D linear interpolation
    with flat extrapolation beyond the endpoints.
    """

    T: float  # years
    x: FloatArray  # log-moneyness grid: ln(K/F(T))
    w: FloatArray  # total variance: T * iv^2

    def w_at(self, xq: ArrayLike) -> FloatArray:
        """Interpolate total variance at queried log-moneyness.

        Parameters
        ----------
        xq : ArrayLike
            Query point(s) in log-moneyness space. May be a scalar float or a NumPy
            array of floats.

        Returns
        -------
        FloatArray
            Interpolated total variance values. If `xq` is scalar, a 0-d array is
            returned; if `xq` is an array, the output matches its shape.

        Notes
        -----
        Uses :func:`numpy.interp` with flat extrapolation:
        values left/right of the grid are clamped to ``w[0]`` / ``w[-1]``.
        """
        xq_arr = np.asarray(xq, dtype=np.float64)

        # np.interp is often typed as returning Any -> wrap with np.asarray to satisfy mypy
        out = np.interp(
            xq_arr,
            self.x,
            self.w,
            left=float(self.w[0]),
            right=float(self.w[-1]),
        )
        return np.asarray(out, dtype=np.float64)

    def iv_at(self, xq: ArrayLike) -> FloatArray:
        """Interpolate implied volatility at queried log-moneyness.

        Parameters
        ----------
        xq : ArrayLike
            Query point(s) in log-moneyness space ``ln(K/F(T))``.

        Returns
        -------
        FloatArray
            Implied volatility values ``sqrt(max(w/T, 0))`` evaluated at `xq`.

        Notes
        -----
        Implied volatility is computed from interpolated total variance as::

            iv(x) = sqrt(max(w(x) / T, 0))

        where the ``max`` guards against small negative values from interpolation.
        """
        wq = self.w_at(xq)
        out = np.sqrt(np.maximum(wq / np.float64(self.T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VolSurface:
    """Piecewise-linear total-variance volatility surface over expiry.

    The surface is defined by a collection of :class:`Smile` objects at discrete
    expiries and a forward curve ``F(T)``. For a given expiry `T` and strike `K`,
    the surface:

    1. Converts strike to log-moneyness ``x = ln(K / F(T))``.
    2. Interpolates total variance ``w(x)`` within each bracketing smile.
    3. Linearly interpolates total variance in time between expiries.
    4. Converts back to implied volatility via ``iv = sqrt(max(w/T, 0))``.

    Parameters
    ----------
    expiries : FloatArray
        Sorted array of expiry times (years).
    smiles : tuple[Smile, ...]
        Smiles corresponding one-to-one with `expiries`.
    forward : Callable[[float], float]
        Forward curve function. Must satisfy ``forward(T) > 0`` for all queried `T`.

    Notes
    -----
    - Extrapolation in expiry is clamped: for `T` outside the known range, the
      nearest smile is used (still evaluated at the query `x` for that `T`).
    - Extrapolation in log-moneyness inside each smile is flat (endpoint clamping).
    """

    expiries: FloatArray
    smiles: tuple[Smile, ...]
    forward: Callable[[float], float]  # forward(T) -> float

    @classmethod
    def from_grid(
        cls,
        rows: Iterable[tuple[float, float, float]],  # (T, K, iv)
        forward: Callable[[float], float],
        *,
        expiry_round_decimals: int = 10,
    ) -> VolSurface:
        """Construct a surface from scattered (T, K, iv) points.

        Points are bucketed by expiry (after rounding), then each bucket is converted
        into a :class:`Smile` defined on a strike-sorted log-moneyness grid with
        total variance values.

        Parameters
        ----------
        rows : Iterable[tuple[float, float, float]]
            Iterable of ``(T, K, iv)`` points where:

            - `T` is expiry in years (or consistent units),
            - `K` is strike (must be positive),
            - `iv` is Black-Scholes implied volatility (non-negative recommended).

        forward : Callable[[float], float]
            Forward curve function ``F(T)``. Must return strictly positive values
            for the expiries present in `rows`.
        expiry_round_decimals : int, default 10
            Number of decimals used when bucketing expiries. Useful when input data
            contains floating-point noise in `T`.

        Returns
        -------
        VolSurface
            Volatility surface built from the provided grid.

        Raises
        ------
        ValueError
            If any ``forward(T) <= 0`` for an expiry in the grid.
        ValueError
            If any strike ``K <= 0`` (log-moneyness undefined).
        ValueError
            If the resulting log-moneyness grid is not strictly increasing for an
            expiry bucket (e.g., duplicate strikes that map to duplicate `x`).

        Notes
        -----
        Total variance is computed as ``w = T * iv^2`` for each point.
        """
        buckets: dict[float, list[tuple[float, float]]] = {}

        for T_raw, K_raw, iv_raw in rows:
            T_key = round(float(T_raw), expiry_round_decimals)
            buckets.setdefault(T_key, []).append((float(K_raw), float(iv_raw)))

        expiries = np.asarray(sorted(buckets.keys()), dtype=np.float64)
        smiles: list[Smile] = []

        for T_np in expiries:
            pts = buckets[float(T_np)]
            pts.sort(key=lambda p: p[0])  # sort by strike

            K = np.asarray([p[0] for p in pts], dtype=np.float64)
            iv = np.asarray([p[1] for p in pts], dtype=np.float64)

            F = float(forward(float(T_np)))
            if F <= 0.0:
                raise ValueError(
                    f"forward(T) must be > 0, got {F} at T_np={float(T_np)}"
                )
            if np.any(K <= 0.0):
                raise ValueError(
                    f"All strikes must be > 0 for log-moneyness at T={float(T_np)}"
                )

            x = np.log(K / np.float64(F)).astype(np.float64, copy=False)
            w = (np.float64(T_np) * (iv**2)).astype(np.float64, copy=False)

            if np.any(np.diff(x) <= 0):
                raise ValueError(
                    f"x grid not strictly increasing at T_np={float(T_np)}"
                )

            smiles.append(Smile(T=float(T_np), x=x, w=w))

        return cls(expiries=expiries, smiles=tuple(smiles), forward=forward)

    def iv(self, K: ArrayLike, T: float) -> FloatArray:
        """Evaluate implied volatility for strike(s) at a given expiry.

        Parameters
        ----------
        K : ArrayLike
            Strike(s) at which to evaluate volatility. May be a scalar float or an
            array of floats. All strikes must be strictly positive.
        T : float
            Expiry at which to evaluate volatility. Must be strictly positive.

        Returns
        -------
        FloatArray
            Implied volatility at the requested strike(s) and expiry. If `K` is
            scalar, a 0-d array is returned; otherwise the output shape matches `K`.

        Raises
        ------
        ValueError
            If ``T <= 0``.
        ValueError
            If any strike in `K` is ``<= 0``.
        ValueError
            If ``forward(T) <= 0`` (indirectly, because log-moneyness is undefined).

        Notes
        -----
        The evaluation proceeds as:

        1. Compute ``x = ln(K / forward(T))``.
        2. If `T` lies outside the surface expiries, clamp to the nearest smile.
        3. Otherwise, linearly interpolate total variance between bracketing expiries:
           ``w(T) = (1-a)*w0 + a*w1`` where ``a = (T-T0)/(T1-T0)``.
        4. Convert to implied volatility: ``iv = sqrt(max(w/T, 0))``.
        """
        T = float(T)
        if T <= 0:
            raise ValueError("T must be > 0")

        K_arr = np.asarray(K, dtype=np.float64)
        if np.any(K_arr <= 0.0):
            raise ValueError("Strikes must be > 0 for log-moneyness.")

        xq = np.log(K_arr / np.float64(self.forward(T))).astype(np.float64, copy=False)

        # Clamp outside known range
        if T <= float(self.expiries[0]):
            return self.smiles[0].iv_at(xq)
        if T >= float(self.expiries[-1]):
            return self.smiles[-1].iv_at(xq)

        # Find bracketing expiries
        j = int(np.searchsorted(self.expiries, np.float64(T)))
        i = j - 1

        T0 = float(self.expiries[i])
        T1 = float(self.expiries[j])

        s0 = self.smiles[i]
        s1 = self.smiles[j]

        w0 = s0.w_at(xq)
        w1 = s1.w_at(xq)

        a = (T - T0) / (T1 - T0)
        w = np.float64(1.0 - a) * w0 + np.float64(a) * w1
        out = np.sqrt(np.maximum(w / np.float64(T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)
