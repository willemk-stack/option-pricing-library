"""Finite-difference differentiation utilities on nonuniform 1D grids.

This module provides helpers to compute first and second derivatives of an
array-valued function sampled on a strictly increasing 1D grid. The functions
support vectorized/batched inputs by differentiating along a chosen axis.

Design notes
------------
- These routines are intended as low-level numerics primitives. They do not
  enforce domain-specific conventions (e.g., PDE advection schemes, finance
  formulas).
- For PDE operator assembly, prefer coefficient-based helpers in
  `option_pricing.numerics.fd.stencils` and build tri-diagonal operators in
  `option_pricing.numerics.pde`.
- Boundary handling uses one-sided 3-point formulas by default. Derivatives
  at the boundaries can be substantially less reliable than interior points
  on irregular/noisy data (e.g., Dupire local vol in the wings).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .stencils import (
    d1_central_nonuniform_coeffs,
    d2_central_nonuniform_coeffs,
    lagrange_3pt_weights,
)


def _validate_inputs(y: NDArray, x: NDArray, axis: int) -> int:
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if x.size < 3:
        raise ValueError("x must have at least 3 points.")
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("x must be strictly increasing.")

    # Normalize axis
    if axis < 0:
        axis = y.ndim + axis
    if axis < 0 or axis >= y.ndim:
        raise ValueError(f"axis {axis} out of bounds for y.ndim={y.ndim}.")

    if y.shape[axis] != x.size:
        raise ValueError(f"y.shape[{axis}] must match len(x) ({x.size}).")

    return axis


def diff1_nonuniform(y: NDArray, x: NDArray, axis: int = -1) -> NDArray:
    """Compute the first derivative dy/dx on a nonuniform 1D grid.

    Parameters
    ----------
    y:
        Samples of a function evaluated on the grid `x`. `y` may be an N-D array.
        The length of the differentiation axis must match `len(x)`.
    x:
        Strictly increasing 1D grid of shape (n,). The grid spacing may be
        nonuniform.
    axis:
        Axis of `y` corresponding to the grid dimension. By default `axis=-1`
        (the last axis), which supports common layouts like:
        - (n,) for 1D data
        - (..., n) for batched data
        - (nT, nK) with K as the last axis (differentiate w.r.t. strike)
        Use `axis=0` for layouts where the grid is the first dimension.

    Returns
    -------
    np.ndarray
        Array of the same shape as `y` containing the first derivative along
        `axis`.

    Notes
    -----
    - Interior points use a 3-point central difference that is second-order
      accurate on nonuniform grids.
    - Boundary points use one-sided 3-point formulas (also second-order in
      smooth settings). On irregular grids and/or noisy data, boundary
      derivatives can be unstable; callers may want to mask/trim boundaries.

    Raises
    ------
    ValueError
        If `x` is not 1D, not strictly increasing, or if `y.shape[axis] != len(x)`.
    """
    axis = _validate_inputs(y, x, axis)

    y_moved = np.moveaxis(y, axis, -1)  # (..., N)
    out = np.empty_like(y_moved, dtype=float)

    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]

    # interior: central 3pt, 2nd order on nonuniform grids
    dl, dd, du = d1_central_nonuniform_coeffs(hm, hp)
    out[..., 1:-1] = (
        dl * y_moved[..., :-2] + dd * y_moved[..., 1:-1] + du * y_moved[..., 2:]
    )

    # boundaries: one-sided 3pt Lagrange at x[0] and x[-1]
    w0, w1, w2 = lagrange_3pt_weights(x[0], x[1], x[2], x[0], deriv=1)
    out[..., 0] = w0 * y_moved[..., 0] + w1 * y_moved[..., 1] + w2 * y_moved[..., 2]

    w0, w1, w2 = lagrange_3pt_weights(x[-3], x[-2], x[-1], x[-1], deriv=1)
    out[..., -1] = w0 * y_moved[..., -3] + w1 * y_moved[..., -2] + w2 * y_moved[..., -1]

    return np.moveaxis(out, -1, axis)


def diff2_nonuniform(y: NDArray, x: NDArray, axis: int = -1) -> NDArray:
    """Compute the second derivative d^2y/dx^2 on a nonuniform 1D grid.

    Parameters
    ----------
    y:
        Samples of a function evaluated on the grid `x`. `y` may be an N-D array.
        The length of the differentiation axis must match `len(x)`.
    x:
        Strictly increasing 1D grid of shape (n,). The grid spacing may be
        nonuniform.
    axis:
        Axis of `y` corresponding to the grid dimension. By default `axis=-1`
        (the last axis), which supports common layouts like:
        - (n,) for 1D data
        - (..., n) for batched data
        - (nT, nK) with K as the last axis (differentiate w.r.t. strike)
        Use `axis=0` for layouts where the grid is the first dimension.

    Returns
    -------
    np.ndarray
        Array of the same shape as `y` containing the first derivative along
        `axis`.

    Notes
    -----
    - Interior points use a 3-point central difference that is second-order
      accurate on nonuniform grids.
    - Boundary points use one-sided 3-point formulas (also second-order in
      smooth settings). On irregular grids and/or noisy data, boundary
      derivatives can be unstable; callers may want to mask/trim boundaries.

    Raises
    ------
    ValueError
        If `x` is not 1D, not strictly increasing, or if `y.shape[axis] != len(x)`.
    """
    axis = _validate_inputs(y, x, axis)

    y_moved = np.moveaxis(y, axis, -1)  # (..., N)
    out = np.empty_like(y_moved, dtype=float)

    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]

    # interior: central 3pt, 2nd order on nonuniform grids
    dl, dd, du = d2_central_nonuniform_coeffs(hm, hp)
    out[..., 1:-1] = (
        dl * y_moved[..., :-2] + dd * y_moved[..., 1:-1] + du * y_moved[..., 2:]
    )

    # boundaries: one-sided 3pt Lagrange at x[0] and x[-1]
    w0, w1, w2 = lagrange_3pt_weights(x[0], x[1], x[2], x[0], deriv=2)
    out[..., 0] = w0 * y_moved[..., 0] + w1 * y_moved[..., 1] + w2 * y_moved[..., 2]

    w0, w1, w2 = lagrange_3pt_weights(x[-3], x[-2], x[-1], x[-1], deriv=2)
    out[..., -1] = w0 * y_moved[..., -3] + w1 * y_moved[..., -2] + w2 * y_moved[..., -1]

    return np.moveaxis(out, -1, axis)
