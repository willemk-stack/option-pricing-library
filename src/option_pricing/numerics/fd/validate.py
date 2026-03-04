"""
Docstring for option_pricing.numerics.fd.validate
numerics/fd/validate.py (optional)

Responsibility: tiny helpers like:
- assert_strictly_increasing(x, name)
- assert_min_points(x, n, name)



Why: both PDE and Dupire do the same validation (monotonic grid, min sizes). This avoids “slightly different errors” across codepaths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def assert_strictly_increasing(x: np.ndarray, name: str) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"{name} must be strictly increasing")


def validate_inputs(y: NDArray, x: NDArray, axis: int) -> int:
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
