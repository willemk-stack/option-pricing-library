"""Small typed shims for SciPy objects whose stubs lag runtime behavior."""

from __future__ import annotations

from typing import Any, cast

from scipy import optimize as _optimize
from scipy.optimize import OptimizeResult

least_squares = cast(Any, _optimize.least_squares)

__all__ = ["OptimizeResult", "least_squares"]
