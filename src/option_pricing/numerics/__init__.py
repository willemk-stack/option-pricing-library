# src/option_pricing/numerics/__init__.py
"""
Numerical building blocks (advanced API).

Top-level package `option_pricing` exposes the everyday pricing API.
This subpackage exposes reusable numerical primitives.
"""

from .root_finding import RootMethod, RootResult, ensure_bracket, get_root_method
from .tridiag import (
    DEFAULT_BC,
    BoundaryCoupling,
    Tridiag,
    solve_tridiag_scipy,
    solve_tridiag_thomas,
    tridiag_mv,
    tridiag_to_dense,
)

__all__ = [
    # Root finding
    "RootMethod",
    "RootResult",
    "ensure_bracket",
    "get_root_method",
    # Tridiagonal
    "Tridiag",
    "BoundaryCoupling",
    "DEFAULT_BC",
    "solve_tridiag_thomas",
    "solve_tridiag_scipy",
    "tridiag_mv",
    "tridiag_to_dense",
]
