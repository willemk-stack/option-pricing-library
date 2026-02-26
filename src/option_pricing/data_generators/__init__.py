"""
Synthetic data helpers.

This package contains small utilities used by demos and diagnostics.

Notes
-----
Historically, this directory did not include an ``__init__.py`` and therefore was
not included in setuptools' ``find_packages`` output. Adding it ensures the
synthetic generators are packaged and importable from an installed wheel.
"""

from .Synthetic_Surface import (
    BadSVISmileCase,
    SyntheticSurface,
    generate_bad_svi_smile_case,
    generate_synthetic_surface,
)

__all__ = [
    "BadSVISmileCase",
    "SyntheticSurface",
    "generate_bad_svi_smile_case",
    "generate_synthetic_surface",
]
