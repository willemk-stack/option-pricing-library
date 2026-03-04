"""Backward-compatible re-exports.

The shared implementation lives in :mod:`option_pricing.diagnostics._pde_compare.tables`.
"""

import warnings

from .._pde_compare.tables import *  # noqa: F401,F403

warnings.warn(
    "option_pricing.diagnostics.pde_vs_bs.tables is deprecated; "
    "import from option_pricing.diagnostics._pde_compare.tables instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
