"""Re-exports of shared table helpers."""

import warnings

from .._pde_compare.tables import *  # noqa: F401,F403

warnings.warn(
    "option_pricing.diagnostics.pde_vs_digital.tables is deprecated; "
    "import from option_pricing.diagnostics._pde_compare.tables instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
