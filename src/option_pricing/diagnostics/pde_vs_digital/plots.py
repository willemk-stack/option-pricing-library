"""Re-exports of shared plot helpers."""

import warnings

from .._pde_compare.plots import *  # noqa: F401,F403

warnings.warn(
    "option_pricing.diagnostics.pde_vs_digital.plots is deprecated; "
    "import from option_pricing.diagnostics._pde_compare.plots instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
