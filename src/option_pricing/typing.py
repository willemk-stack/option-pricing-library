from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

# Use this everywhere
# typing only
type FloatArray = NDArray[np.floating]  # typing only
type ArrayLike = float | np.ndarray | np.floating
type ScalarFn = Callable[[float], float]

# Runtime types
FloatDType = np.float64  # runtime dtype only
