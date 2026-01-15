from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

# Use this everywhere
type FloatArray = NDArray[np.floating]  # typing only
type FloatDType = np.float64  # runtime dtype only
type ScalarFn = Callable[[float], float]
type ArrayLike = float | np.ndarray | np.floating
