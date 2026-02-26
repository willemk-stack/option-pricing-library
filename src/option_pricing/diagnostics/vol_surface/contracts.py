from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from option_pricing.typing import ArrayLike
from option_pricing.vol.vol_types import SmileSlice


class VolSurfaceLike(Protocol):
    @property
    def smiles(self) -> Sequence[SmileSlice]: ...

    def iv(self, K: ArrayLike, T: float) -> np.ndarray: ...


class VolSurfaceClass(Protocol):
    @classmethod
    def from_grid(
        cls,
        rows: list[tuple[float, float, float]],
        *,
        forward,
    ) -> VolSurfaceLike: ...


class Black76Module(Protocol):
    def black76_call_price_vec(
        self,
        *,
        forward: float,
        strikes: np.ndarray,
        sigma: np.ndarray,
        tau: float,
        df: float,
    ) -> np.ndarray: ...
