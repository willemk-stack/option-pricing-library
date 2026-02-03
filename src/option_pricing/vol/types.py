# src/option_pricing/vol/types.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from option_pricing.typing import ArrayLike, FloatArray


class SmileSlice(Protocol):
    @property
    def T(self) -> float: ...

    def w_at(self, xq: ArrayLike) -> FloatArray: ...
    def iv_at(self, xq: ArrayLike) -> FloatArray: ...

    # domain so we can sample slices that don't have a native grid (e.g. SVI)
    @property
    def x_min(self) -> float: ...

    @property
    def x_max(self) -> float: ...


@runtime_checkable
class GridSmileSlice(SmileSlice, Protocol):
    @property
    def x(self) -> FloatArray: ...

    @property
    def w(self) -> FloatArray: ...
