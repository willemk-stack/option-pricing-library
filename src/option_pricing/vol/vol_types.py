# src/option_pricing/vol/types.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from option_pricing.typing import ArrayLike, FloatArray


class SmileSlice(Protocol):
    """
    A single-expiry smile slice represented in total variance space.

    Coordinate:
        y = ln(K / F(T))   (log-moneyness)

    Required:
        - T: expiry
        - w_at(y): total variance
        - iv_at(y): implied vol
        - y_min: lowerbound
        - y_max:
    """

    @property
    def T(self) -> float: ...

    @property
    def y_min(self) -> float: ...

    @property
    def y_max(self) -> float: ...

    def w_at(self, xq: ArrayLike) -> FloatArray: ...
    def iv_at(self, xq: ArrayLike) -> FloatArray: ...


@runtime_checkable
class GridSmileSlice(SmileSlice, Protocol):
    @property
    def y(self) -> FloatArray: ...

    @property
    def w(self) -> FloatArray: ...
