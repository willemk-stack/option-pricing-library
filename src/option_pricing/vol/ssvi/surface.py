from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...typing import ArrayLike
from .math import (
    essvi_total_variance,
    essvi_total_variance_dk,
    essvi_total_variance_dk_dT,
    essvi_total_variance_dkk,
    essvi_total_variance_dT,
    essvi_total_variance_dTT,
    essvi_w_and_derivs,
)
from .models import ESSVIParamSurface


@dataclass(frozen=True, slots=True)
class ESSVISmileSlice:
    """Single-expiry analytic eSSVI smile slice in total-variance space."""

    T: float
    params: ESSVIParamSurface
    y_min: float = -2.5
    y_max: float = 2.5

    def __post_init__(self) -> None:
        self.params.validate(self.T)

    def w_at(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance(y=yq, params=self.params, T=self.T)

    def iv_at(self, yq: ArrayLike) -> np.ndarray:
        w = self.w_at(yq)
        return np.asarray(
            np.sqrt(np.maximum(w / np.float64(self.T), np.float64(0.0))),
            dtype=np.float64,
        )

    def dw_dy(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance_dk(y=yq, params=self.params, T=self.T)

    def d2w_dy2(self, yq: ArrayLike) -> np.ndarray:
        return essvi_total_variance_dkk(y=yq, params=self.params, T=self.T)


@dataclass(frozen=True, slots=True)
class ESSVIImpliedSurface:
    """Continuous analytic implied surface driven by an eSSVI parameter surface."""

    params: ESSVIParamSurface
    y_min: float = -2.5
    y_max: float = 2.5

    def slice(self, T: float) -> ESSVISmileSlice:
        return ESSVISmileSlice(
            T=float(T),
            params=self.params,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def w(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance(y=y, params=self.params, T=T)

    def iv(self, y: ArrayLike, T: float) -> np.ndarray:
        T = float(T)
        if T <= 0.0:
            raise ValueError("T must be > 0.")
        w = self.w(y=y, T=T)
        return np.asarray(
            np.sqrt(np.maximum(w / np.float64(T), np.float64(0.0))),
            dtype=np.float64,
        )

    def dw_dy(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dk(y=y, params=self.params, T=T)

    def d2w_dy2(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dkk(y=y, params=self.params, T=T)

    def dw_dT(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dT(y=y, params=self.params, T=T)

    def d2w_dy_dT(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dk_dT(y=y, params=self.params, T=T)

    def d2w_dT2(self, y: ArrayLike, T: float) -> np.ndarray:
        return essvi_total_variance_dTT(y=y, params=self.params, T=T)

    def w_and_derivs(
        self,
        y: np.ndarray,
        T: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return essvi_w_and_derivs(y=y, params=self.params, T=T)
