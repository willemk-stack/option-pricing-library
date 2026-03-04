from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .diagnostics import SVIFitDiagnostics
from .math import jw_to_raw, raw_to_jw


@dataclass(frozen=True, slots=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_jw(self, T: float) -> JWParams:
        return raw_to_jw(self, T)


@dataclass(frozen=True, slots=True)
class JWParams:
    v: float
    psi: float
    p: float
    c: float
    v_tilde: float

    def to_raw(self, T: float) -> SVIParams:
        return jw_to_raw(self, T)


@dataclass(frozen=True, slots=True)
class SVIFitResult:
    params: SVIParams
    diag: SVIFitDiagnostics


@dataclass(frozen=True, slots=True)
class SVISmile:
    """Analytic SVI smile slice in total-variance space.

    Coordinate:
        y = ln(K / F(T))  (log-moneyness)

    Notes
    -----
    * The SVI functional form is defined for all real ``y``.
    * ``y_min`` / ``y_max`` are treated as the *recommended* domain for sampling
      (diagnostics, plots, calendar checks), not hard limits.
    """

    T: float
    params: SVIParams
    y_min: float = -1.25
    y_max: float = 1.25
    diagnostics: SVIFitDiagnostics | None = None

    def w_at(self, yq: Any) -> Any:
        yq_arr = np.asarray(yq, dtype=np.float64)
        yq_1d = np.atleast_1d(yq_arr)

        from .math import svi_total_variance

        out = svi_total_variance(yq_1d.astype(np.float64, copy=False), self.params)
        out = np.asarray(out, dtype=np.float64)

        if yq_arr.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(yq_arr.shape)

    def iv_at(self, yq: Any) -> Any:
        wq = self.w_at(yq)
        out = np.sqrt(np.maximum(wq / np.float64(self.T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)

    # ---- analytic derivatives in y (log-moneyness) ----
    def dw_dy(self, yq: Any) -> Any:
        yq_arr = np.asarray(yq, dtype=np.float64)
        yq_1d = np.atleast_1d(yq_arr)
        from .math import svi_total_variance_dy

        out = svi_total_variance_dy(yq_1d.astype(np.float64, copy=False), self.params)
        out = np.asarray(out, dtype=np.float64)
        if yq_arr.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(yq_arr.shape)

    def d2w_dy2(self, yq: Any) -> Any:
        yq_arr = np.asarray(yq, dtype=np.float64)
        yq_1d = np.atleast_1d(yq_arr)
        from .math import svi_total_variance_dyy

        out = svi_total_variance_dyy(yq_1d.astype(np.float64, copy=False), self.params)
        out = np.asarray(out, dtype=np.float64)
        if yq_arr.ndim == 0:
            return np.asarray(out[0], dtype=np.float64)
        return out.reshape(yq_arr.shape)
