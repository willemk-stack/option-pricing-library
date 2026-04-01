"""
- typed Heston parameter object
- validation and bounds
- constrained/unconstrained transforms for calibration
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_POSITIVE_EPS = 1e-12
_RHO_CLIP_EPS = 1e-12


def _softplus(x: float) -> float:
    return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))


def _softplus_inv(y: float) -> float:
    y_safe = max(float(y), _POSITIVE_EPS)
    return float(y_safe + np.log(-np.expm1(-y_safe)))


def _atanh_clipped(x: float) -> float:
    clipped = float(np.clip(x, -1.0 + _RHO_CLIP_EPS, 1.0 - _RHO_CLIP_EPS))
    return float(np.arctanh(clipped))


@dataclass(frozen=True, slots=True)
class HestonParams:
    kappa: float
    vbar: float
    eta: float
    rho: float
    v: float

    def __post_init__(self) -> None:
        values = {
            "kappa": self.kappa,
            "vbar": self.vbar,
            "eta": self.eta,
            "rho": self.rho,
            "v": self.v,
        }
        for name, value in values.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if self.kappa <= 0.0:
            raise ValueError(
                rf"Mean reversion speed $\kappa$ must be positive. $\kappa$ = {self.kappa}"
            )
        if self.vbar < 0.0:
            raise ValueError(
                rf"Long-run mean variance $\bar{{v}}$ must be nonnegative. "
                rf"$\bar{{v}}$ = {self.vbar}"
            )
        if self.eta < 0.0:
            raise ValueError(
                rf"Volatility of vol $\eta$ must be nonnegative. $\eta$ = {self.eta}"
            )
        if self.v < 0.0:
            raise ValueError(
                rf"Initial variance $v$ must be nonnegative. $v$ = {self.v}"
            )
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(
                rf"Vol-spot correlation $\rho$ must be in [-1, 1]. "
                rf"Current value $\rho$ = {self.rho}"
            )

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.kappa, self.vbar, self.eta, self.rho, self.v], dtype=np.float64
        )

    def TransformToUnconstrained(self) -> np.ndarray:
        return np.array(
            [
                _softplus_inv(self.kappa),
                _softplus_inv(self.vbar),
                _softplus_inv(self.eta),
                _atanh_clipped(self.rho),
                _softplus_inv(self.v),
            ],
            dtype=np.float64,
        )

    @classmethod
    def TransformToConstrained(
        cls, raw: np.ndarray | list[float] | tuple[float, ...]
    ) -> HestonParams:
        raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw_arr.size != 5:
            raise ValueError(
                f"Expected 5 unconstrained parameters, got {raw_arr.size}."
            )
        if np.any(~np.isfinite(raw_arr)):
            raise ValueError("Unconstrained parameters must be finite.")

        return cls(
            kappa=_softplus(float(raw_arr[0])) + _POSITIVE_EPS,
            vbar=_softplus(float(raw_arr[1])),
            eta=_softplus(float(raw_arr[2])),
            rho=float(np.tanh(float(raw_arr[3]))),
            v=_softplus(float(raw_arr[4])),
        )

    transform_to_unconstrained = TransformToUnconstrained
    transform_to_constrained = TransformToConstrained
