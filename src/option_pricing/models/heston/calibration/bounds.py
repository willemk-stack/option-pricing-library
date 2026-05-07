"""Calibration bounds and bounded transforms for Heston optimization.

This module keeps human-readable bounds in natural Heston parameter space and
provides the sigmoid transform used by bounded raw-space calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..charfunc import HESTON_ANALYTIC_JAC_ETA_MIN
from ..params import HESTON_PARAM_NAMES, HestonParams

BoundsPair = tuple[float, float]
_LOGIT_EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class HestonCalibrationBounds:
    """Practical calibration bounds in constrained Heston parameter space.

    Bounds are ordered as [kappa, vbar, eta, rho, v] to match
    HESTON_PARAM_NAMES and the Heston calibration transform.

    Notes
    -----
    These are practical optimizer limits, not no-arbitrage constraints.
    The Feller condition is deliberately not enforced here. It should be handled
    separately as a diagnostic or optional soft regularization penalty.
    """

    kappa: BoundsPair = (0.05, 20.0)
    vbar: BoundsPair = (1e-6, 1.0)
    eta: BoundsPair = (1e-4, 5.0)
    rho: BoundsPair = (-0.999, 0.999)
    v: BoundsPair = (1e-6, 1.0)

    def __post_init__(self) -> None:
        for name, pair in zip(HESTON_PARAM_NAMES, self.as_pairs(), strict=True):
            if len(pair) != 2:
                raise ValueError(f"{name} bounds must be a length-2 pair.")

            lower, upper = float(pair[0]), float(pair[1])

            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"{name} bounds must be finite.")

            if lower >= upper:
                raise ValueError(
                    f"{name} lower bound must be strictly below upper bound. "
                    f"Got ({lower}, {upper})."
                )

        kappa_lo, _ = self.kappa
        vbar_lo, _ = self.vbar
        eta_lo, _ = self.eta
        rho_lo, rho_hi = self.rho
        v_lo, _ = self.v

        if kappa_lo <= 0.0:
            raise ValueError("kappa lower bound must be positive.")
        if vbar_lo < 0.0:
            raise ValueError("vbar lower bound must be nonnegative.")
        if eta_lo < 0.0:
            raise ValueError("eta lower bound must be nonnegative.")
        if v_lo < 0.0:
            raise ValueError("v lower bound must be nonnegative.")
        if not (-1.0 < rho_lo < rho_hi < 1.0):
            raise ValueError("rho bounds must lie strictly inside (-1, 1).")

    def as_pairs(
        self,
    ) -> tuple[BoundsPair, BoundsPair, BoundsPair, BoundsPair, BoundsPair]:
        """Return bounds in HESTON_PARAM_NAMES order."""
        return (self.kappa, self.vbar, self.eta, self.rho, self.v)

    def lower_params(self) -> HestonParams:
        """Return the lower constrained bound as a validated parameter object."""
        return HestonParams(
            kappa=float(self.kappa[0]),
            vbar=float(self.vbar[0]),
            eta=float(self.eta[0]),
            rho=float(self.rho[0]),
            v=float(self.v[0]),
        )

    def upper_params(self) -> HestonParams:
        """Return the upper constrained bound as a validated parameter object."""
        return HestonParams(
            kappa=float(self.kappa[1]),
            vbar=float(self.vbar[1]),
            eta=float(self.eta[1]),
            rho=float(self.rho[1]),
            v=float(self.v[1]),
        )

    def lower_array(self) -> np.ndarray:
        """Return lower constrained bounds in calibration order."""
        return self.lower_params().as_array()

    def upper_array(self) -> np.ndarray:
        """Return upper constrained bounds in calibration order."""
        return self.upper_params().as_array()

    def raw_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return optimizer bounds in raw/transformed coordinates."""
        return raw_bounds_from_constrained_bounds(self)

    def require_analytic_jacobian_compatible(self) -> HestonCalibrationBounds:
        """Raise if these bounds allow unsupported analytic-Jacobian eta.

        The eta floor is a numerical policy for the Cui analytic-gradient
        formulas, not a financial admissibility condition. Price-only Heston
        diagnostics may still evaluate deterministic-limit prices below this
        floor.
        """
        eta_lo, _ = self.eta
        if float(eta_lo) < HESTON_ANALYTIC_JAC_ETA_MIN:
            raise ValueError(
                "Analytic Heston calibration Jacobians require eta lower bound "
                f">= {HESTON_ANALYTIC_JAC_ETA_MIN:g}. Price-only deterministic-limit "
                "pricing near eta=0 is separate; pass use_analytic_jac=False or "
                "raise the bounded calibration eta lower bound."
            )
        return self


def _as_raw_vector(raw: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if raw_arr.size != 5:
        raise ValueError(f"Expected 5 bounded raw parameters, got {raw_arr.size}.")
    if np.any(~np.isfinite(raw_arr)):
        raise ValueError("Bounded raw parameters must be finite.")
    return raw_arr


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=np.float64)
    flat = x_arr.reshape(-1)
    out = np.empty_like(flat, dtype=np.float64)

    positive = flat >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-flat[positive]))
    exp_x = np.exp(flat[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)

    result = out.reshape(x_arr.shape)
    if x_arr.ndim == 0:
        return float(result.item())
    return result


def _bounds_arrays(bounds: HestonCalibrationBounds) -> tuple[np.ndarray, np.ndarray]:
    return bounds.lower_array(), bounds.upper_array()


def transform_to_bounded_constrained(
    raw: np.ndarray | list[float] | tuple[float, ...],
    bounds: HestonCalibrationBounds,
) -> HestonParams:
    """Map raw optimizer variables into finite Heston calibration bounds."""
    raw_arr = _as_raw_vector(raw)
    lower, upper = _bounds_arrays(bounds)
    z = np.asarray(_sigmoid(raw_arr), dtype=np.float64)
    values = lower + (upper - lower) * z

    return HestonParams(
        kappa=float(values[0]),
        vbar=float(values[1]),
        eta=float(values[2]),
        rho=float(values[3]),
        v=float(values[4]),
    )


def transform_to_bounded_unconstrained(
    params: HestonParams,
    bounds: HestonCalibrationBounds,
) -> np.ndarray:
    """Map bounded Heston parameters into raw optimizer variables."""
    values = params.as_array()
    lower, upper = _bounds_arrays(bounds)
    outside = (values < lower) | (values > upper)
    if np.any(outside):
        idx = int(np.flatnonzero(outside)[0])
        name = HESTON_PARAM_NAMES[idx]
        raise ValueError(
            f"{name}={values[idx]} lies outside bounded calibration interval "
            f"[{lower[idx]}, {upper[idx]}]."
        )

    z = (values - lower) / (upper - lower)
    z = np.clip(z, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.asarray(np.log(z / (1.0 - z)), dtype=np.float64)


def bounded_transform_jac_diag_from_raw(
    raw: np.ndarray | list[float] | tuple[float, ...],
    bounds: HestonCalibrationBounds,
) -> np.ndarray:
    """Return ``d constrained_parameter / d raw`` for the bounded transform."""
    raw_arr = _as_raw_vector(raw)
    lower, upper = _bounds_arrays(bounds)
    z = np.asarray(_sigmoid(raw_arr), dtype=np.float64)
    return np.asarray((upper - lower) * z * (1.0 - z), dtype=np.float64)


def bounded_transform_fraction(
    params: HestonParams,
    bounds: HestonCalibrationBounds,
) -> np.ndarray:
    """Return each parameter's fractional position within the bounded box."""
    values = params.as_array()
    lower, upper = _bounds_arrays(bounds)
    return np.asarray((values - lower) / (upper - lower), dtype=np.float64)


def raw_bounds_from_constrained_bounds(
    bounds: HestonCalibrationBounds,
) -> tuple[np.ndarray, np.ndarray]:
    """Map constrained Heston bounds into raw optimizer bounds.

    Parameters
    ----------
    bounds
        Practical calibration bounds expressed in natural Heston parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper bounds suitable for scipy.optimize.least_squares when
        calibration uses parameter_transform="unconstrained".
    """
    lower = bounds.lower_params().transform_to_unconstrained()
    upper = bounds.upper_params().transform_to_unconstrained()

    if np.any(lower >= upper):
        raise ValueError(
            "Raw calibration bounds are invalid after transformation. "
            f"lower={lower}, upper={upper}"
        )

    return lower, upper


def default_heston_calibration_bounds() -> HestonCalibrationBounds:
    """Return the default practical Heston calibration bounds."""
    return HestonCalibrationBounds()
